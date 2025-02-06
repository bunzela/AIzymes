import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer, AutoModel
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
import logging
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
from src.tools import reset_cuda, setup_torch
import matplotlib

# Set the logging level to suppress warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

print("### PLM trainer loaded. ###")

class ProteinRegressor(nn.Module):
    def __init__(self, esm_model, p_loss, num_features):
        super(ProteinRegressor, self).__init__()
        self.esm_model = esm_model
        self.dropout = nn.Dropout(p=p_loss)
        self.regressor = nn.Linear(esm_model.config.hidden_size, num_features)  # Output multiple continuous values

    def forward(self, input_ids, attention_mask):
        outputs = self.esm_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        activities = self.regressor(cls_output)
        return activities
    
class ProteinDataset(Dataset):
    def __init__(self, sequences, activities, tokenizer):
        self.sequences = sequences
        self.activities = activities
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        activities = self.activities[idx]
        inputs = self.tokenizer(sequence, padding='max_length', truncation=True, return_tensors="pt")
        return {**inputs, 'activities': torch.tensor(activities, dtype=torch.float)}
    
class PLM_trainer():
        
    def __init__(self, 
                 verbose         = True,
                 output_folder   = './AIzymes_resi14'):

        self.output_folder = output_folder
        self.verbose = verbose
        os.makedirs(output_folder, exist_ok=True)

    def load_dataset(self, 
                     df_path='./data/all_scores_pooled_cut.csv', 
                     scores=['total_score'], 
                     labels=['score_taken_from', 'design_method', 'cat_resn', 'cat_resi', 'parent_index', 'generation', 'mutations', 'landscape'], 
                     cat_resi=None, 
                     select_unique=True,
                     normalize='both', 
                     scores_invert= ['total_score','interface_score','total_potential','interface_potential']):
        
        self.df_path         = df_path
        self.scores          = scores
        self.labels          = labels
        self.cat_resi        = cat_resi
        self.select_unique   = select_unique
        self.normalize       = normalize
        self.scores_invert   = scores_invert

        self.df, self.lengths, self.max_len, self.sequences = self.read_datasets_csv()
        self.df = self.normalize_scores(self.df)

    def normalize_scores(self, df):

        for score in self.scores:
            score_list = df[score].to_numpy().astype(np.float32)
            if self.normalize == 'minmax':
                df[f'norm_{score}'] = self.min_max(score_list)
            elif self.normalize == 'both':
                score_list = self.min_max(score_list)
                df[f'norm_{score}'] = self.z_score(score_list)
            elif self.normalize == 'z_score':
                df[f'norm_{score}'] = self.z_score(score_list)
        
        for score_invert in self.scores_invert:
            if f'norm_{score_invert}' in df.columns:
                df[f'norm_{score_invert}'] = -df[f'norm_{score_invert}']

        print('### Data normalized. ###')
        
        return df
    
    def read_datasets_csv(self):
        """
        Reads one or multiple CSV files into a single DataFrame.
        """

        if isinstance(self.df_path, str):
            self.df_path = [self.df_path]  # Convert to list if a single path is provided

        df_list = []

        for file in self.df_path:
            print(f"Loading dataset: {file}")
            temp_df = pd.read_csv(file)

            if 'sequence' not in temp_df.columns:
                print(f"Warning: 'sequence' column missing in {file}. Skipping this file.")
                continue

            temp_df = temp_df[temp_df['sequence'].notnull()]
            df_list.append(temp_df)

        # Combine all DataFrames
        df = pd.concat(df_list, ignore_index=True)

        # Handle unique sequences
        if self.select_unique:
            score_df = df.groupby('sequence')[self.scores].mean().reset_index()
            label_df = df.drop_duplicates('sequence')[self.labels + ['sequence']]
            df = pd.merge(score_df, label_df, on='sequence')

        if self.cat_resi is not None:
            df = df[df['cat_resi'] == self.cat_resi]

        sequences = df['sequence'].tolist()
        lengths = np.array([len(seq) for seq in sequences])
        max_len = np.max(lengths)

        print(f'### {len(self.df_path)} files loaded into one dataset. ###')

        return df, lengths, max_len, sequences
    
    def train_PLM(self, 
                  epochs               = 3, 
                  esm2_model_name      = None,
                  p_loss               = 0.1, 
                  liveplot             = False, 
                  overwrite            = False,
                  validation           = "test_train",
                  validation_landscape = None,
                  test_size            = 0.2,
                  stratify_nan         = True,
                  print_testtrain      = True,
                  figure_name          = ""):
        
        self.stratify_nan    = stratify_nan
        self.print_testtrain = print_testtrain
        self.epochs = epochs
        self.esm2_model_name = esm2_model_name
        self.p_loss = p_loss
        self.liveplot = liveplot
        self.figure_name = figure_name
        self.overwrite = overwrite
     
        if validation == "test_train":
 
            # Stratify based on binned score values
            stratify_bins = pd.qcut(self.df[self.scores[0]], q=4, duplicates="drop", labels=False)  # 4 bins

            # Perform train-test split with stratification
            self.train_df, self.test_df = train_test_split(
                self.df, 
                test_size=test_size, 
                random_state=42, 
                stratify=stratify_bins
            )

            if print_testtrain:
                print("train_df")
                display(self.train_df)
                print("test_df")
                display(self.test_df)

        if validation == "landscape":

            self.train_df = self.df[self.df["landscape"] != validation_landscape]
            self.test_df = self.df[self.df["landscape"] == validation_landscape]

            if print_testtrain:
                print("train_df")
                display(self.train_df)
                print("test_df")
                display(self.test_df)

        if esm2_model_name is not None:
            self.plm_model_name = esm2_model_name
            self.tokenize_esm2_model(p_loss)

        if not self.overwrite:
            if os.path.isfile(f'{self.output_folder}/plm_self_{self.file_title()}.pkl'):
                print(f'{self.output_folder}/plm_self_{self.file_title()}.pkl exists! Stopping calculation')
                return

        if liveplot:
            matplotlib.use('TkAgg')

        self.trainer()
        print(f'PLM {self.plm_model_name} trained for {", ".join([f"norm_{score}" for score in self.scores])}')
        self.eval_test_train()
        self.save_self_to_file()
        self.plot_learning()

        # Save the model
        torch.save(self.model.state_dict(), f'{self.output_folder}/model_{self.file_title()}.pt')
        torch.cuda.empty_cache()

    def loss_fn_with_nan_handling(self, predictions, targets):
        """
        Custom loss function to handle NaNs.
        It computes the loss only for non-NaN values.
        
        :param predictions: Model output predictions
        :param targets: Ground truth target values
        :return: Loss value computed only for non-NaN values
        """
        # Create a mask to identify non-NaN values in targets
        mask = ~torch.isnan(targets)
        
        # Apply the mask to both predictions and targets
        masked_predictions = predictions[mask]
        masked_targets = targets[mask]

        # Compute the loss only for non-NaN values
        if masked_predictions.numel() > 0:  # Ensure there are valid (non-NaN) values
            loss = nn.MSELoss()(masked_predictions, masked_targets)
        else:
            loss = torch.tensor(0.0, requires_grad=True)  # Return zero loss if no valid values
        return loss
    
    def trainer(self):
        self.train_loss = []
        self.test_loss = []
        self.device, self.dtype, self.SMOKE_TEST = setup_torch()

        train_sequences = self.train_df['sequence'].tolist()
        train_activities = self.train_df[[f'norm_{score}' for score in self.scores]].values.tolist()
        test_sequences = self.test_df['sequence'].tolist()
        test_activities = self.test_df[[f'norm_{score}' for score in self.scores]].values.tolist()

        train_dataset = ProteinDataset(train_sequences, train_activities, self.tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_dataset = ProteinDataset(test_sequences, test_activities, self.tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        optimizer = optim.Adam(self.model.parameters(), lr=5e-5)
        self.model.to(self.device)

        if self.liveplot:
            # Set up live plot with two subplots
            plt.ion()  # Turn on interactive mode
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Loss plot
            train_loss_line, = ax1.plot([], [], label='Train Loss', color='r')
            test_loss_line, = ax1.plot([], [], label='Test Loss', color='b')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.set_xlim(0, self.epochs)
            ax1.set_ylim(0, 1)  # Adjust based on initial expectations

            # Predicted vs Actual plot
            scatter_train = ax2.scatter([], [], color='r', label='Train Predictions')
            scatter_test = ax2.scatter([], [], color='b', label='Test Predictions')
            ax2.set_xlabel('True Activity')
            ax2.set_ylabel('Predicted Activity')
            ax2.legend()
            ax2.plot([0, 1], [0, 1], transform=ax2.transAxes, color='gray', linestyle='--')  # Diagonal for perfect predictions

        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            total_train_loss = 0
            train_preds, train_acts = [], []

            # Training loop
            for batch in train_dataloader:
                input_ids = batch['input_ids'].squeeze(1).to(self.device)
                attention_mask = batch['attention_mask'].squeeze(1).to(self.device)
                activities = batch['activities'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.loss_fn_with_nan_handling(outputs, activities)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                train_preds.extend(outputs.detach().cpu().numpy())
                train_acts.extend(activities.detach().cpu().numpy())

            avg_train_loss = total_train_loss / len(train_dataloader)
            self.train_loss.append(avg_train_loss)

            # Testing loop
            self.model.eval()
            total_test_loss = 0
            test_preds, test_acts = [], []
            with torch.no_grad():
                for batch in test_dataloader:
                    input_ids = batch['input_ids'].squeeze(1).to(self.device)
                    attention_mask = batch['attention_mask'].squeeze(1).to(self.device)
                    activities = batch['activities'].to(self.device)

                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = self.loss_fn_with_nan_handling(outputs, activities)
                    total_test_loss += loss.item()

                    test_preds.extend(outputs.cpu().numpy())
                    test_acts.extend(activities.cpu().numpy())

            avg_test_loss = total_test_loss / len(test_dataloader)
            self.test_loss.append(avg_test_loss)

            # Update the loss plot
            if self.liveplot:
                train_loss_line.set_xdata(range(1, len(self.train_loss) + 1))
                train_loss_line.set_ydata(self.train_loss)
                test_loss_line.set_xdata(range(1, len(self.test_loss) + 1))
                test_loss_line.set_ydata(self.test_loss)

                # Adjust the y-limits dynamically based on the loss values
                ax1.set_ylim(0, max(max(self.train_loss), max(self.test_loss)) * 1.1)

                # Update the Predicted vs Actual plot
                ax2.clear()  # Clear previous points
                ax2.scatter(train_acts, train_preds, color='r', alpha=0.5, label='Train Predictions')
                ax2.scatter(test_acts, test_preds, color='b', alpha=0.5, label='Test Predictions')
                ax2.set_xlabel('True Activity')
                ax2.set_ylabel('Predicted Activity')
                ax2.legend()
                ax2.plot([np.min(test_acts), np.max(test_acts)], [np.min(test_acts), np.max(test_acts)], color='gray', linestyle='--')  # Diagonal line

                # Redraw both plots
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)  # Allow plot to update

            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} completed.\ttrain loss: {avg_train_loss:.2f}.\ttest loss: {avg_test_loss:.2f}")

        if self.liveplot:
            # Turn off interactive mode once training is done
            plt.ioff()
            plt.show()

    def tokenize_esm2_model(self, p_loss):
        
        self.esm_model = EsmModel.from_pretrained(self.esm2_model_name)
        self.tokenizer = EsmTokenizer.from_pretrained(self.esm2_model_name)
        self.model = ProteinRegressor(self.esm_model, p_loss, num_features=len(self.scores))

    def min_max(self, scores):
        return (scores - np.nanmin(scores)) / (np.nanmax(scores) - np.nanmin(scores))
    
    def z_score(self, scores):
        return (scores - np.nanmean(scores)) / np.nanstd(scores)
    
    def file_title(self):
        return f'model_{self.esm2_model_name.split("/")[-1][:-6]}_scores_{"_".join(self.scores)}_cat_{self.cat_resi}_epochs_{self.epochs}_ploss_{self.p_loss}'
    
    def save_self_to_file(self):
        with open(f'{self.output_folder}/plm_self_{self.file_title()}.pkl', 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load_self_from_file(self, epochs, esm2_model_name, cat_resi, p_loss, df_path, scores):
        self.epochs = epochs
        self.esm2_model_name = esm2_model_name
        self.cat_resi = cat_resi
        self.p_loss = p_loss
        self.df_path = df_path
        self.scores = scores
        
        if not os.path.isfile(f'{self.output_folder}/plm_self_{self.file_title()}.pkl'):
            print(f'### {self.output_folder}/plm_self_{self.file_title()}.pkl does not exist. ###' )
            return
        
        with open(f'{self.output_folder}/plm_self_{self.file_title()}.pkl', 'rb') as f:
            self.__dict__.update(pickle.load(f))
        print(f'All variables loaded from {self.output_folder}/plm_self_{self.file_title()}.pkl')

        return self.df

    def eval_test_train(self):
        self.model.eval()

        train_sequences = self.train_df['sequence'].tolist()
        train_activities = self.train_df[[f'norm_{score}' for score in self.scores]].values
        test_sequences = self.test_df['sequence'].tolist()
        test_activities = self.test_df[[f'norm_{score}' for score in self.scores]].values

        train_dataset = ProteinDataset(train_sequences, train_activities, self.tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        test_dataset = ProteinDataset(test_sequences, test_activities, self.tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        train_preds, test_preds = [], []

        with torch.no_grad():
            for batch in train_dataloader:
                input_ids = batch['input_ids'].squeeze(1).to(self.device)
                attention_mask = batch['attention_mask'].squeeze(1).to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                train_preds.extend(outputs.cpu().numpy())

            for batch in test_dataloader:
                input_ids = batch['input_ids'].squeeze(1).to(self.device)
                attention_mask = batch['attention_mask'].squeeze(1).to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                test_preds.extend(outputs.cpu().numpy())

        self.train_activities = train_activities
        self.train_preds = np.array(train_preds)
        self.test_activities = test_activities
        self.test_preds = np.array(test_preds)

        self.train_metrics = {}
        self.test_metrics = {}

        for i, score in enumerate(self.scores):
            train_act = self.train_activities[:, i]
            train_pred = self.train_preds[:, i]
            test_act = self.test_activities[:, i]
            test_pred = self.test_preds[:, i]

            # Filter out NaNs from both predictions and ground truth
            train_mask = ~np.isnan(train_act) & ~np.isnan(train_pred)
            test_mask = ~np.isnan(test_act) & ~np.isnan(test_pred)

            filtered_train_act = train_act[train_mask]
            filtered_train_pred = train_pred[train_mask]
            filtered_test_act = test_act[test_mask]
            filtered_test_pred = test_pred[test_mask]

            # Check if the filtered arrays are non-empty before calculating metrics
            if filtered_train_act.size > 0 and filtered_train_pred.size > 0:
                self.train_metrics[score] = {
                    'slope': stats.linregress(filtered_train_act, filtered_train_pred).slope,
                    'intercept': stats.linregress(filtered_train_act, filtered_train_pred).intercept,
                    'r_value': stats.linregress(filtered_train_act, filtered_train_pred).rvalue,
                    'p_value': stats.linregress(filtered_train_act, filtered_train_pred).pvalue,
                    'std_err': stats.linregress(filtered_train_act, filtered_train_pred).stderr,
                    'rmse': np.sqrt(mean_squared_error(filtered_train_act, filtered_train_pred))
                }
            else:
                print(f"Skipping metrics calculation for train score {score} due to empty data.")

            if filtered_test_act.size > 0 and filtered_test_pred.size > 0:
                self.test_metrics[score] = {
                    'slope': stats.linregress(filtered_test_act, filtered_test_pred).slope,
                    'intercept': stats.linregress(filtered_test_act, filtered_test_pred).intercept,
                    'r_value': stats.linregress(filtered_test_act, filtered_test_pred).rvalue,
                    'p_value': stats.linregress(filtered_test_act, filtered_test_pred).pvalue,
                    'std_err': stats.linregress(filtered_test_act, filtered_test_pred).stderr,
                    'rmse': np.sqrt(mean_squared_error(filtered_test_act, filtered_test_pred))
                }
            else:
                print(f"Skipping metrics calculation for test score {score} due to empty data.")

    def plot_learning(self, top_p=0.8):

        if not os.path.isfile(f'{self.output_folder}/plm_self_{self.file_title()}.pkl'):
            print(f'### {self.output_folder}/plm_self_{self.file_title()}.pkl does not exist. ###')
            return

        fig, axs = plt.subplots(2, len(self.scores), figsize=(4 * len(self.scores), 8))
        if len(self.scores) == 1: axs = np.array([[axs[0]], [axs[1]]])  # Ensure axs is always 2D

        c_test = "b"
        c_train = "r"

        for i, score in enumerate(self.scores):

            # Get min and max values of the dataset
            min_val = min(np.nanmin(self.train_activities[:, i]), np.nanmin(self.train_preds[:, i]),
                        np.nanmin(self.test_activities[:, i]), np.nanmin(self.test_preds[:, i]))
            max_val = max(np.nanmax(self.train_activities[:, i]), np.nanmax(self.train_preds[:, i]),
                        np.nanmax(self.test_activities[:, i]), np.nanmax(self.test_preds[:, i]))

            # Filter the top {top} of the training data
            top_n_train = int(len(self.train_activities) * top_p)
            top_n_test = int(len(self.test_activities) * top_p)

            train_idx = np.argsort(self.train_activities[:, i])[-top_n_train:]
            test_idx = np.argsort(self.test_activities[:, i])[-top_n_test:]

            train_filtered_activities = self.train_activities[train_idx, i]
            train_filtered_preds = self.train_preds[train_idx, i]
            test_filtered_activities = self.test_activities[test_idx, i]
            test_filtered_preds = self.test_preds[test_idx, i]

            # Remove NaNs from filtered activities and predictions
            valid_train_idx = ~np.isnan(train_filtered_activities) & ~np.isnan(train_filtered_preds)
            valid_test_idx = ~np.isnan(test_filtered_activities) & ~np.isnan(test_filtered_preds)

            train_filtered_activities = train_filtered_activities[valid_train_idx]
            train_filtered_preds = train_filtered_preds[valid_train_idx]
            test_filtered_activities = test_filtered_activities[valid_test_idx]
            test_filtered_preds = test_filtered_preds[valid_test_idx]

            # Recalculate RMSE for the top {top}
            if len(train_filtered_activities) > 0 and len(train_filtered_preds) > 0:
                self.train_metrics[score][f"train_rmse_top{top_p}"] = np.sqrt(mean_squared_error(train_filtered_activities, train_filtered_preds))
            else:
                print(f"Skipping train RMSE calculation for {score} due to empty data.")

            if len(test_filtered_activities) > 0 and len(test_filtered_preds) > 0:
                self.test_metrics[score][f"test_rmse_top{top_p}"] = np.sqrt(mean_squared_error(test_filtered_activities, test_filtered_preds))
            else:
                print(f"Skipping test RMSE calculation for {score} due to empty data.")

            # Top row: Scatter plot for test and train datasets
            axs[0, i].plot([min_val, max_val], [min_val, max_val], 'k', alpha=0.5, zorder=100)

            axs[0, i].scatter(self.train_activities[:, i], self.train_preds[:, i], alpha=0.5, c=c_train)
            line = self.train_metrics[score]['slope'] * np.array([min_val, max_val]) + self.train_metrics[score]['intercept']
            axs[0, i].plot([min_val, max_val], line, c=c_train, zorder=10, 
                        label=f'train R2={self.train_metrics[score]["r_value"]**2:.2f}, rmse_top{int(top_p*100)}={self.train_metrics[score][f"train_rmse_top{top_p}"]:.2f}')

            axs[0, i].scatter(self.test_activities[:, i], self.test_preds[:, i], alpha=0.5, c=c_test)
            line = self.test_metrics[score]['slope'] * np.array([min_val, max_val]) + self.test_metrics[score]['intercept']
            axs[0, i].plot([min_val, max_val], line, c=c_test, zorder=20, 
                        label=f'test R2={self.test_metrics[score]["r_value"]**2:.2f}, rmse_top{int(top_p*100)}={self.test_metrics[score][f"test_rmse_top{top_p}"]:.2f}')

            axs[0, i].set_xlim([min_val, max_val])
            axs[0, i].set_ylim([min_val, max_val])       
            axs[0, i].set_xlabel('True Activity')
            axs[0, i].set_ylabel('Predicted Activity')
            axs[0, i].set_title(score)
            axs[0, i].legend()

            # Bottom row: Loss vs. Epoch plot
            axs[1, i].plot(range(1, len(self.train_loss) + 1), self.train_loss, label='train_loss', c=c_train, zorder=1)
            axs[1, i].plot(range(1, len(self.train_loss) + 1), self.test_loss, label='test_loss', c=c_test, zorder=2)
            axs[1, i].set_xlabel('Epoch')
            axs[1, i].set_ylabel('Loss')
            axs[1, i].set_xlim(0, len(self.train_loss))
            axs[1, i].set_ylim(bottom=0)
            axs[1, i].legend()

        plt.suptitle(self.file_title().replace('_', ' '))
        plt.tight_layout()
        if self.figure_name == "":
            plt.savefig(f"{self.output_folder}/{self.file_title()}.png")
        else:
            plt.savefig(f"{self.output_folder}/{self.figure_name}_{self.file_title()}.png")
        plt.show()

        # Collect data
        filename = f"{self.output_folder}/results.pkl"  

        if os.path.isfile(filename):
            df = pd.read_pickle(filename)
        else:
            df = pd.DataFrame()
            df['file_title'] = None

        data = {
            'file_title': self.file_title,
            'esm2_model_name': self.esm2_model_name,
            'cat_resi': self.cat_resi,
            'epochs': self.epochs,
            'p_loss': self.p_loss,
            'df_path': self.df_path
        }

        for score in self.scores:
            data[f'test_r_value_{score}'] = self.test_metrics[score]['r_value']
            data[f'test_p_value_{score}'] = self.test_metrics[score]['p_value']
            data[f'test_rmse_{score}'] = self.test_metrics[score]['rmse']
            data[f'test_rmse_top{top_p}_{score}'] = self.test_metrics[score][f"test_rmse_top{top_p}"]

        # Convert data to a DataFrame
        new_data = pd.DataFrame([data])

        # Check if file_title already exists in the DataFrame
        if self.file_title in df['file_title'].values:
            df.loc[df['file_title'] == self.file_title, :] = new_data.values
        else:
            df = pd.concat([df, new_data], ignore_index=True)

        df.to_pickle(filename)

def plot_summary(output_folder,scores=None,models=None,top_p=0.8):

    filename=f"{output_folder}/results.pkl"
    df = pd.read_pickle(filename)
    df = df.dropna()

    if scores == None:
        scores = ['_'.join(col.split('_')[2:]) for col in df.columns if col.startswith(f'test_rmse_top{top_p}')]
    if models == None:
        models = sorted(set(df['esm2_model_name']))
    p_losses = sorted(set(df['p_loss']))
    
    fig, axs = plt.subplots(1, len(scores), figsize=(4 * len(scores), 4))
    if len(scores) == 1: axs = np.array([[axs[0]], [axs[1]]])  # Ensure axs is always 2D

    for i, score in enumerate(scores):

        results = np.zeros((len(p_losses), len(models)))

        for model_idx, model in enumerate(models):
            for p_loss_idx, p_loss in enumerate(p_losses):
                # Filter the DataFrame for the specific model and p_loss
                filtered_df = df[(df['esm2_model_name'] == model) & (df['p_loss'] == p_loss)]

                if not filtered_df.empty:
                    results[p_loss_idx, model_idx] = filtered_df[f'test_rmse_{score}'].values[0]
                else:
                    results[p_loss_idx, model_idx] = np.nan

        # Create a heatmap
        sns.heatmap(results, annot=True, fmt=".2f", cmap="Blues_r", xticklabels=[model.split('/')[-1][:-6] for model in models], yticklabels=p_losses, ax=axs[i], 
                    vmin=0.18, vmax=0.33, cbar=False)
        axs[i].set_title(f'{score} RMSE top{int(top_p*100)}')
        axs[i].set_ylabel('p_loss')

    plt.suptitle('One model trained combined for all scores')
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)  # Adjust to make room for the suptitle
    plt.show()

    filename=f"{output_folder}/results.pkl"
    df = pd.read_pickle(filename)
    df = df[df.isna().any(axis=1)]

    if scores == None:
        scores = ['_'.join(col.split('_')[2:]) for col in df.columns if col.startswith(f'test_rmse_top{top_p}_')]
    if models == None:
        models = sorted(set(df['esm2_model_name']))
    p_losses = sorted(set(df['p_loss']))

    fig, axs = plt.subplots(1, len(scores), figsize=(4 * len(scores), 4))
    if len(scores) == 1: axs = np.array([[axs[0]], [axs[1]]])  # Ensure axs is always 2D

    for i, score in enumerate(scores):

        results = np.zeros((len(p_losses), len(models)))

        for model_idx, model in enumerate(models):
            for p_loss_idx, p_loss in enumerate(p_losses):

                # Filter the DataFrame for the specific model and p_loss               
                filtered_df = df[(df['esm2_model_name'] == model) & (df['p_loss'] == p_loss)]
                filtered_df = filtered_df.dropna(subset=[f'test_rmse_{score}'])

                if not filtered_df.empty:
                    results[p_loss_idx, model_idx] = filtered_df[f'test_rmse_{score}'].values[0]
                else:
                    results[p_loss_idx, model_idx] = np.nan

        # Create a heatmap
        sns.heatmap(results, annot=True, fmt=".2f", cmap="Blues_r", xticklabels=[model.split('/')[-1][:-6] for model in models], yticklabels=p_losses, ax=axs[i], 
                    vmin=0.18, vmax=0.33, cbar=False)
        axs[i].set_title(f'{score} RMSE top{int(top_p*100)}')
        axs[i].set_ylabel('p_loss')

    plt.suptitle('Models trained individually for each score')
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)  # Adjust to make room for the suptitle
    plt.show()
