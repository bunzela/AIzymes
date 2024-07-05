import os
import pandas as pd
import numpy as np
import torch
from transformers import EsmModel, EsmTokenizer
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
from src.tools import reset_cuda, setup_torch
import pickle
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import logging
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
from sklearn.metrics import mean_squared_error
import os 

# Set the logging level to suppress warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

print("### PLM trainer loaded. ###")

class ProteinRegressor(nn.Module):
    def __init__(self, esm_model, p_loss):
        super(ProteinRegressor, self).__init__()
        self.esm_model = esm_model
        self.dropout = nn.Dropout(p=p_loss)  
        self.regressor = nn.Linear(esm_model.config.hidden_size, 1)  # Output is a single continuous value

    def forward(self, input_ids, attention_mask):
        outputs = self.esm_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)  
        activity = self.regressor(cls_output)
        return activity

class ProteinDataset(Dataset):
    def __init__(self, sequences, activities, tokenizer):
        self.sequences = sequences
        self.activities = activities
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        activity = self.activities[idx]
        inputs = self.tokenizer(sequence, padding='max_length', truncation=True, return_tensors="pt")
        return {**inputs, 'activity': torch.tensor(activity, dtype=torch.float)}
    
class PLM_trainer():
        
    def __init__(self, 
                 verbose         = True,
                 output_folder   = './AIzymes_resi14'):

        # Set variables
        self.output_folder   = output_folder
        self.verbose         = verbose

        # Make folder
        os.makedirs(output_folder, exist_ok=True)

    def load_dataset(self,
                     df_path = './data/all_scores_pooled_cut.csv',
                     score           = 'total_score',
                     labels          = ['score_taken_from', 'design_method', 'cat_resn', 'cat_resi', 'parent_index', 'generation', 'mutations'],
                     cat_resi        = 14,
                     select_unique   = True,
                     normalize       = 'both',
                     test_size       = 0.2):

        # Set variables
        self.df_path         = df_path
        self.score           = score
        self.labels          = labels
        self.cat_resi        = cat_resi
        self.select_unique   = select_unique
        self.normalize       = normalize

        # Read in dataset
        self.df, self.lengths, self.max_len, self.sequences = self.read_datasets_csv()

        # Normalize the datasets
        self.df = self.normalize_scores(self.df)

        # Split the dataset
        self.train_df, self.test_df = train_test_split(self.df, test_size=test_size, random_state=42)

    def normalize_scores(self, df):
        score_list = df[self.score].to_numpy().astype(np.float32)
        if self.normalize == 'minmax':
            df[f'norm_{self.score}'] = self.min_max(score_list)
        if self.normalize == 'both':
            score_list = self.min_max(score_list)
            df[f'norm_{self.score}'] = self.z_score(score_list)
        if self.normalize == 'z_score':
            df[f'norm_{self.score}'] = self.z_score(score_list)
                    
        print('### Data normalized. ###')

        return df   
    
    def read_datasets_csv(self):

        df = pd.read_csv(self.df_path)
        df = df[df['sequence'].notnull()]
        
        if self.select_unique:
            score_df = df.groupby('sequence')[[self.score]].mean().reset_index()
            label_df = df.drop_duplicates('sequence')[self.labels + ['sequence']]
            df = pd.merge(score_df, label_df, on='sequence')
        if self.cat_resi is not None:
            df = df[df['cat_resi'] == self.cat_resi]

        sequences = df['sequence'].tolist()
        lengths = np.array([len(seq) for seq in sequences])
        max_len = np.max(lengths)

        print(f'### Data loaded from: {self.df_path} ###')

        return df, lengths, max_len, sequences
    
    def train_PLM(self,
                  epochs          = 3,
                  esm2_model_name = None,
                  p_loss          = 0.1):

        self.epochs          = epochs
        self.esm2_model_name = esm2_model_name
        self.p_loss          = p_loss

        if esm2_model_name is not None:
            self.plm_model_name = esm2_model_name    
            self.tokenize_esm2_model(p_loss)

        if os.path.isfile(f'{self.output_folder}/plm_self_{self.file_title()}.pkl'):

            print(f'{self.output_folder}/plm_self_{self.file_title()}.pkl exists! Stopping calculation')

            return
                              
        self.trainer()

        print(f'PLM {self.plm_model_name} trained for norm_{self.score}')

        self.eval_test_train()
        self.save_self_to_file()
        self.plot_learning()

        torch.cuda.empty_cache()
    
    def trainer(self):
        self.train_loss = []
        self.test_loss = []

        # Initialize torch
        self.device, self.dtype, self.SMOKE_TEST = setup_torch()

        # Load and preprocess data
        train_sequences = self.train_df['sequence'].tolist()
        train_activities = self.train_df[f'norm_{self.score}'].tolist()
        test_sequences = self.test_df['sequence'].tolist()
        test_activities = self.test_df[f'norm_{self.score}'].tolist()

        train_dataset = ProteinDataset(train_sequences, train_activities, self.tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_dataset = ProteinDataset(test_sequences, test_activities, self.tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=5e-5)

        self.model.to(self.device)

        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            total_train_loss = 0
            for batch in train_dataloader:
                input_ids = batch['input_ids'].squeeze(1).to(self.device)
                attention_mask = batch['attention_mask'].squeeze(1).to(self.device)
                activities = batch['activity'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.squeeze(), activities)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_dataloader)
            self.train_loss.append(avg_train_loss)

            # Evaluate on test set
            self.model.eval()
            total_test_loss = 0
            with torch.no_grad():
                for batch in test_dataloader:
                    input_ids = batch['input_ids'].squeeze(1).to(self.device)
                    attention_mask = batch['attention_mask'].squeeze(1).to(self.device)
                    activities = batch['activity'].to(self.device)

                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fn(outputs.squeeze(), activities)
                    total_test_loss += loss.item()

            avg_test_loss = total_test_loss / len(test_dataloader)
            self.test_loss.append(avg_test_loss)
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} completed.\ttrain loss: {avg_train_loss:.2f}.\ttest loss: {avg_test_loss:.2f}")

    def evaluate(self):
        self.model.eval()
        test_sequences = self.test_df['sequence'].tolist()
        test_activities = self.test_df[f'norm_{self.score}'].tolist()

        test_dataset = ProteinDataset(test_sequences, test_activities, self.tokenizer, max_length=128)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        loss_fn = nn.MSELoss()
        total_loss = 0

        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids'].squeeze(1).to(self.device)
                attention_mask = batch['attention_mask'].squeeze(1).to(self.device)
                activities = batch['activity'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.squeeze(), activities)
                total_loss += loss.item()

        avg_loss = total_loss / len(test_dataloader)
        print(f"Test Loss: {avg_loss}")
        return avg_loss

    def tokenize_esm2_model(self, p_loss):

        from transformers import AutoModel
        self.esm_model = EsmModel.from_pretrained(self.esm2_model_name)
        self.tokenizer = EsmTokenizer.from_pretrained(self.esm2_model_name)
        self.model     = ProteinRegressor(self.esm_model, p_loss)

    def min_max(self, scores):
        return (scores - scores.min()) / (scores.max() - scores.min())
    
    def z_score(self, scores):
        return (scores - scores.mean()) / scores.std()
    
    def file_title(self):
        return f'model_{self.esm2_model_name.split("/")[-1]}_cat_{self.cat_resi}_epochs_{self.epochs}_ploss_{self.p_loss}'
    
    def save_self_to_file(self):
        with open(f'{self.output_folder}/plm_self_{self.file_title()}.pkl', 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load_self_from_file(self, epochs, esm2_model_name, cat_resi, p_loss, df_path):
        self.epochs = epochs
        self.esm2_model_name = esm2_model_name
        self.cat_resi = cat_resi
        self.p_loss = p_loss
        self.df_path = df_path

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
        train_activities = self.train_df[f'norm_{self.score}'].tolist()
        test_sequences = self.test_df['sequence'].tolist()
        test_activities = self.test_df[f'norm_{self.score}'].tolist()

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
                train_preds.extend(outputs.squeeze().cpu().numpy())

            for batch in test_dataloader:
                input_ids = batch['input_ids'].squeeze(1).to(self.device)
                attention_mask = batch['attention_mask'].squeeze(1).to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                test_preds.extend(outputs.squeeze().cpu().numpy())
       
        self.train_slope, self.train_intercept, self.train_r_value, self.train_p_value, self.train_std_err = stats.linregress(train_activities, train_preds)
        self.test_slope,  self.test_intercept,  self.test_r_value,  self.test_p_value,  self.test_std_err = stats.linregress(test_activities, test_preds)

        self.train_rmse = np.sqrt(mean_squared_error(train_activities, train_preds))
        self.test_rmse  = np.sqrt(mean_squared_error(test_activities, test_preds))

        self.train_activities = train_activities
        self.train_preds = train_preds
        self.test_activities = test_activities
        self.test_preds = test_preds

    def plot_learning(self):
       
        if not os.path.isfile(f'{self.output_folder}/plm_self_{self.file_title()}.pkl'):
            print(f'### {self.output_folder}/plm_self_{self.file_title()}.pkl does not exist. ###' )
            return
        
        fig = plt.figure(figsize=(10, 5))
        gs = fig.add_gridspec(1, 2)
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[:, 1])

        c_test = "b"
        c_train = "r"

        # Left panel: y vs y_hat scatter plot for test and train datasets
        min_val = min(min(self.train_activities), min(self.train_preds), min(self.test_activities), min(self.test_preds))
        max_val = max(max(self.train_activities), max(self.train_preds), max(self.test_activities), max(self.test_preds))
        
        ax1.plot([min_val, max_val], [min_val, max_val], 'k', zorder=100)

        ax1.scatter(self.train_activities, self.train_preds, label='Train', alpha=0.5, c=c_train)
        line = self.train_slope * np.array([min_val, max_val]) + self.train_intercept
        ax1.plot([min_val, max_val], line, label=f'R2={self.train_r_value**2:.2f}, p={self.train_p_value:.2g}, rmse={self.train_rmse:.2g}', c=c_train, zorder=10)
        
        ax1.scatter(self.test_activities, self.test_preds, label='Test', alpha=0.5, c=c_test)
        line = self.test_slope * np.array([min_val, max_val]) + self.test_intercept
        ax1.plot([min_val, max_val], line, label=f'R2={self.test_r_value**2:.2f}, p={self.test_p_value:.2g}, rmse={self.test_rmse:.2g}', c=c_test, zorder=20)
        
        ax1.set_xlim([min_val, max_val])
        ax1.set_ylim([min_val, max_val])       
        ax1.set_xlabel('True Activity')
        ax1.set_ylabel('Predicted Activity')
        ax1.set_title('True vs Predicted Activity')
        ax1.legend()

        # Middle panel: Loss vs. Epoch plot
        ax2.plot(range(1,len(self.train_loss)+1,1), self.train_loss, label='train_loss', c=c_train, zorder=1)
        ax2.plot(range(1,len(self.train_loss)+1,1), self.test_loss,  label='test_loss', c=c_test  , zorder=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_xlim(0,len(self.train_loss))
        ax2.set_ylim(bottom=0)
        ax2.set_title('Loss vs. Epoch')
        ax2.legend()

        plt.suptitle(self.file_title().replace('_', ' '))
        plt.tight_layout()
        plt.savefig(f"{self.output_folder}/{self.file_title()}.png")
        plt.show()

        # Collect data
        filename = f"{self.output_folder}/results.pkl"  
        df = pd.DataFrame()
        df['file_title'] = self.file_title

        if os.path.isfile(filename):
            df = pd.read_pickle(filename)

        data = {
            'file_title': self.file_title,
            'test_r_value': self.test_r_value,
            'test_p_value': self.test_p_value,
            'test_rmse': self.test_rmse,
            'esm2_model_name': self.esm2_model_name,
            'cat_resi': self.cat_resi,
            'epochs': self.epochs,
            'p_loss': self.p_loss,
            'df_path': self.df_path
        }

        if self.file_title in df['file_title'].values:
            df.loc[df['file_title'] == self.file_title, :] = pd.DataFrame([data])
        else:
            new_df = pd.DataFrame([data])
            df = pd.concat([df, new_df], ignore_index=True)
        df.to_pickle(filename)