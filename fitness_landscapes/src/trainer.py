from abc import abstractmethod
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from src.losses import bt_loss
from sklearn.model_selection import train_test_split
from src.ff import MLP
import torch
from src.tools import setup_torch
import numpy as np
from scipy.stats import pearsonr, linregress
from tqdm import tqdm
import pickle

print("Trainers loaded.")

class df_to_dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.df = df
    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):

        out = {}
        for key in self.df.keys():
            if isinstance(self.df[key].iloc[item], str):
                out[key] = self.df[key].iloc[item]
            else:
                out[key] = torch.tensor(self.df[key].iloc[item], dtype=torch.float)

        return out

def train_model(dataset,
                encoding = 'onehot_plm_pca',
                epochs = 10,
                verbose = True,
                output_folder = "./"):

    output = {}

    train_df, test_df = train_test_split(dataset.df, test_size = 0.1)

    train_dl = torch.utils.data.DataLoader(df_to_dataset(train_df), batch_size = 64)
    test_dl = torch.utils.data.DataLoader(df_to_dataset(test_df), batch_size = len(test_df))
    
    score = 'norm_total_score'
    features = None
    
    input_size = len(train_df[encoding].iloc[0])

    n_features = [64,1]
    n_layers = len(n_features)
    dropout_prob = 0.3
    input_model = MLP(input_size, n_layers, n_features, dropout_prob)

    optimizer = torch.optim.Adam

    trainer = TrainerGeneral(input_model, optimizer, 
                            encoding = encoding, 
                            features = features, 
                            score = score, 
                            epochs = epochs,
                            verbose = verbose)
    
    output['model'], output['train_epoch'], output['train_loss_per_epoch'], output['train_mae_per_epoch'], output['test_epoch'], output['test_loss_per_epoch'], output['test_mae_per_epoch'] = trainer.train(train_dl, test_dl)

    if features != None: output['features'].append('_'.join(features))

    output['yyhat'] = trainer.predict_mc_dropout(test_dl, score, forward_passes = 50)
    pears = pearsonr(output['yyhat']['y'], output['yyhat']['yhat']).statistic
    pvals = pearsonr(output['yyhat']['y'], output['yyhat']['yhat']).pvalue
    slope, intercept, r_value, p_value, std_err = linregress(output['yyhat']['y'], output['yyhat']['yhat'])
    output['pearsonr']  = pears
    output['pval']      = pvals
    output['slope']     = slope
    output['intercept'] = intercept
    output['r_value']   = r_value
    output['p_value']   = p_value
    output['std_err']   = std_err
    output['encoding']  = encoding
    output['score']     = score
    output['epochs']    = epochs

    #Save results to file
    title = f"score_{output['score']}__encoding_{output['encoding']}__epochs_{output['epochs']}"
    with open(f'{output_folder}/{title}.pkl', 'wb') as file:
        pickle.dump(output, file)

    torch.cuda.empty_cache()

    return output
    
class FitnessTrainer:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.history = {'train_epoch': [], 
                        'train_loss_per_epoch': [],
                        'train_mae_per_epoch': [],
                        'test_epoch': [], 
                        'test_loss_per_epoch': [],
                        'test_mae_per_epoch': []}      
        self.score = None
        self.device = None
        self.loss_fn = None
    
class TrainerGeneral(FitnessTrainer):
    def __init__(self, model, optim,
                       lr = 0.001,
                       epochs = 20,
                       encoding = 'onehot',
                       score = 'norm_interface_score',
                       features = None,
                       verbose = True,
                       checkpoint_path = './checkpoint',
                       checkpoint_epoch = 10,
                       validate_epoch = 1):

        super().__init__()

        #initialize torch
        self.device, self.dtype, self.SMOKE_TEST = setup_torch()

        self.epochs = epochs
        self.score = score
        self.features = features
        self.encoding = encoding
        self.verbose = verbose

        self.model = model.to(self.device)
        self.optimizer = optim(self.model.parameters(), lr = lr)
        self.loss_fn = F.mse_loss

        self.checkpoint_path = checkpoint_path
        if self.checkpoint_path is not None and os.path.isdir(self.checkpoint_path) is False:
            os.mkdir(self.checkpoint_path)
        self.checkpoint_epoch = checkpoint_epoch
        self.validate_epoch = validate_epoch

    def train(self, train_dl, valid_dl):

        if self.verbose:
            print(f'### Training regression model to predict: {self.score} ###')

        for epoch, _ in zip(range(1, self.epochs + 1), tqdm(range(self.epochs + 1))): #uses tqdm to display progress bar, output not used in code

            self.model.train()

            losses_per_epoch = []
            mae_per_epoch = []

            for iter, batch in enumerate(train_dl):

                y = batch[self.score].to(self.device)

                self.optimizer.zero_grad()
                y_hat = self.predict(batch).squeeze()
                loss = self.loss_fn(y_hat, y)
                loss.backward()

                self.optimizer.step()

                losses_per_epoch.append(loss.item())

                mae = F.mse_loss(y_hat, y).item()
                mae_per_epoch.append(mae)

            self.history['train_epoch'].append(epoch)
            self.history['train_loss_per_epoch'].append(np.mean(losses_per_epoch))
            self.history['train_mae_per_epoch'].append(np.mean(mae_per_epoch))

            if self.verbose:
                print(f"[train] epoch:{epoch} \t loss:{np.mean(losses_per_epoch):.4f} \t mae:{np.mean(mae_per_epoch):.4f}")

            if epoch % self.validate_epoch == 0:
                valid_loss, valid_mae = self.validate(valid_dl, self.score)
                self.history['test_epoch'].append(epoch)
                self.history['test_loss_per_epoch'].append(valid_loss)
                self.history['test_mae_per_epoch'].append(valid_mae)

                if self.verbose:
                    print(f"[test]  epoch:{epoch} \t loss:{valid_loss:.4f} \t mae:{valid_mae:.4f}")

            if self.checkpoint_path is not None:
                if epoch % self.checkpoint_epoch == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': np.mean(losses_per_epoch),
                    }, os.path.join(self.checkpoint_path, f'model_{self.score}_checkpoint_{epoch}.pt'))

        return self.model, self.history['train_epoch'], self.history['train_loss_per_epoch'], self.history['train_mae_per_epoch'], self.history['test_epoch'], self.history['test_loss_per_epoch'], self.history['test_mae_per_epoch']

    def encode_inputs(self, input):
        
        ''' OLD PROBABLY JUNK
        if self.encoding == 'onehot_flatten':
            enc = input['onehot'].reshape(input['onehot'].shape[0], -1) #Flattens one-hot

        elif self.encoding == 'onehot':
            enc = input['onehot']

        elif self.encoding == 'onehot_aizymes_features':
            enc = input['onehot'].reshape(input['onehot'].shape[0], -1)
            feat_list = [input[feat] for feat in self.features] + [enc]
            enc = torch.cat(feat_list, axis = -1)

        elif self.encoding == 'esm2':
            enc = input['esm2']

        elif self.encoding == 'vae':
            enc = input['vae']

        elif self.encoding == 'aizymes_features':
            feat_list = [input[feat] for feat in self.features]
            enc = torch.cat(feat_list, axis = -1)
        else:
            raise ValueError(f'Unrecognized encoding argument {self.encoding}!')
        '''

        enc = input[self.encoding]

        enc = enc.to(self.device)

        return enc

    def predict(self, input):
        enc = self.encode_inputs(input)

        return self.model(enc)

    def predict_mc_dropout(self, input, score, forward_passes = 50):

        yyhat = {'yhat': [], 'dyhat' : [], 'y' : []}
        
        for batch in input:

            enc = self.encode_inputs(batch)

            dropout_predictions = np.empty((0, enc.shape[0], 1))

            for i in range(forward_passes):
                self.model.eval()
                self.enable_dropout(self.model)
                predictions = self.model(enc).detach().cpu().numpy()
                dropout_predictions = np.vstack((dropout_predictions,
                                                predictions[np.newaxis, :, :]))

            yyhat['yhat'].append(np.mean(dropout_predictions, axis=0))
            yyhat['dyhat'].append(np.std(dropout_predictions, axis=0))
            yyhat['y'].append(batch[score])

        yyhat['yhat']  = [i[0] for i in yyhat['yhat'][0]]
        yyhat['dyhat'] = [i[0] for i in yyhat['dyhat'][0]]
        yyhat['y']      = [i.item() for i in yyhat['y'][0]]

        return yyhat

    def predict_numpy(self, input, score):
        enc = self.encode_inputs(input)
        self.model.eval()
        predictions = self.model(enc).detach().numpy()
        y = input[score].cpu().numpy()

        return {'y_hat': np.squeeze(predictions, axis = -1),
                'mean': None,
                'std': None,
                'y': np.squeeze(y, axis = -1)}

    def validate(self, valid_dl, score):

        self.model.eval()
        valid_losses = []
        mae_per_epoch = []

        with torch.no_grad():
            for iter, batch in enumerate(valid_dl):
                y = batch[score].to(self.device)
                y_hat = self.predict(batch).squeeze()

                valid_losses.append(self.loss_fn(y_hat, y).item())

                mae = F.l1_loss(y_hat, y).item()
                mae_per_epoch.append(mae)

        return np.mean(valid_losses), np.mean(mae_per_epoch)


    def enable_dropout(self, model):
        """ Function to enable the dropout layers during test-time """
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()