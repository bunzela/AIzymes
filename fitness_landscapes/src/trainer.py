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

def train_model(dataset):

    train_df, test_df = train_test_split(dataset.df, test_size = 0.1)

    train_dl = torch.utils.data.DataLoader(df_to_dataset(train_df), batch_size = 64)
    test_dl = torch.utils.data.DataLoader(df_to_dataset(test_df), batch_size = len(test_df))
    
    output = {'loss': [], 'score': [], 'dataset': [], 'features': [], 'pearsonr': [], 'pval': []}
    score = 'norm_total_score'
    feature = 'onehot_plm_pca'
    loss = 'bt'
    input_size = len(train_df[feature])
    input_size = 64    
    encoding = 'onehot_flatten'

    epochs = 10 #Seems decent ATM - most valid losses increase past this
    n_layers = 2
    n_features = [64,1]
    dropout_prob = 0.3
    model = MLP(input_size, n_layers, n_features, dropout_prob)
    optimizer = torch.optim.Adam
    vals = next(iter(test_dl))

    trainer = TrainerGeneral(model, optimizer, 
                            encoding = encoding, loss = loss, features = feature, 
                            score = score, epochs = epochs)
    
    models = trainer.train(train_dl, test_dl)
    
    torch.cuda.empty_cache()

    return "done"   

    
    output['loss'].append(loss)
    output['score'].append(score)
    output['dataset'].append(data)
    feature_name = '_'.join(feature)
    output['features'].append(feature_name)
    output['pearsonr'].append(pears)
    output['pval'].append(pvals)

    mc = trainer.predict_mc_dropout(vals, scores_pred, forward_passes = 50)
    pears = pearsonr(mc['y'], mc['mean']).statistic
    pvals = pearsonr(mc['y'], mc['mean']).pvalue
    
class FitnessTrainer:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.history = {'train_epoch': [], 'train_loss_per_epoch': [], 'validation_epoch': [], 'validation_loss_per_epoch': []}
        self.score = None
        self.device = None
        self.loss_fn = None

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def plot_training_metrics(self):
        pass

    @abstractmethod
    def save_history(self):
        pass

    @abstractmethod
    def update_history(self):
        pass

    @abstractmethod
    def save_model_checkpoint(self):
        pass

    @abstractmethod
    def load_model_checkpoint(self):
        pass

    @abstractmethod
    def encode_inputs(self, input):
        pass
    
class TrainerGeneral(FitnessTrainer):
    def __init__(self, model, optim,
                       lr = 0.001,
                       epochs = 20,
                       loss = 'bt',
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

        print("test")

        if loss == 'bt':
            self.loss_fn = bt_loss
        else:
            self.loss_fn = F.mse_loss

        self.checkpoint_path = checkpoint_path
        if self.checkpoint_path is not None and os.path.isdir(self.checkpoint_path) is False:
            os.mkdir(self.checkpoint_path)
        self.checkpoint_epoch = checkpoint_epoch
        self.validate_epoch = validate_epoch

    def train(self, train_dl, valid_dl):

        if self.verbose:
            print(f'### Training regression model to predict: {self.score} ###')

        for epoch in range(1, self.epochs + 1):

            self.model.train()

            losses_per_epoch = []
            for iter, batch in enumerate(train_dl):

                y = batch[self.score].to(self.device)

                self.optimizer.zero_grad()
                y_hat = self.predict(batch)

                loss = self.loss_fn(y_hat, y)
                loss.backward()

                self.optimizer.step()

                losses_per_epoch.append(loss.item())

            self.history['train_epoch'].append(epoch)
            self.history['train_loss_per_epoch'].append(np.mean(losses_per_epoch))

            if self.verbose:
                print(f"[Train] epoch:{epoch} \t loss:{np.mean(losses_per_epoch):.4f}")

            if epoch % self.validate_epoch == 0:
                valid_loss = self.validate(valid_dl, self.score)
                self.history['validation_epoch'].append(epoch)
                self.history['validation_loss_per_epoch'].append(valid_loss)

                if self.verbose:
                    print(f"[Validation] epoch:{epoch} \t loss:{valid_loss:.4f}")


            if self.checkpoint_path is not None:
                if epoch % self.checkpoint_epoch == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.models[self.score].state_dict(),
                        'optimizer_state_dict': self.optimizers[self.score].state_dict(),
                        'loss': np.mean(losses_per_epoch),
                    }, os.path.join(self.checkpoint_path, f'model_{self.score}_checkpoint_{epoch}.pt'))

        return self.models

    def encode_inputs(self, input):
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

        enc = enc.to(self.device)

        return enc

    def predict(self, input):
        enc = self.encode_inputs(input)

        return self.model(enc)

    def predict_mc_dropout(self, input, score, forward_passes = 50):
        enc = self.encode_inputs(input)

        dropout_predictions = np.empty((0, enc.shape[0], 1))

        for i in range(forward_passes):
            self.models[score].eval()
            self.enable_dropout(self.models[score])
            predictions = self.models[score](enc).detach().numpy()
            dropout_predictions = np.vstack((dropout_predictions,
                                             predictions[np.newaxis, :, :]))


        mean = np.mean(dropout_predictions, axis=0)
        std = np.std(dropout_predictions, axis=0)
        y = input[score].cpu().numpy()

        return {'y_hat': np.squeeze(dropout_predictions, axis = -1),
                'mean': np.squeeze(mean, axis = -1),
                'std': np.squeeze(std, axis = -1),
                'y': np.squeeze(y, axis = -1)}

    def predict_numpy(self, input, score):
        enc = self.encode_inputs(input)
        self.models[score].eval()
        predictions = self.models[score](enc).detach().numpy()
        y = input[score].cpu().numpy()

        return {'y_hat': np.squeeze(predictions, axis = -1),
                'mean': None,
                'std': None,
                'y': np.squeeze(y, axis = -1)}

    def validate(self, valid_dl, score):

        self.models[score].eval()
        valid_losses = []

        with torch.no_grad():
            for iter, batch in enumerate(valid_dl):
                y = batch[score].to(self.device)
                y_hat = self.predict(batch)

                valid_losses.append(self.loss_fn(y_hat, y).item())

        return np.mean(valid_losses)


    def enable_dropout(self, model):
        """ Function to enable the dropout layers during test-time """
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()