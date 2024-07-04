from abc import abstractmethod
import os

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from src.losses import bt_loss

class FitnessTrainer:
    def __init__(self):
        self.models = {}
        self.optimizers = {}
        self.history = {'train_epoch': [], 'train_loss_per_epoch': [], 'validation_epoch': [], 'validation_loss_per_epoch': []}
        self.scores = []
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
                       device,
                       lr = 0.001,
                       epochs = 20,
                       loss = 'bt',
                       encoding = 'onehot',
                       scores = ['norm_interface_score', 'norm_total_score'],
                       features = None,
                       verbose = True,
                       checkpoint_path = './checkpoint',
                       checkpoint_epoch = 10,
                       validate_epoch = 1):

        super().__init__()

        self.epochs = epochs
        self.scores = scores
        self.device = device
        self.features = features
        self.encoding = encoding
        self.verbose = verbose

        for score in self.scores:
            self.models[score] = model.to(device)
            self.optimizers[score] = optim(self.models[score].parameters(), lr = lr)

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
        for score in self.scores:
            if self.verbose:
                print(f'### Training regression model to predict: {score} ###')

            for epoch in range(1, self.epochs + 1):
                self.models[score].train()

                losses_per_epoch = []
                for iter, batch in enumerate(train_dl):

                    y = batch[score].to(device)

                    self.optimizers[score].zero_grad()
                    y_hat = self.predict(batch, score)

                    loss = self.loss_fn(y_hat, y)
                    loss.backward()

                    self.optimizers[score].step()

                    losses_per_epoch.append(loss.item())

                self.history['train_epoch'].append(epoch)
                self.history['train_loss_per_epoch'].append(np.mean(losses_per_epoch))

                if self.verbose:
                    print(f"[Train] epoch:{epoch} \t loss:{np.mean(losses_per_epoch):.4f}")

                if epoch % self.validate_epoch == 0:
                    valid_loss = self.validate(valid_dl, score)
                    self.history['validation_epoch'].append(epoch)
                    self.history['validation_loss_per_epoch'].append(valid_loss)

                    if self.verbose:
                        print(f"[Validation] epoch:{epoch} \t loss:{valid_loss:.4f}")


                if self.checkpoint_path is not None:
                    if epoch % self.checkpoint_epoch == 0:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.models[score].state_dict(),
                            'optimizer_state_dict': self.optimizers[score].state_dict(),
                            'loss': np.mean(losses_per_epoch),
                        }, os.path.join(self.checkpoint_path, f'model_{score}_checkpoint_{epoch}.pt'))

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

    def predict(self, input, score):
        enc = self.encode_inputs(input)

        return self.models[score](enc)

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
                y = batch[score].to(device)
                y_hat = self.predict(batch, score)

                valid_losses.append(self.loss_fn(y_hat, y).item())

        return np.mean(valid_losses)


    def enable_dropout(self, model):
        """ Function to enable the dropout layers during test-time """
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()