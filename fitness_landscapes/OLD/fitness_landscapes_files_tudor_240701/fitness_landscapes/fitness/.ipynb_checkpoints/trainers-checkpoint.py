from abc import abstractmethod
import os

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from fitness.losses import bt_loss

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

    
def train_mimo(train_ds, 
               test_ds, 
               device,
               model,
               scores = ['total_score', 'interface_score', 'catalytic_score'], 
               training_iter = 50, 
               lr = 1.0,
               n_ensemble = 5,
               batch_size = 128):
    
    input_dim = train_ds.embeddings['esm2'][0].shape[-1]
    train_dataloaders = [torch.utils.data.DataLoader(train_ds, batch_size = batch_size, shuffle = True) for _ in range(n_ensemble)]
    test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size = batch_size, shuffle = True)
    model = model.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=len(train_ds.sequences), gamma=0.7)

    global_step = 0

    for epoch in range(1, training_iter + 1):
        for datum in zip(*train_dataloaders):
            train_x = torch.stack([data['esm2'] for data in datum]).to(device)
            train_y = torch.stack([data['norm_interface_score'] for data in datum]).float().to(device)
            #train_y = torch.cat(train_y, axis = -1)
            n_ensemble, batch_size = list(train_y.size())

            optimizer.zero_grad()
            outputs = model(train_x)
            #print(outputs.reshape(n_ensemble * batch_size, -1).shape)
            #print(train_y.reshape(n_ensemble * batch_size, -1).shape)
            loss = F.mse_loss(
                    outputs.reshape(n_ensemble * batch_size, -1), train_y.reshape(n_ensemble * batch_size, -1)
            )
                     
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            if global_step != 0 and global_step % 10 == 0:
                print(f"[Train] epoch:{epoch} \t global step:{global_step} \t loss:{loss:.4f}")
            
            
                model.eval()
                test_loss = 0
                correct = 0
                with torch.no_grad():
                    for data in test_dataloader:
                        model_inputs = torch.stack([data['esm2']] * 5).to(device)
                        target = data['norm_interface_score'].to(device)

                        outputs = model(model_inputs)
                        output = torch.mean(outputs, axis=1)

                        test_loss += F.mse_loss(output, target, reduction="sum").item()
                       
        test_loss /= len(test_dataloader.dataset)
        print(f"[Valid] Average loss: {test_loss:.4f}")
        model.train()
    
    return model

def train_gps_simple(model_name, model, likelihood, train_ds, test_ds, scores, training_iter, device, lr = 0.1,
                     encoding = 'esm2'):
    observed_preds = []
    val_per_iter = []
    test_ys = []
    #ids = np.random.choice(np.arange(len(train_ds.sequences)), size = 10000)
    trained_models = []
    for score in scores:
        
        torch.cuda.empty_cache()
        gc.collect()  
        
        if encoding == 'esm2':
            train_x = torch.tensor(train_ds.embeddings['esm2']).to(device)
            train_x = F.normalize(train_x)
            train_y = torch.tensor(train_ds.scores['norm_' + score]).to(device)
            test_x = torch.tensor(test_ds.embeddings['esm2']).to(device)
            test_x = F.normalize(test_x)
            test_y = torch.tensor(test_ds.scores['norm_' + score]).to(device)
        else:
            train_x = torch.stack(train_ds.embeddings['onehot']).to(device)
            train_x = train_x.reshape(train_x.shape[0], -1)
            train_x = F.normalize(train_x)

            train_y = torch.tensor(train_ds.scores['norm_' + score]).to(device)
            
            test_x =  torch.stack(test_ds.embeddings['onehot']).to(device)
            test_x = test_x.reshape(test_x.shape[0], -1)
            test_x = F.normalize(test_x)

            test_y = torch.tensor(test_ds.scores['norm_' + score]).to(device)


        
        likelihood = likelihood.to(device)
        model = model.to(device)
        
        model.train()
        likelihood.train()
        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        loss_dict = {'loss':[], 'noise': [], 'train_iter': [], 
                     'mae':[], 'nlpd':[], 'msll':[], 'mse':[]}
        for i in range(training_iter):
            model.train()
            likelihood.train() 
            
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            #print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            #    i + 1, training_iter, loss.item(),
            #    model.covar_module.base_kernel.lengthscale.item(),
            #    model.likelihood.noise.item()
            #))
            if model_name != 'FixedNoiseGP':
                print('Iter %d/%d - Loss: %.3f     noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    #model.covar_module.base_kernel.lengthscale.item(),
                    model.likelihood.noise.item()
                ))
                loss_dict['loss'].append(loss.item())
                loss_dict['noise'].append(model.likelihood.noise.item())
                loss_dict['train_iter'].append(i)
            else:
                print('Iter %d/%d - Loss: %.3f     noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    #model.covar_module.base_kernel.lengthscale.item(),
                    model.likelihood.noise[0].item()
                ))
                loss_dict['loss'].append(loss.item())
                loss_dict['noise'].append(model.likelihood.noise[0].item())
                loss_dict['train_iter'].append(i)
                

            optimizer.step()
            
            model.eval()
            likelihood.eval()
            
            if model_name != 'FixedNoiseGP':
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    observed_pred = likelihood(model(test_x))
                    loss_dict['mae'].append(gpytorch.metrics.mean_absolute_error(observed_pred, test_y).item())
                    #loss_dict['mse'].append(gpytorch.metrics.mean_squared_error(observed_pred, test_y).item())
                    #loss_dict['nlpd'].append(gpytorch.metrics.negative_log_predictive_density(observed_pred, test_y).item())
                    #loss_dict['msll'].append(gpytorch.metrics.mean_standardized_log_loss(observed_pred, test_y).item())
            else:
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    test_noises = torch.ones(test_y.shape) * 0.01
                    observed_pred = likelihood(model(test_x), noise = test_noises)
                    loss_dict['mae'].append(gpytorch.metrics.mean_absolute_error(observed_pred, test_y).item())
                
        val_per_iter.append(loss_dict)
        
        model.eval()
        likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(test_x))
            
        observed_preds.append(observed_pred)
        test_ys.append(test_y)
        
        trained_models.append(model)
    
    plot_pearson(observed_preds, test_ys, scores, model_name)
    plot_metrics_per_iter(val_per_iter, scores, model_name)
    
    return (trained_models, observed_preds, test_ys, val_per_iter)


def train_multitask_gps_simple(model_name, model, likelihood, train_ds, test_ds, scores, training_iter, device, lr = 0.1):
    observed_preds = []
    val_per_iter = []
    train_ys = []
    test_ys = []
    n_tasks = len(scores)
    
    trained_models = []
    
    train_x = torch.tensor(train_ds.embeddings['esm2']).to(device)
    train_x = F.normalize(train_x)
    
    test_x = torch.tensor(test_ds.embeddings['esm2']).to(device)
    test_x = F.normalize(test_x)
    
    for score in scores:
        train_ys.append(torch.tensor(train_ds.scores['norm_' + score]).to(device).unsqueeze(-1))
        test_ys.append(torch.tensor(test_ds.scores['norm_' + score]).to(device).unsqueeze(-1))
    
    train_y = torch.cat(train_ys, axis = -1)
    test_y = torch.cat(test_ys, axis = -1)
    
    likelihood = likelihood.to(device)
    model = model.to(device)
    
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    loss_dict = {'loss':[], 'noise': [], 'train_iter': [], 
                     'mae':[], 'nlpd':[], 'msll':[], 'mse':[]}
    
    for i in range(training_iter):
        model.train()
        likelihood.train() 

        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
     
        print('Iter %d/%d - Loss: %.3f     noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            #model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        loss_dict['loss'].append(loss.item())
        loss_dict['noise'].append(model.likelihood.noise.item())
        loss_dict['train_iter'].append(i)
            

        optimizer.step()

        model.eval()
        likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(test_x))
            loss_dict['mae'].append(gpytorch.metrics.mean_absolute_error(observed_pred, test_y).unsqueeze(1))
            #loss_dict['mse'].append(gpytorch.metrics.mean_squared_error(observed_pred, test_y).item())
            #loss_dict['nlpd'].append(gpytorch.metrics.negative_log_predictive_density(observed_pred, test_y).item())
            #loss_dict['msll'].append(gpytorch.metrics.mean_standardized_log_loss(observed_pred, test_y).item())
    
    loss_dict['mae'] = torch.cat(loss_dict['mae'], axis = 1).transpose(1,0)
    val_per_iter.append(loss_dict)
   
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_preds = likelihood(model(test_x))

    trained_models.append(model)
    #plot_pearson(observed_preds, test_ys, scores, model_name)
    #plot_metrics_per_iter(val_per_iter, scores, model_name)

    return (observed_preds, test_y, val_per_iter)
