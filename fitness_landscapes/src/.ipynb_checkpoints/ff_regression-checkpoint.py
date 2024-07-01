import torch
import torch.nn.functional as F
import torch.nn as nn

from torch import vmap


class LinearSimple(torch.nn.Module):
    def __init__(self, input_dim, 
                  hidden_dim = 10, 
                  activation = torch.nn.ReLU(),
                  flatten = True,
                  classification = False,
                  dropout = 0.3):
        super(LinearSimple, self).__init__()
        
        if flatten:
            self.flatten = nn.Flatten()
        else:
            self.flatten = None
        
        self.layer_1 = torch.nn.Linear(input_dim, hidden_dim)
        
        if classification:
            self.layer_2 = torch.nn.Sequential(
                               activation,
                               torch.nn.Softmax(dim = -1))
        else:
            self.layer_2 = torch.nn.Sequential(
                                activation,
                                torch.nn.Linear(hidden_dim, 1))
        
        if dropout is not None:
            self.dropout = torch.nn.Dropout(p = dropout)
        else:
            self.dropout = None
        
    def forward(self, x):
        if self.flatten is not None:
            x = self.flatten(x)
        
        x = self.layer_1(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        x = self.layer_2(x)
        
        return x
        
        
class LinearResidual(torch.nn.Module):
    def __init__(self, input_dim, 
                 hidden_dim = 128, 
                 out_dim = 128, 
                 n_layers = 1, 
                 activation = torch.nn.ReLU(), 
                 residual = True, 
                 flatten = False,
                 dropout = 0.1):
        super(LinearResidual, self).__init__()
        
        layer_dims = [input_dim] + [hidden_dim] * n_layers + [out_dim]
        self.layer_dims = zip(layer_dims[:-1], layer_dims[1:])
        self.residual = residual
        self.flatten = flatten
        layers = [nn.Sequential(
                                nn.Linear(input_dim, out_dim),
                                nn.LayerNorm(out_dim),
                                activation,
                                nn.Dropout(dropout)) for i, (input_dim, out_dim) in enumerate(self.layer_dims)]

        self.layers = nn.Sequential(*layers)
            
        if self.residual:
            self.res_layer = nn.Linear(input_dim, out_dim) if input_dim != out_dim else nn.Identity()
            
        if self.flatten:
            self.flatten_layer = nn.Flatten()
        

    def forward(self, x):
        
        x = self.flatten_layer(x) if self.flatten else x
            
        out = self.layers(x)
        out = out + self.res_layer(x) if self.residual else out

        return out
    
class LinearEnsemble(torch.nn.Module):
    def __init__(self, n_ensemble, input_dim, 
                 hidden_dim = 128, n_layers = 1, 
                 activation = torch.nn.GELU(), 
                 residual = True, flatten = False, dropout = 0.1,
                 ensemble_method = None):
        super(LinearEnsemble, self).__init__()
        
        self.n_ensemble = n_ensemble
        self.regression_ensemble = [LinearResidual(input_dim = input_dim,
                                                   hidden_dim = hidden_dim,
                                                   out_dim = 1,
                                                   activation = activation,
                                                   residual = residual,
                                                   flatten = flatten,
                                                   dropout = dropout) for _ in range(n_ensemble)]
        
        self.mean = None
        self.std = None
        self.ensemble_method = ensemble_method
        
        if self.ensemble_method == 'weighted':
            self.weight_linear = nn.Linear(n_ensemble, 1)
            
    
    def forward(self, x):
        outs = [self.regression_ensemble[i](x) for i in range(self.n_ensemble)]
        out = torch.cat(outs, axis = 1)
        
        self.mean = out.mean(axis = 1)
        self.std = out.std(axis = 1)
        
        if self.ensemble_method = 'average':
            out = self.mean
        elif self.ensemble_method = 'weighted':
            out = self.weight_linear(out)
        
        return out
    

#Adapted from https://github.com/noowad93/MIMO-pytorch/blob/master/mimo/model.py
#Will also modify to adapt for multitask
class LinearMIMO(torch.nn.Module):
    def __init__(self, n_ensemble, input_dim, 
                 hidden_dim = 128, out_dim = 3, n_layers = 1, 
                 activation = torch.nn.ReLU(), 
                 residual = True, dropout = 0.1,
                 ensemble_method = 'average'):
        super(LinearMIMO, self).__init__()

        self.n_ensemble = n_ensemble
        self.input_layer = nn.Linear(input_dim, hidden_dim * n_ensemble)
        self.backbone_model = LinearResidual(input_dim = hidden_dim * n_ensemble,
                                             hidden_dim = hidden_dim,
                                             out_dim = hidden_dim,
                                             activation = activation,
                                             residual = residual,
                                             dropout = dropout)
                                     
        self.output_layer = nn.Linear(hidden_dim, n_ensemble * out_dim)

        
        if ensemble_method == 'average':
            self.linear_ensemble = None
        else:
            self.linear_ensemble = nn.Linear(n_ensemble, out_dim)
            
        self.mean = None
        self.std = None
        
        
    def forward(self, x):
        ensemble_num, batch_size, *_ = list(x.size())
        x = x.transpose(1, 0).view(
            batch_size, ensemble_num, -1
        )  # (batch_size, ensemble_num, hidden_dim)                             
                                                       
        x = self.input_layer(x)
        x = self.backbone_model(x)
        x = self.output_layer(x)
        x = x.reshape(
            batch_size, ensemble_num, -1, ensemble_num
        )
        x = torch.diagonal(x, offset=0, dim1=1, dim2=3).transpose(2, 1)
        #x = x.squeeze(-1) #Will remove for multitask
        
        #self.mean = x.mean(axis = 1)
        #self.std = x.std(axis = 1)
        
        #if self.linear_ensemble is not None:
        #    x = self.linear_ensemble(x)
        #else:
        #    x = self.mean

        return x