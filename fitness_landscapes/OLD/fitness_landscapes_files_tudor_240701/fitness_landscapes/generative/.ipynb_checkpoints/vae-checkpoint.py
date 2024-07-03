import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch import nn, optim
import torch.nn.init as init
from torch.nn import functional as F
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from itertools import chain


class LossFunctions:
    eps = 1e-8

    def mean_squared_error(self, real, predictions):
      """Mean Squared Error between the true and predicted outputs
         loss = (1/n)*Σ(real - predicted)^2

      Args:
          real: (array) corresponding array containing the true labels
          predictions: (array) corresponding array containing the predicted labels
 
      Returns:
          output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      loss = (real - predictions).pow(2)
      return loss.sum(-1).mean()


    def reconstruction_loss(self, real, predicted, rec_type='mse' ):
      """Reconstruction loss between the true and predicted outputs
         mse = (1/n)*Σ(real - predicted)^2
         bce = (1/n) * -Σ(real*log(predicted) + (1 - real)*log(1 - predicted))

      Args:
          real: (array) corresponding array containing the true labels
          predictions: (array) corresponding array containing the predicted labels
 
      Returns:
          output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      if rec_type == 'mse':
        loss = (real - predicted).pow(2)
      elif rec_type == 'bce':
        loss = F.binary_cross_entropy(predicted, real, reduction='none')
      elif rec_type == 'cce':
        loss = F.cross_entropy(predicted, real, reduction='none')
      else:
        raise "invalid loss function... try bce or mse..."
      return loss.sum(-1).mean()


    def log_normal(self, x, mu, var):
      """Logarithm of normal distribution with mean=mu and variance=var
         log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

      Args:
         x: (array) corresponding array containing the input
         mu: (array) corresponding array containing the mean 
         var: (array) corresponding array containing the variance

      Returns:
         output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      if self.eps > 0.0:
        var = var + self.eps
      return -0.5 * torch.sum(
        np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)


    def gaussian_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):
      """Variational loss when using labeled data without considering reconstruction loss
         loss = log q(z|x,y) - log p(z) - log p(y)

      Args:
         z: (array) array containing the gaussian latent variable
         z_mu: (array) array containing the mean of the inference model
         z_var: (array) array containing the variance of the inference model
         z_mu_prior: (array) array containing the prior mean of the generative model
         z_var_prior: (array) array containing the prior variance of the generative mode
         
      Returns:
         output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)
      return loss.mean()


    def entropy(self, logits, targets):
      """Entropy loss
          loss = (1/n) * -Σ targets*log(predicted)

      Args:
          logits: (array) corresponding array containing the logits of the categorical variable
          real: (array) corresponding array containing the true labels
 
      Returns:
          output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      log_q = F.log_softmax(logits, dim=-1)
      return -torch.mean(torch.sum(targets * log_q, dim=-1))


class Metrics:

  # Code taken from the work 
  # VaDE (Variational Deep Embedding:A Generative Approach to Clustering)
  def cluster_acc(self, Y_pred, Y):
    Y_pred, Y = np.array(Y_pred), np.array(Y)

    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
      w[Y_pred[i], Y[i]] += 1
    row, col = linear_sum_assignment(w.max()-w)
    return sum([w[row[i],col[i]] for i in range(row.shape[0])]) * 1.0/Y_pred.size


  def nmi(self, Y_pred, Y):
    Y_pred, Y = np.array(Y_pred), np.array(Y)
    assert Y_pred.size == Y.size
    return normalized_mutual_info_score(Y_pred, Y, average_method='arithmetic')

# Flatten layer
class Flatten(nn.Module):
  def forward(self, x):
    return x.view(x.size(0), -1)

# Reshape layer    
class Reshape(nn.Module):
  def __init__(self, outer_shape):
    super(Reshape, self).__init__()
    self.outer_shape = outer_shape
  def forward(self, x):
    return x.view(x.size(0), *self.outer_shape)

# Sample from the Gumbel-Softmax distribution and optionally discretize.
class GumbelSoftmax(nn.Module):

  def __init__(self, f_dim, c_dim):
    super(GumbelSoftmax, self).__init__()
    self.logits = nn.Linear(f_dim, c_dim)
    self.f_dim = f_dim
    self.c_dim = c_dim
     
  def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
    U = torch.rand(shape)
    if is_cuda:
      U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)

  def gumbel_softmax_sample(self, logits, temperature):
    y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
    return F.softmax(y / temperature, dim=-1)

  def gumbel_softmax(self, logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    #categorical_dim = 10
    y = self.gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard 
  
  def forward(self, x, temperature=1.0, hard=False):
    logits = self.logits(x).view(-1, self.c_dim)
    prob = F.softmax(logits, dim=-1)
    y = self.gumbel_softmax(logits, temperature, hard)
    return logits, prob, y

# Sample from a Gaussian distribution
class Gaussian(nn.Module):
  def __init__(self, in_dim, z_dim):
    super(Gaussian, self).__init__()
    self.mu = nn.Linear(in_dim, z_dim)
    self.var = nn.Linear(in_dim, z_dim)

  def reparameterize(self, mu, var):
    std = torch.sqrt(var + 1e-10)
    noise = torch.randn_like(std)
    z = mu + noise * std
    return z      

  def forward(self, x):
    mu = self.mu(x)
    var = F.softplus(self.var(x))
    z = self.reparameterize(mu, var)
    return mu, var, z 

# Inference Network
class InferenceNet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(InferenceNet, self).__init__()

    # q(y|x)
    self.inference_qyx = torch.nn.ModuleList([
        nn.Linear(x_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        GumbelSoftmax(512, y_dim)
    ])

    # q(z|y,x)
    self.inference_qzyx = torch.nn.ModuleList([
        nn.Linear(x_dim + y_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        Gaussian(512, z_dim)
    ])

  # q(y|x)
  def qyx(self, x, temperature, hard):
    num_layers = len(self.inference_qyx)
    for i, layer in enumerate(self.inference_qyx):
      if i == num_layers - 1:
        #last layer is gumbel softmax
        x = layer(x, temperature, hard)
      else:
        x = layer(x)
    return x

  # q(z|x,y)
  def qzxy(self, x, y):
    concat = torch.cat((x, y), dim=1)  
    for layer in self.inference_qzyx:
      concat = layer(concat)
    return concat
  
  def forward(self, x, temperature=1.0, hard=0):
    #x = Flatten(x)

    # q(y|x)
    logits, prob, y = self.qyx(x, temperature, hard)
    
    # q(z|x,y)
    mu, var, z = self.qzxy(x, y)

    output = {'mean': mu, 'var': var, 'gaussian': z, 
              'logits': logits, 'prob_cat': prob, 'categorical': y}
    return output


# Generative Network
class GenerativeNet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(GenerativeNet, self).__init__()

    # p(z|y)
    self.y_mu = nn.Linear(y_dim, z_dim)
    self.y_var = nn.Linear(y_dim, z_dim)

    # p(x|z)
    self.generative_pxz = torch.nn.ModuleList([
        nn.Linear(z_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, x_dim),
        torch.nn.Sigmoid()
    ])

  # p(z|y)
  def pzy(self, y):
    y_mu = self.y_mu(y)
    y_var = F.softplus(self.y_var(y))
    return y_mu, y_var
  
  # p(x|z)
  def pxz(self, z):
    for layer in self.generative_pxz:
      z = layer(z)
    return z

  def forward(self, z, y):
    # p(z|y)
    y_mu, y_var = self.pzy(y)
    
    # p(x|z)
    x_rec = self.pxz(z)

    output = {'y_mean': y_mu, 'y_var': y_var, 'x_rec': x_rec}
    return output


# GMVAE Network
class GMVAENet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(GMVAENet, self).__init__()

    self.inference = InferenceNet(x_dim, z_dim, y_dim)
    self.generative = GenerativeNet(x_dim, z_dim, y_dim)

    # weight initialization
    for m in self.modules():
      if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias.data is not None:
          init.constant_(m.bias, 0) 

  def forward(self, x, temperature=1.0, hard=0):
    x = x.view(x.size(0), -1)
    out_inf = self.inference(x, temperature, hard)
    z, y = out_inf['gaussian'], out_inf['categorical']
    out_gen = self.generative(z, y)
    
    # merge output
    output = out_inf
    for key, value in out_gen.items():
      output[key] = value
    return output

class GMVAE:

  def __init__(self, args):
    self.num_epochs = args.epochs
    self.cuda = args.cuda
    self.verbose = args.verbose

    self.batch_size = args.batch_size
    self.batch_size_val = args.batch_size_val
    self.learning_rate = args.learning_rate
    self.decay_epoch = args.decay_epoch
    self.lr_decay = args.lr_decay
    self.w_cat = args.w_categ
    self.w_gauss = args.w_gauss    
    self.w_rec = args.w_rec
    self.rec_type = args.rec_type 

    self.num_classes = args.num_classes
    self.gaussian_size = args.gaussian_size
    self.input_size = args.input_size

    # gumbel
    self.init_temp = args.init_temp
    self.decay_temp = args.decay_temp
    self.hard_gumbel = args.hard_gumbel
    self.min_temp = args.min_temp
    self.decay_temp_rate = args.decay_temp_rate
    self.gumbel_temp = self.init_temp

    self.network = GMVAENet(self.input_size, self.gaussian_size, self.num_classes)
    self.losses = LossFunctions()
    self.metrics = Metrics()

    if self.cuda:
      self.network = self.network.cuda() 
  

  def unlabeled_loss(self, data, out_net):
    """Method defining the loss functions derived from the variational lower bound
    Args:
        data: (array) corresponding array containing the input data
        out_net: (dict) contains the graph operations or nodes of the network output

    Returns:
        loss_dic: (dict) contains the values of each loss function and predictions
    """
    # obtain network variables
    z, data_recon = out_net['gaussian'], out_net['x_rec'] 
    logits, prob_cat = out_net['logits'], out_net['prob_cat']
    y_mu, y_var = out_net['y_mean'], out_net['y_var']
    mu, var = out_net['mean'], out_net['var']
    
    # reconstruction loss
    loss_rec = self.losses.reconstruction_loss(data, data_recon, self.rec_type)

    # gaussian loss
    loss_gauss = self.losses.gaussian_loss(z, mu, var, y_mu, y_var)

    # categorical loss
    loss_cat = -self.losses.entropy(logits, prob_cat) - np.log(0.1)

    # total loss
    loss_total = self.w_rec * loss_rec + self.w_gauss * loss_gauss + self.w_cat * loss_cat

    # obtain predictions
    _, predicted_labels = torch.max(logits, dim=1)

    loss_dic = {'total': loss_total, 
                'predicted_labels': predicted_labels,
                'reconstruction': loss_rec,
                'gaussian': loss_gauss,
                'categorical': loss_cat}
    return loss_dic
    
    
  def train_epoch(self, optimizer, data_loader):
    """Train the model for one epoch

    Args:
        optimizer: (Optim) optimizer to use in backpropagation
        data_loader: (DataLoader) corresponding loader containing the training data

    Returns:
        average of all loss values, accuracy, nmi
    """
    self.network.train()
    total_loss = 0.
    recon_loss = 0.
    cat_loss = 0.
    gauss_loss = 0.

    accuracy = 0.
    nmi = 0.
    num_batches = 0.
    
    true_labels_list = []
    predicted_labels_list = []

    # iterate over the dataset
    for (data, labels) in data_loader:
      if self.cuda == 1:
        data = data.cuda()

      optimizer.zero_grad()

      # flatten data
      data = data.view(data.size(0), -1)
      
      # forward call
      out_net = self.network(data, self.gumbel_temp, self.hard_gumbel) 
      unlab_loss_dic = self.unlabeled_loss(data, out_net) 
      total = unlab_loss_dic['total']

      # accumulate values
      total_loss += total.item()
      recon_loss += unlab_loss_dic['reconstruction'].item()
      gauss_loss += unlab_loss_dic['gaussian'].item()
      cat_loss += unlab_loss_dic['categorical'].item()

      # perform backpropagation
      total.backward()
      optimizer.step()  

      # save predicted and true labels
      predicted = unlab_loss_dic['predicted_labels']
      true_labels_list.append(labels)
      predicted_labels_list.append(predicted)   
   
      num_batches += 1. 

    # average per batch
    total_loss /= num_batches
    recon_loss /= num_batches
    gauss_loss /= num_batches
    cat_loss /= num_batches
    
    # concat all true and predicted labels
    true_labels = torch.cat(true_labels_list, dim=0).cpu().numpy()
    predicted_labels = torch.cat(predicted_labels_list, dim=0).cpu().numpy()

    # compute metrics
    accuracy = 100.0 * self.metrics.cluster_acc(predicted_labels, true_labels)
    nmi = 100.0 * self.metrics.nmi(predicted_labels, true_labels)

    return total_loss, recon_loss, gauss_loss, cat_loss, accuracy, nmi


  def test(self, data_loader, return_loss=False):
    """Test the model with new data

    Args:
        data_loader: (DataLoader) corresponding loader containing the test/validation data
        return_loss: (boolean) whether to return the average loss values
          
    Return:
        accuracy and nmi for the given test data

    """
    self.network.eval()
    total_loss = 0.
    recon_loss = 0.
    cat_loss = 0.
    gauss_loss = 0.

    accuracy = 0.
    nmi = 0.
    num_batches = 0.
    
    true_labels_list = []
    predicted_labels_list = []

    with torch.no_grad():
      for data, labels in data_loader:
        if self.cuda == 1:
          data = data.cuda()
      
        # flatten data
        data = data.view(data.size(0), -1)

        # forward call
        out_net = self.network(data, self.gumbel_temp, self.hard_gumbel) 
        unlab_loss_dic = self.unlabeled_loss(data, out_net)  

        # accumulate values
        total_loss += unlab_loss_dic['total'].item()
        recon_loss += unlab_loss_dic['reconstruction'].item()
        gauss_loss += unlab_loss_dic['gaussian'].item()
        cat_loss += unlab_loss_dic['categorical'].item()

        # save predicted and true labels
        predicted = unlab_loss_dic['predicted_labels']
        true_labels_list.append(labels)
        predicted_labels_list.append(predicted)   
   
        num_batches += 1. 

    # average per batch
    if return_loss:
      total_loss /= num_batches
      recon_loss /= num_batches
      gauss_loss /= num_batches
      cat_loss /= num_batches
    
    # concat all true and predicted labels
    true_labels = torch.cat(true_labels_list, dim=0).cpu().numpy()
    predicted_labels = torch.cat(predicted_labels_list, dim=0).cpu().numpy()

    # compute metrics
    accuracy = 100.0 * self.metrics.cluster_acc(predicted_labels, true_labels)
    nmi = 100.0 * self.metrics.nmi(predicted_labels, true_labels)

    if return_loss:
      return total_loss, recon_loss, gauss_loss, cat_loss, accuracy, nmi
    else:
      return accuracy, nmi


  def train(self, train_loader, val_loader):
    """Train the model

    Args:
        train_loader: (DataLoader) corresponding loader containing the training data
        val_loader: (DataLoader) corresponding loader containing the validation data

    Returns:
        output: (dict) contains the history of train/val loss
    """
    optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
    train_history_acc, val_history_acc = [], []
    train_history_nmi, val_history_nmi = [], []

    for epoch in range(1, self.num_epochs + 1):
      train_loss, train_rec, train_gauss, train_cat, train_acc, train_nmi = self.train_epoch(optimizer, train_loader)
      val_loss, val_rec, val_gauss, val_cat, val_acc, val_nmi = self.test(val_loader, True)

      # if verbose then print specific information about training
      if self.verbose == 1:
        print("(Epoch %d / %d)" % (epoch, self.num_epochs) )
        print("Train - REC: %.5lf;  Gauss: %.5lf;  Cat: %.5lf;" % \
              (train_rec, train_gauss, train_cat))
        print("Valid - REC: %.5lf;  Gauss: %.5lf;  Cat: %.5lf;" % \
              (val_rec, val_gauss, val_cat))
        print("Accuracy=Train: %.5lf; Val: %.5lf   NMI=Train: %.5lf; Val: %.5lf   Total Loss=Train: %.5lf; Val: %.5lf" % \
              (train_acc, val_acc, train_nmi, val_nmi, train_loss, val_loss))
      else:
        print('(Epoch %d / %d) Train_Loss: %.3lf; Val_Loss: %.3lf   Train_ACC: %.3lf; Val_ACC: %.3lf   Train_NMI: %.3lf; Val_NMI: %.3lf' % \
              (epoch, self.num_epochs, train_loss, val_loss, train_acc, val_acc, train_nmi, val_nmi))

      # decay gumbel temperature
      if self.decay_temp == 1:
        self.gumbel_temp = np.maximum(self.init_temp * np.exp(-self.decay_temp_rate * epoch), self.min_temp)
        if self.verbose == 1:
          print("Gumbel Temperature: %.3lf" % self.gumbel_temp)

      train_history_acc.append(train_acc)
      val_history_acc.append(val_acc)
      train_history_nmi.append(train_nmi)
      val_history_nmi.append(val_nmi)
    return {'train_history_nmi' : train_history_nmi, 'val_history_nmi': val_history_nmi,
            'train_history_acc': train_history_acc, 'val_history_acc': val_history_acc}
  

  def latent_features(self, data_loader, return_labels=False):
    """Obtain latent features learnt by the model

    Args:
        data_loader: (DataLoader) loader containing the data
        return_labels: (boolean) whether to return true labels or not

    Returns:
       features: (array) array containing the features from the data
    """
    self.network.eval()
    N = len(data_loader.dataset)
    features = np.zeros((N, self.gaussian_size))
    if return_labels:
      true_labels = np.zeros(N, dtype=np.int64)
    start_ind = 0
    with torch.no_grad():
      for (data, labels) in data_loader:
        if self.cuda == 1:
          data = data.cuda()
        # flatten data
        data = data.view(data.size(0), -1)  
        out = self.network.inference(data, self.gumbel_temp, self.hard_gumbel)
        latent_feat = out['mean']
        end_ind = min(start_ind + data.size(0), N+1)

        # return true labels
        if return_labels:
          true_labels[start_ind:end_ind] = labels.cpu().numpy()
        features[start_ind:end_ind] = latent_feat.cpu().detach().numpy()  
        start_ind += data.size(0)
    if return_labels:
      return features, true_labels
    return features


  def reconstruct_data(self, data_loader, sample_size=-1):
    """Reconstruct Data

    Args:
        data_loader: (DataLoader) loader containing the data
        sample_size: (int) size of random data to consider from data_loader
      
    Returns:
        reconstructed: (array) array containing the reconstructed data
    """
    self.network.eval()

    # sample random data from loader
    indices = np.random.randint(0, len(data_loader.dataset), size=sample_size)
    test_random_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=sample_size, sampler=SubsetRandomSampler(indices))
  
    # obtain values
    it = iter(test_random_loader)
    test_batch_data, _ = it.next()
    original = test_batch_data.data.numpy()
    if self.cuda:
      test_batch_data = test_batch_data.cuda()  

    # obtain reconstructed data  
    out = self.network(test_batch_data, self.gumbel_temp, self.hard_gumbel) 
    reconstructed = out['x_rec']
    return original, reconstructed.data.cpu().numpy()


  def plot_latent_space(self, data_loader, save=False):
    """Plot the latent space learnt by the model

    Args:
        data: (array) corresponding array containing the data
        labels: (array) corresponding array containing the labels
        save: (bool) whether to save the latent space plot

    Returns:
        fig: (figure) plot of the latent space
    """
    # obtain the latent features
    features = self.latent_features(data_loader)
    
    # plot only the first 2 dimensions
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(features[:, 0], features[:, 1], c=labels, marker='o',
            edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s = 10)
    plt.colorbar()
    if(save):
        fig.savefig('latent_space.png')
    return fig
  
  
  def random_generation(self, num_elements=1):
    """Random generation for each category

    Args:
        num_elements: (int) number of elements to generate

    Returns:
        generated data according to num_elements
    """ 
    # categories for each element
    arr = np.array([])
    for i in range(self.num_classes):
      arr = np.hstack([arr,np.ones(num_elements) * i] )
    indices = arr.astype(int).tolist()

    categorical = F.one_hot(torch.tensor(indices), self.num_classes).float()
    
    if self.cuda:
      categorical = categorical.cuda()
  
    # infer the gaussian distribution according to the category
    mean, var = self.network.generative.pzy(categorical)

    # gaussian random sample by using the mean and variance
    noise = torch.randn_like(var)
    std = torch.sqrt(var)
    gaussian = mean + noise * std

    # generate new samples with the given gaussian
    generated = self.network.generative.pxz(gaussian)

    return generated.cpu().detach().numpy() 


class Args:
    def __init__(self):
        self.epochs = 50
        self.cuda = 1
        self.gpuID = 0
        self.verbose = 1
        self.batch_size = 64
        self.batch_size_val = 200
        self.learning_rate = 1e-3
        self.decay_epoch = -1
        self.lr_decay = 0.5

        self.w_categ = 1
        self.w_gauss = 1   
        self.w_rec = 1

        self.rec_type = 'cce'
        self.num_classes = 4
        self.gaussian_size = 64
        self.input_size = 740

        self.init_temp = 1.0
        self.decay_temp = 1
        self.hard_gumbel = 0
        self.min_temp = 0.5
        self.decay_temp_rate = 0.013862944

        

class VDJ_dataset(torch.utils.data.Dataset):
    def __init__(self, vdj, labels, task, max_len = None):
        self.task = task
        self.max_len = max_len

        if max_len is None:
            self.max_len = self.get_max_len(vdj)
        else:
            self.max_len = max_len
        vdj = self.add_mask(vdj)
        self.vdj = self.encode_sequences(vdj)

        if labels is not None:
            if self.task == 'binary.classification' or self.task == 'multiclass.classification':
                self.label_to_ix = dict(zip(set(labels), range(len(set(labels)))))
                self.ix_to_label =  dict(zip(range(len(set(labels))), set(labels)))
                self.labels = [self.label_to_ix[label] for label in labels]
                self.n_classes = len(np.unique(self.labels))
                print('Found {} classes for {}'.format(self.n_classes, self.task))

                #self.class_weights = self.make_weights_for_balanced_classes(self.labels)
                #self.labels = [F.one_hot(torch.tensor(label), self.n_classes).float() for label in self.labels]
                self.labels = [torch.tensor(label).int() for label in self.labels]

            elif self.task == 'gex':
                self.label_to_ix = None
                self.ix_to_label = None
                self.labels = np.array(labels)
                self.n_classes = self.labels.shape[-1]        #To do: Add gex normalization!
                print('Found {} genes for {}'.format(self.n_classes, self.task))
                
                self.labels = [torch.Tensor(label) for label in self.labels]


            elif self.task == 'regression':
                self.label_to_ix = None
                self.ix_to_label = None
                self.labels = np.array(labels)
                self.labels = self.z_score(self.min_max(self.labels))
                
        else:
            self.label_to_ix = None
            self.ix_to_label = None
            self.labels = None
            self.n_classes =  None


    def __len__(self):
        return len(self.vdj)

    def __getitem__(self, item):
        vdj = self.vdj[item]
        label = self.labels[item]

        return (vdj, label)
      
    def get_max_len(self, sequences):
        current_len = 0
        for seq in sequences:
            current_len = max(current_len, len(seq))

        return current_len
    
    def add_mask(self, sequences, mask_token = '0'):
        padded = []
        for sequence in sequences:
            padding = [mask_token] * (self.max_len - len(sequence))
            sequence += ''.join(padding)
            padded.append(sequence)
        
        return padded

    def encode_sequences(self, sequences):
        ids = [list(set(i)) for i in sequences]
        ids = set(chain(*ids))
        self.seq_to_ids = dict(zip( ids, list(range(len(ids)))))
        self.ids_to_seq = dict(zip( list(range(len(ids))), ids ))
        self.n_residues = len(ids)

        sequences = [F.one_hot(torch.Tensor([self.seq_to_ids[residue] for residue in sequence]).to(torch.int64), num_classes = self.n_residues).float() for sequence in sequences]
        sequences = [F.pad(sequence, (0,0,0,self.max_len - sequence.size(0))) for sequence in sequences]

        return sequences
    
    def min_max(self, scores):
        return (scores - scores.min()) / (scores.max() - scores.min())

    def z_score(self, scores):
        scores = np.array(scores)
        out = (scores - scores.mean()) / scores.std()
        return scores.tolist()