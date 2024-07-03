import torch
import torch.nn.functional as F

def generate_comparisons(y, noise = .0):
    y_t = y.t()
    y_t = y_t + noise * torch.randn_like(y_t)

    comparison_matrix = (y > y_t).int().float()  # Shape (batch, batch)

    return comparison_matrix

def bt_model(y_hat, beta = 1.):
    comparison_matrix = F.sigmoid(beta * (y_hat - y_hat.t())) #BT model = sigmoid(beta(y_i - y_j))
    return comparison_matrix

def bt_loss(y_hat, y, beta = 1., noise = 0.):

    y = generate_comparisons(y, noise = noise) #Generate comparison given a set of scores of shape (batch)
    y_hat = bt_model(y_hat, beta = beta) #BT model = sigmoid(beta(y_i - y_j))

    loss = F.cross_entropy(y_hat, y)
    return loss