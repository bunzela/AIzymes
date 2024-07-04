from sklearn.model_selection import train_test_split
from src.trainers import TrainerGeneral
from src.ff import MLP
import torch

def train_model(dataset,
                training_method = "tmp",
                embedding       = "tmp",
                load_self       = False):
        
    train_df, test_df = train_test_split(dataset.df, test_size = 0.1)

    train_dl = torch.utils.data.DataLoader(train_df, batch_size = 64)
    test_dl = torch.utils.data.DataLoader(test_df, batch_size = len(test_df))
    
    output = {'loss': [], 'score': [], 'dataset': [], 'features': [], 'pearsonr': [], 'pval': []}
    scores_pred = ['norm_total_score']
    feature = 'onehot'
    loss = 'bt'
    input_size = 64 + 1280
    encoding = 'onehot_flatten'

    epochs = 10 #Seems decent ATM - most valid losses increase past this
    n_layers = 2
    n_features = [64,1]
    dropout_prob = 0.3
    model = MLP(input_size, n_layers, n_features, dropout_prob)
    optimizer = torch.optim.Adam



    trainer = TrainerGeneral(model, optimizer, dataset.device, 
                            encoding = encoding, loss = loss, features = feature, 
                            scores = scores_pred, epochs = epochs)
    
    models = trainer.train(train_dl, test_dl)
 
    return "done"   

    #vals = next(iter(test_dl))
    output['loss'].append(loss)
    output['score'].append(score)
    output['dataset'].append(data)
    feature_name = '_'.join(feature)
    output['features'].append(feature_name)
    output['pearsonr'].append(pears)
    output['pval'].append(pvals)
    
    



def junk():

    

    
    
    mc = trainer.predict_mc_dropout(vals, scores_pred, forward_passes = 50)
    pears = pearsonr(mc['y'], mc['mean']).statistic
    pvals = pearsonr(mc['y'], mc['mean']).pvalue
    
