#from generative.vae import Args, VDJ_dataset, GMVAE

#import shap
import torch
import os
import pandas as pd
from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
#from umap import UMAP
import re
import torch
import torch.nn.functional as F
#import torch.utils.data as data
import numpy as np
import pandas as pd
from itertools import chain
from transformers import EsmTokenizer, EsmModel, AutoTokenizer, EsmForMaskedLM
from tqdm import tqdm
import pickle
from src.tools import setup_torch, reset_cuda

print("Fitness Landscape Tools loaded.")


class make_dataset():

    def __init__(self, 
                 output_folder   = './AIzymes_resi14'):

        #set variables
        self.output_folder   = output_folder

        #make folder
        os.makedirs(output_folder, exist_ok=True)

    def load_dataset(self,
                     df_path = './data/all_scores_pooled_cut.csv',
                     scores          = ['interface_score', 'total_score', 'interface_potential', 'total_potential'],
                     labels          = ['score_taken_from', 'design_method', 'cat_resn', 'cat_resi', 'parent_index', 'generation', 'mutations'],
                     cat_resi        = 14,
                     select_unique   = True,
                     normalize       = 'both'):

        #set variables
        self.df_path         = df_path
        self.scores          = scores
        self.labels          = labels
        self.cat_resi        = cat_resi
        self.select_unique   = select_unique
        self.normalize       = normalize

        #Read in dataset
        self.df, self.lengths, self.max_len, self.sequences = self.read_datasets_csv()

        #run normalization
        self.df = self.normalize_scores()

        return self.df
    
    def make_embeddings(self,
                        embeddings     = ['onehot','plm','plm_pca','onehot_plm_pca'],
                        pooling_method = 'concatenate',
                        pca_dim        = 50,
                        load_self      = False):
  
        #set variables      
        self.embeddings      = embeddings
        self.pooling_method  = pooling_method
        self.pca_dim         = pca_dim
        
        #initialize torch
        self.device, self.dtype, self.SMOKE_TEST = setup_torch()

        if 'onehot' in self.embeddings:
            self.df = self.onehot_sequences()
        if 'plm' in self.embeddings:
            self.df = self.plm_sequences()
        if 'plm_pca' in self.embeddings:
            self.df = self.plm_pca()
        if 'onehot_plm_pca' in self.embeddings:
            self.df = self.onehot_plm_pca()   

        print(f'Embeddings done for {", ".join(self.embeddings)}')

        self.save_self_to_file()

        torch.cuda.empty_cache()

        return self.df

    def onehot_plm_pca(self):

        if 'plm_pca' not in self.embeddings: 
             print(f'Error! Embedding "plm_pca" needed to create "onehot_plm_pca"')
        if 'onehot' not in self.embeddings: 
            print(f'Error! Embedding "onehot" needed to create "onehot_plm_pca"')
    
        self.df['onehot'] = self.df['onehot'].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [])
        self.df['plm_pca'] = self.df['plm_pca'].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [])
        self.df['onehot_plm_pca'] = self.df.apply(lambda row: row['onehot'] + row['plm_pca'], axis=1)

        return self.df
       
    def plm_pca(self):
                  
        if 'plm' not in self.embeddings: 
            print(f'Error! Embedding "plm" needed to create "plm_pca"')

        pca = PCA(n_components = self.pca_dim)
        embeddings = np.stack(self.df['plm'])
        proj = pca.fit_transform(embeddings)
        embeddings = [proj[i] for i in range(proj.shape[0])]                                                    

        self.df['plm_pca'] = embeddings

        return self.df

    def plm_sequences(self):

        self.plm = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D") #Will update to the better ones/ more parameters
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.plm.to(self.device)

        embeddings = []
        for sequence, _ in zip(self.sequences, tqdm(range(len(self.sequences)))): #uses tqdm to display progress bar, output not used in code
            sequence = sequence.upper()
            tokenized_sequence = self.tokenizer(sequence, return_tensors= 'pt').to(self.device)
            output = self.plm(**tokenized_sequence)
            embeddings.append(self.pool_output(output))
            
        self.df['plm'] = embeddings

        return self.df
    
    def pool_output(self, output):

        #last hidden state. removed class token. only embedd residues. average accross all residues
        if self.pooling_method == 'average':
            output = torch.mean(output.last_hidden_state[:, 1:-1, :].squeeze(0), axis = 1)

        #results in one embedding for whole sequence. Standard option, double check what this is doing
        elif self.pooling_method == 'pooler':
            output = output.pooler_output[0]

        #class token from last hidden state
        elif self.pooling_method == 'first':
            output = output.last_hidden_state[0,0,:]

        #last hidden state. removed class token. only embedd residues.        
        elif self.pooling_method == 'concatenate':
            output = output.last_hidden_state[:, 1:-1, :].squeeze(0)
            output = output.reshape(-1)
        
        else:
            raise ValueError(f'Pooling method {self.pooling_method} not implemented')
            
        return output.cpu().detach().numpy()
                            
    def onehot_sequences(self):

        ids = [list(set(i)) for i in self.sequences]
        ids = set(chain(*ids))
        seq_to_ids = dict(zip( ids, list(range(len(ids)))))
        n_residues = len(ids)

        sequences = [F.one_hot(torch.Tensor([seq_to_ids[residue] for residue in sequence]).to(torch.int64), num_classes = n_residues).int() for sequence in self.sequences]
        sequences = [F.pad(sequence, (0,0,0,self.max_len - sequence.size(0))) for sequence in sequences]
        sequences = [np.array(sequence).flatten().tolist() for sequence in sequences]
        self.df['onehot'] = sequences

        return self.df
                                                
    def read_datasets_csv(self):

        df = pd.read_csv(self.df_path)
        df = df[df['sequence'].notnull()]
        
        if self.select_unique:
            score_df = df.groupby('sequence')[self.scores].mean().reset_index()
            label_df = df.drop_duplicates('sequence')[self.labels + ['sequence']]
            df = pd.merge(score_df, label_df, on='sequence')

        if self.cat_resi != None:
            df = df[df['cat_resi'] == self.cat_resi]

        sequences = df['sequence'].tolist()
        lengths = np.array([len(seq) for seq in sequences])
        max_len = np.max(lengths)

        print(f'Data loaded from: {self.df_path}')

        return df, lengths, max_len, sequences

    def normalize_scores(self):

        for score in self.scores:
                score_list = self.df[score].to_numpy().astype(np.float32)
                if self.normalize == 'minmax':
                    self.df[f'norm_{score}'] = self.min_max(score_list)
                if self.normalize == 'both':
                    score_list = self.min_max(score_list)
                    self.df[f'norm_{score}'] = self.z_score(score_list)
                if self.normalize == 'z_score':
                    self.df[f'norm_{score}'] = self.z_score(score_list)
                    
        print('Data normalized.')

        return self.df

    def min_max(self, scores):
        return (scores - scores.min()) / (scores.max() - scores.min())
    
    def z_score(self, scores):
        return (scores - scores.mean()) / scores.std()
    
    def save_self_to_file(self):
        with open(f'{self.output_folder}/self.pkl', 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load_self_from_file(self, folder):
        with open(f'{folder}/self.pkl', 'rb') as f:
            self.__dict__.update(pickle.load(f))
        print(f'All variables loaded from {self.output_folder}/self.pkl')

        return self.df
