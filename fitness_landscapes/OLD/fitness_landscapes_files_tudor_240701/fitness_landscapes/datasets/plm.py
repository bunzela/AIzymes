from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

from transformers import EsmTokenizer, EsmModel, AutoTokenizer, EsmForMaskedLM

from sklearn.decomposition import PCA

import re
import numpy as np


class EnzymePLM(nn.Module):
    def __init__(self, device, plm = 'esm2', pooling_method = 'average', pca_dim = 50):
        super(EnzymePLM, self).__init__()
        self.device = device

        if plm == 'esm2':
            self.plm = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D") #Will update to the better ones/ more parameters
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
            self.plm.to(device)

        elif plm == 'esm2_masked':
            self.masked_plm = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
            self.masked_plm.to(device)

        else:
            pass

        self.pooling_method = pooling_method
        self.pca_dim = pca_dim

    def forward(self, x):
        return self.plm(**x['plm_input'])

    def get_embeddings(self, x):
        pass

    def pppl_all(self, ds):
        sequences = ds.sequences
        pppl = []

        for sequence, _ in zip(enumerate(sequences), tqdm(range(len(sequences)))):
            pppl.append(self.calculate_pppl(sequence[1], ds.tokenizer))

        return pppl

    def embeddings_all(self, sequences, pooling_method = None):
        if pooling_method is not None:
            self.pooling_method = pooling_method
            
        embeddings = []
        for sequence, _ in zip(enumerate(sequences), tqdm(range(len(sequences)))):
            sequence = self.preprocess_sequence(sequence[1])
            tokenized_sequence = self.tokenizer(sequence, return_tensors= 'pt').to(self.device)
            output = self.plm(**tokenized_sequence)
            embeddings.append(self.pool_output(output))
            
        if (self.pca_dim is not None) and (self.pooling_method in ['pca_average', 'pca_pooler', 'pca_last', 'pca_first', 'pca_class', 'pca_concatenate']):
            pca = PCA(n_components = self.pca_dim)
            embeddings = np.stack(embeddings)
            proj = pca.fit_transform(embeddings)
            embeddings = [proj[i] for i in range(proj.shape[0])]                                                    

        return embeddings

    def pool_output(self, output):

        if self.pooling_method in ['average', 'pca_average']:
            output = torch.mean(output.last_hidden_state[:, 1:-1, :].squeeze(0), axis = 1)

        elif self.pooling_method in ['pooler', 'pca_pooler']:
            output = output.pooler_output[0]

        elif self.pooling_method in ['last', 'pca_last']:
            output = output.last_hidden_state[0,-1,:]

        elif self.pooling_method in ['first', 'pca_first', 'class', 'pca_class']:
            output = output.last_hidden_state[0,0,:]
        
        elif self.pooling_method in ['concatenate', 'pca_concatenate']:
            output = output.last_hidden_state[:, 1:-1, :].squeeze(0)
            output = output.reshape(-1)
        
        else:
            raise ValueError(f'Pooling method {self.pooling_method} not implemented')
            
        return output.cpu().detach().numpy()

    def calculate_pppl(self, sequence, tokenizer):
        token_ids = tokenizer.encode(sequence, return_tensors='pt').to(self.device)
        input_length = token_ids.size(1)
        log_likelihood = 0.0

        for i in range(input_length):
            # Create a copy of the token IDs
            masked_token_ids = token_ids.clone()
            # Mask a token that we will try to predict back
            masked_token_ids[0, i] = tokenizer.mask_token_id

            with torch.no_grad():
                output = self.masked_plm(masked_token_ids)
                logit_prob = torch.nn.functional.log_softmax(output.logits, dim=-1)

            log_likelihood += logit_prob[0, i, token_ids[0, i]]

        # Calculate the average log likelihood per token
        avg_log_likelihood = log_likelihood / input_length

        # Compute and return the pseudo-perplexity
        pppl = torch.exp(-avg_log_likelihood)
        return pppl.item()
    
    
    def preprocess_sequence(self, sequence):
        sequence = re.sub(r"[UZOB]", "X", ''.join(sequence.upper()))

        return sequence

