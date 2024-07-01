from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

from transformers import EsmTokenizer, EsmModel, AutoTokenizer, EsmForMaskedLM


class EnzymePLM(nn.Module):
    def __init__(self, device, plm = 'esm2', pooling_method = 'average'):
        super(EnzymePLM, self).__init__()
        self.device = device

        if plm == 'esm2':
            self.plm = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D") #Will update to the better ones/ more parameters
            self.plm.to(device)

        elif plm == 'esm2_masked':
            self.masked_plm = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
            self.masked_plm.to(device)

        else:
            pass

        self.pooling_method = pooling_method

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

    def embeddings_all(self, ds):
        sequences = ds.sequences
        embeddings = []
        for sequence, _ in zip(enumerate(sequences), tqdm(range(len(sequences)))):
            sequence = ds.preprocess_sequence(sequence[1])
            tokenized_sequence = ds.tokenizer(sequence, return_tensors= 'pt').to(self.device)
            output = self.plm(**tokenized_sequence)
            embeddings.append(self.pool_output(output))

        return embeddings

    def pool_output(self, output):

        if self.pooling_method == "average":
            output = torch.mean(output.last_hidden_state[:, 1:-1, :].squeeze(0), axis = 1)

        elif self.pooling_method == "pooler":
            output = output.pooler_output[0]

        elif self.pooling_method == "last":
            output = output.last_hidden_state[0,-1,:]

        elif self.pooling_method == "first" or self.pooling_method == "class": #similar to pooler but without layernorm
            output = output.last_hidden_state[0,0,:]
        
        elif self.pooling_method == "concatenate":
            output = output.last_hidden_state[:, 1:-1, :].squeeze(0)
            output = output.reshape(-1)
        
        elif self.pooling_method == "concatenate_class":
            output = output.last_hidden_state[:, :-1, :].squeeze(0)
            output = output.reshape(-1)
            
        elif self.pooling_method == "average_class":
            output = torch.mean(output.last_hidden_state[:, :-1, :].squeeze(0), axis = 1)
            
        elif self.pooling_method == 'pca_concatenate':
            pass
        elif self.pooling_method = 'pca_concatenate_class':
            pass

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

