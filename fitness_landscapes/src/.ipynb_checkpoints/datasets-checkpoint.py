import re

import torch
import torch.nn.functional as F
import torch.utils.data as data

import numpy as np
import pandas as pd

from itertools import chain


class AIzymesDataset(data.Dataset):
    def __init__(self, dataset, tokenizer, scores = [], labels = [], normalize = 'both', select_unique = False):

        self.dataset = dataset[dataset['sequence'].notnull()]
        if select_unique:
            self.dataset = self.select_unique(self.dataset, scores, labels)

        self.tokenizer = tokenizer
        self.sequences = self.dataset['sequence'].tolist()
        self.max_len = self.get_max_len(self.sequences)

        self.embeddings = None #will update when using ESM2
        self.pppl = None #will update when using ESM2
        self.additional_features = None #update to include all possible features in AIZymes for training

        self.scores = None
        if len(scores) > 0:
            self.scores = {}
            for score in scores:
                self.scores[score] = self.dataset[score].to_numpy().astype(np.float32)

                if normalize == 'minmax':
                    self.scores['norm_' + score] = self.min_max(self.scores[score])
                elif normalize == 'both':
                    self.scores['norm_' + score] = self.z_score(self.min_max(self.scores[score]))
                else:
                    self.scores['norm_' + score] = self.z_score(self.scores[score])

        self.labels = None
        if len(labels) > 0:
            self.labels = {}
            for label in labels:
                self.labels[label] = self.dataset[label].tolist()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):

        sequence = str(self.sequences[item])
        processed_sequence = self.preprocess_sequence(sequence)

        encoding = self.tokenizer(
            processed_sequence,
            return_tensors='pt')

        out = {'protein_sequence': processed_sequence,
               'plm_input': encoding}

        if self.scores is not None:
            for key in self.scores.keys():
                out.update({key: self.scores[key][item]})

        if self.labels is not None:
            for key in self.labels.keys():
                out.update({key: self.labels[key][item]})

        if self.embeddings is not None:
            for key in self.embeddings.keys():
                out.update({key: self.embeddings[key][item]})

        if self.pppl is not None:
            for key in self.pppl.keys():
                out.update({key: self.pppl[key][item]})
                
        return out

    def get_max_len(self, sequences):
        lengths = np.array([len(seq) for seq in sequences])
        return np.max(lengths)

    def preprocess_sequence(self, sequence):
        sequence = re.sub(r"[UZOB]", "X", ''.join(sequence.upper()))
        return sequence

    def mask_sequence(self, input_ids, mask_prob = 0.15, replace_w_mask_token = 0.8, replace_w_aa = 0.1, keep_same = 0.1):

        assert mask_prob <= 1
        assert replace_w_mask_token + replace_w_aa + keep_same == 1

        label_ids = input_ids.clone()
        mask = np.random.binomial(1, mask_prob, input_ids[input_ids > 4].size())

        label_ids[np.where(mask == 0)[0] + 1] = -100
        label_ids[0] = -100
        label_ids[len(mask) + 1] = -100

        for id in (np.where(mask)[0] + 1):
            mask_type = np.random.choice(['mask', 'aa', 'same'], p = [replace_w_mask_token, replace_w_aa, keep_same])

            if mask_type == 'mask':
                input_ids[id] = self.tokenizer.mask_token_id

            elif mask_type == 'aa':
                prev_aa = input_ids[id]
                all_aas = np.arange(5,30)
                all_aas = all_aas[all_aas != prev_aa]
                input_ids.numpy()[id] = np.random.choice(all_aas[0], 1)


        return input_ids, label_ids

    def mask_catalytic():
        pass

    def mask_backbone():
        pass

    def min_max(self, scores):
        return (scores - scores.min()) / (scores.max() - scores.min())

    def z_score(self, scores):
        scores = np.array(scores)
        out = (scores - scores.mean()) / scores.std()
        return scores.tolist()

    def select_unique(self, df, scores, labels):
        score_df = df.groupby('sequence')[scores].mean().reset_index()
        label_df = df.drop_duplicates(['sequence'])[labels]
        df = pd.merge(score_df, label_df, on='sequence')
  
        return df
    
    def onehot_sequences(self, sequences):
        ids = [list(set(i)) for i in sequences]
        ids = set(chain(*ids))
        self.seq_to_ids = dict(zip( ids, list(range(len(ids)))))
        self.ids_to_seq = dict(zip( list(range(len(ids))), ids ))
        self.n_residues = len(ids)

        sequences = [F.one_hot(torch.Tensor([self.seq_to_ids[residue] for residue in sequence]).to(torch.int64), num_classes = self.n_residues).float() for sequence in sequences]
        sequences = [F.pad(sequence, (0,0,0,self.max_len - sequence.size(0))) for sequence in sequences]

        return sequences