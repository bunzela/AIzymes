import re

import torch
import torch.nn.functional as F
import torch.utils.data as data

import numpy as np
import pandas as pd

from itertools import chain


class AIzymesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, scores = [], labels = [], normalize = 'both', select_unique = True):

        self.dataset = dataset[dataset['sequence'].notnull()]
        if select_unique:
            self.dataset = self.select_unique(self.dataset, scores, labels)

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

        out = {'protein_sequence': processed_sequence}

        if self.scores is not None:
            for key in self.scores.keys():
                out.update({key: torch.tensor(self.scores[key][item]).unsqueeze(-1)})

        if self.labels is not None:
            for key in self.labels.keys():
                out.update({key: self.labels[key][item]})

        if self.embeddings is not None:
            for key in self.embeddings.keys():
                out.update({key: self.embeddings[key][item]})

        if self.pppl is not None:
            for key in self.pppl.keys():
                out.update({key: self.pppl[key][item]})

        if self.additional_features is not None:
            for key in self.additional_features.keys():
                out.update({key: self.additiona[key][item]})

        return out

    def get_max_len(self, sequences):
        lengths = np.array([len(seq) for seq in sequences])

        return np.max(lengths)

    def preprocess_sequence(self, sequence):
        sequence = re.sub(r"[UZOB]", "X", ''.join(sequence.upper()))

        return sequence

    def min_max(self, scores):

        return (scores - scores.min()) / (scores.max() - scores.min())

    def z_score(self, scores):
        scores = np.array(scores)
        out = (scores - scores.mean()) / scores.std()

        return scores.tolist()

    def select_unique(self, df, scores, labels):
        score_df = df.groupby('sequence')[scores].mean().reset_index()
        label_df = df.drop_duplicates('sequence')[labels + ['sequence']]
        df = pd.merge(score_df, label_df, on='sequence')

        return df

    def onehot_sequences(self):
        sequences = self.sequences
        ids = [list(set(i)) for i in sequences]
        ids = set(chain(*ids))
        self.seq_to_ids = dict(zip( ids, list(range(len(ids)))))
        self.ids_to_seq = dict(zip( list(range(len(ids))), ids ))
        self.n_residues = len(ids)

        sequences = [F.one_hot(torch.Tensor([self.seq_to_ids[residue] for residue in sequence]).to(torch.int64), num_classes = self.n_residues).float() for sequence in sequences]
        sequences = [F.pad(sequence, (0,0,0,self.max_len - sequence.size(0))) for sequence in sequences]
        
        if self.embeddings is None:
            self.embeddings = {}
        
        self.embeddings['onehot'] = sequences
        
        
    def embed_sequences(self, plm, pooling_methods, vae = None):
                                                           
        if self.embeddings is None:
            self.embeddings = {}
                                                                 
        if plm is not None:
            for method in pooling_methods:
                self.embeddings[method] = plm.embeddings_all(self.sequences, pooling_method = method)                                   

