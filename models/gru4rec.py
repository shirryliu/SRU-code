import copy

import tensorflow.compat.v1 as tf
import numpy as np
import torch
import torch.nn as nn


class GRU4Rec(torch.nn.Module):
    def __init__(self, args):
        super(GRU4Rec, self).__init__()
        self.user_num = args.num_users
        self.item_num = args.num_items
        self.dev = args.device
        self.args = args
        self.cnt = 0

        self.embedding_dim = args.hidden_units
        self.hidden_dim = args.hidden_units
        self.max_length = args.maxlen
        self.hidden_state = torch.zeros([self.hidden_dim], dtype=torch.float).to(args.device)

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.encoder_layer = nn.GRU(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_dim, args.num_items + 1)

    @classmethod
    def code(cls):
        return 'gru4rec'

    def forward(self, seqs):  # for training
        x = self.item_emb(seqs)
        log_feats, _ = self.encoder_layer(x)  # batch * maxlen * emb

        final_feat = log_feats[:, -1, :]  # batch * emb
        tmp_hidden_state = final_feat.transpose(0, 1).max(1, keepdim=False)[0]

        self.hidden_state += torch.sum(final_feat, 0)
        self.cnt += final_feat.size(0)

        outputs = self.output_layer(log_feats)

        return outputs

    def get_hidden_state(self, log_seqs):
        x = self.item_emb(log_seqs)
        log_feats, _ = self.encoder_layer(x)  # batch * maxlen * emb

        final_feat = log_feats[:, -1, :]  # batch * emb

        return final_feat

    def get_item_embedding(self):
        return self.item_emb.weight.data

    def centroids_hidden_state(self):
        return self.hidden_state / torch.tensor(self.cnt).to(self.dev)
