import os

import torch.nn as nn

import copy
import pickle

import tensorflow.compat.v1 as tf
import numpy as np
import torch

from utils import change_args_mode

def get_path(args):
    path = 'experiments/' + args.dataset + '/' + args.sub_model + '/' + str(args.shards) + '/' + args.partition + '_' \
           + str(args.hidden_units) + '_' + str(args.lr) + '_' + str(args.num_epochs) + '_' + str(args.maxlen) + '_'

    if args.del_num != -1:
        path = path + args.del_way + '_' + str(args.del_num) + '_'
    return path


class Aggregation(torch.nn.Module):
    def __init__(self, args):
        super(Aggregation, self).__init__()
        self.attention_size = int(args.hidden_units / 2)
        self.shards = args.shards
        self.relu = torch.nn.ReLU()
        self.softplus = torch.nn.Softplus()
        self.dev = args.device
        self.emb_dim = args.hidden_units
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.sigmoid = torch.nn.Sigmoid()
        self.maxlen = args.maxlen
        self.item_num = args.num_items
        self.args = args

        self.shard_hidden_state = self.load_hidden_state()  # shards * 1 * emb

        self.out = nn.Linear(args.hidden_units, self.item_num + 1)

        # —————————————————— extra weights————————————————#

        # seq attention
        wu = tf.Variable(
            tf.truncated_normal(shape=[int(self.emb_dim), int(self.attention_size)], mean=0.0, stddev=tf.sqrt(
                tf.div(2.0, self.attention_size + self.emb_dim))), dtype=tf.float32)
        bu = tf.Variable(tf.constant(0.00, shape=[self.attention_size]))
        hu = tf.Variable(tf.constant(0.01, shape=[self.attention_size, 1]))
        self.WU = torch.nn.Parameter(torch.from_numpy(wu.numpy()))
        self.BU = torch.nn.Parameter(torch.from_numpy(bu.numpy()))
        self.HU = torch.nn.Parameter(torch.from_numpy(hu.numpy()))

        # trans weights
        initializer = tf.keras.initializers.glorot_normal()
        wt = tf.Variable(initializer([args.shards, self.emb_dim, self.emb_dim]))
        bt = tf.Variable(initializer([args.shards, self.emb_dim]))
        self.WT = torch.nn.Parameter(torch.from_numpy(wt.numpy()))
        self.BT = torch.nn.Parameter(torch.from_numpy(bt.numpy()))

        # ——————————————————-----------————————————————---#

    @classmethod
    def code(cls):
        return 'aggregation'

    def load_hidden_state(self):
        path = get_path(self.args)
        with open(path + 'centroid_hidden_state.pk', 'rb') as fh:
                hidden_state = pickle.load(fh)

        shard_hidden_state = torch.zeros(self.shards, 1, self.emb_dim).to(self.dev)
        for shard in range(self.shards):
            shard_hidden_state[shard][0] = copy.deepcopy(hidden_state[shard])
        shard_hidden_state = shard_hidden_state.transpose(0, 1)

        return shard_hidden_state

    def attention_based_agg(self, embs, shard_embs):
        embs_up = embs * shard_embs
        embs_w = torch.einsum('abd,dk->abk', self.relu(torch.einsum('abd,dk->abk', embs_up, self.WU) + self.BU),
                              self.HU)
        embs_w = torch.exp(embs_w)
        embs_w = torch.div(embs_w, torch.sum(embs_w, 1, keepdim=True))

        agg_emb = torch.sum(torch.multiply(embs_w, embs), 1)  # batch * emb

        return agg_emb, embs_w

    def forward(self, seqs, Model):  # seq: batch * maxlen
        seqs_hidden_state = torch.zeros(self.shards, seqs.size()[0], self.emb_dim).to(self.dev)

        for shard in range(self.shards):
            sub_seq = Model[shard].get_hidden_state(seqs).detach()  # batch * emb
            seqs_hidden_state[shard] = copy.deepcopy(sub_seq)
        seqs_hidden_state = seqs_hidden_state.transpose(0, 1)  # batch * shard * emb

        seq_es = torch.einsum('abc,bcd->abd', seqs_hidden_state, self.WT) + self.BT
        shard_es = torch.einsum('abc,bcd->abd', self.shard_hidden_state, self.WT) + self.BT
        seq_e, seq_w = self.attention_based_agg(seq_es, shard_es)  # batch * emb
        outputs = self.out(seq_e)

        return outputs, seq_w

    def predict(self, test_hidden_state):
        seq_es = torch.einsum('abc,bcd->abd', test_hidden_state, self.WT) + self.BT
        shard_es = torch.einsum('abc,bcd->abd', self.shard_hidden_state, self.WT) + self.BT
        seq_e, seq_w = self.attention_based_agg(seq_es, shard_es)

        return seq_e, seq_w
