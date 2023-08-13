import torch.nn as nn
import copy

from torch import nn as nn
import torch

from models.bert_modules.embedding import BERTEmbedding
from models.bert_modules.transformer import TransformerBlock
from utils import fix_random_seed_as


class BERT4Rec(nn.Module):
    def __init__(self, args):
        super().__init__()
        fix_random_seed_as(args.model_init_seed)

        self.maxlen = args.maxlen
        self.user_num = args.num_users
        self.item_num = args.num_items
        self.n_layers = args.bert_num_blocks
        self.heads = args.bert_num_heads
        self.vocab_size = args.num_items + 2
        self.hidden_units = args.hidden_units
        self.dropout = args.bert_dropout
        self.dev = args.device
        self.cnt = 0

        self.hidden_state = torch.zeros([self.hidden_units], dtype=torch.float).to(args.device)

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=self.vocab_size, embed_size=self.hidden_units, max_len=self.maxlen, dropout=self.dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden_units, self.heads, self.hidden_units * 4, 0) for _ in range(self.n_layers)])
        self.out = nn.Linear(self.hidden_units, args.num_items + 1)

    @classmethod
    def code(cls):
        return 'bert4rec'

    def forward(self, seqs):
        mask = (seqs > 0).unsqueeze(1).repeat(1, seqs.size(1), 1).unsqueeze(1)
        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(seqs)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        final_feat = x[:, -1, :]  # batch * emb
        tmp_hidden_state = final_feat.transpose(0, 1).max(1, keepdim=False)[0]

        self.hidden_state += torch.sum(final_feat, 0)
        self.cnt += final_feat.size(0)

        outputs = self.out(x)
        return outputs

    def get_hidden_state(self, seqs):
        mask = (seqs > 0).unsqueeze(1).repeat(1, seqs.size(1), 1).unsqueeze(1)
        x = self.embedding(seqs)

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        final_feat = x[:, -1, :]  # batch * emb
        return final_feat

    def get_item_embedding(self):
        item_emb = self.embedding.token.weight.data
        return item_emb[:-1, :]

    def centroids_hidden_state(self):
        return self.hidden_state / torch.tensor(self.cnt).to(self.dev)

