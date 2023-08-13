import copy

from .base import AbstractDataloader

import torch
import torch.utils.data as data_utils


class BERTDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.max_len = args.maxlen
        self.mask_prob = args.bert_mask_prob
        self.mask_token = self.item_count + 1

    @classmethod
    def code(cls):
        return 'bert4rec'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        agg_loader = self._get_agg_loader()
        tradition_loader = self._get_tradition_loader()
        return train_loader, val_loader, test_loader, agg_loader, tradition_loader, self.user_count, self.item_count

    def _get_train_loader(self):
        all_dataloader = []
        for i in range(self.args.shards + 1):
            dataset = BERTDataset(self.train[i], 'train', self.max_len, self.mask_prob, self.mask_token,
                                  self.item_count, self.rng)
            dataloader = data_utils.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True)
            all_dataloader.append(dataloader)

        return all_dataloader

    def _get_val_loader(self):
        dataset = BERTDataset(self.val, 'valid', self.max_len, self.mask_prob, self.mask_token,
                              self.item_count, self.rng)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.test_batch_size, shuffle=True, pin_memory=True)
        return dataloader

    def _get_test_loader(self):
        dataset = BERTDataset(self.test, 'test', self.max_len, self.mask_prob, self.mask_token,
                              self.item_count, self.rng)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.test_batch_size, shuffle=True, pin_memory=True)
        return dataloader

    def _get_agg_loader(self):
        all_dataloader = []
        for i in range(self.args.shards + 1):
            dataset = BERTDataset(self.train[i], 'test', self.max_len, self.mask_prob, self.mask_token,
                                  self.item_count, self.rng)
            dataloader = data_utils.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True)
            all_dataloader.append(dataloader)

        return all_dataloader

    def _get_tradition_loader(self):
        all_dataloader = []
        for i in range(self.args.shards + 1):
            dataset = BERTTraditionDataset(self.train[i], self.max_len, self.mask_token)
            dataloader = data_utils.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True)
            all_dataloader.append(dataloader)

        return all_dataloader


class BERTDataset(data_utils.Dataset):
    def __init__(self, data, mode, max_len, mask_prob, mask_token, num_items, rng):
        self.data = data
        self.mode = mode
        self.users = sorted(self.data.keys())
        self.max_len = max_len
        self.mask_token = mask_token
        self.mask_prob = mask_prob
        self.num_items = num_items
        self.rng = rng

        self.index2seq, self.index2user, self.index2label = self.prepare()

    def prepare(self):
        index2seq = {}
        index2user = {}
        index2label = {}
        index = 0
        for user in self.users:
            local_seq = []
            for i in range(len(self.data[user]) - 1):
                local_seq.append(self.data[user][i])
                index2seq[index] = copy.deepcopy(local_seq)
                index2user[index] = user
                index2label[index] = self.data[user][i + 1]
                index += 1

        return index2seq, index2user, index2label

    def __len__(self):
        return len(self.index2seq)

    def __getitem__(self, index):
        seq = self.index2seq[index]
        user = self.index2user[index]
        labels = [self.index2label[index]]

        if self.mode == 'train':
            tokens = []
            labels = []
            for s in seq:
                prob = self.rng.random()
                if prob < self.mask_prob:
                    prob /= self.mask_prob

                    if prob < 0.8:
                        tokens.append(self.mask_token)
                    elif prob < 0.9:
                        tokens.append(self.rng.randint(1, self.num_items))
                    else:
                        tokens.append(s)

                    labels.append(s)
                else:
                    tokens.append(s)
                    labels.append(0)

            tokens = tokens[-self.max_len:]
            labels = labels[-self.max_len:]

            mask_len = self.max_len - len(tokens)

            tokens = [0] * mask_len + tokens
            labels = [0] * mask_len + labels
        else:
            seq = seq + [self.mask_token]
            tokens = seq[-self.max_len:]
            padding_len = self.max_len - len(tokens)
            tokens = [0] * padding_len + tokens

        return torch.LongTensor(tokens), torch.LongTensor(labels), torch.LongTensor([index])


class BERTTraditionDataset(data_utils.Dataset):
    def __init__(self, data, max_len, mask_token):
        self.data = data
        self.users = sorted(self.data.keys())
        self.max_len = max_len
        self.mask_token = mask_token

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)
        labels = [seq[-1]]

        tokens = seq[:-1]
        tokens = tokens + [self.mask_token]
        tokens = tokens[-self.max_len:]
        padding_len = self.max_len - len(tokens)
        tokens = [0] * padding_len + tokens

        return torch.LongTensor(tokens), torch.LongTensor(labels), torch.LongTensor([user])

    def _getseq(self, user):
        return self.data[user]

