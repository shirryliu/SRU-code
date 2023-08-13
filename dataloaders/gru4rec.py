import copy

from .base import AbstractDataloader

import torch
import torch.utils.data as data_utils


class GRUDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.max_len = args.maxlen

    @classmethod
    def code(cls):
        return 'gru4rec'

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
            dataset = GRUDataset(self.train[i], 'train', self.max_len)
            dataloader = data_utils.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True)
            all_dataloader.append(dataloader)

        return all_dataloader

    def _get_val_loader(self):
        dataset = GRUDataset(self.val, 'valid', self.max_len)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.test_batch_size, shuffle=True, pin_memory=True)
        return dataloader

    def _get_test_loader(self):
        dataset = GRUDataset(self.test, 'test', self.max_len)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.test_batch_size, shuffle=True, pin_memory=True)
        return dataloader

    def _get_agg_loader(self):
        all_dataloader = []
        for i in range(self.args.shards + 1):
            dataset = GRUDataset(self.train[i], 'test', self.max_len)
            dataloader = data_utils.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True)
            all_dataloader.append(dataloader)

        return all_dataloader

    def _get_tradition_loader(self):
        all_dataloader = []
        for i in range(self.args.shards + 1):
            dataset = GRUTraditionDataset(self.train[i], self.max_len)
            dataloader = data_utils.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True)
            all_dataloader.append(dataloader)

        return all_dataloader


class GRUDataset(data_utils.Dataset):
    def __init__(self, data, mode, max_length=200, pad_token=0):
        self.data = data
        self.users = sorted(self.data.keys())
        self.max_len = max_length
        self.pad_token = pad_token
        self.mode = mode

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

        # for i in range(len(index2seq)):
        #     print(index2seq[i])
        #     print(index2user[i])
        #     print(index2label[i])
        return index2seq, index2user, index2label

    def __len__(self):
        return len(self.index2seq)

    def __getitem__(self, index):
        seq = self.index2seq[index]
        user = self.index2user[index]
        label = [self.index2label[index]]

        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        if self.mode == 'train':
            label = [0] * (self.max_len - len(label)) + label

        return torch.LongTensor(seq), torch.LongTensor(label), torch.LongTensor([index])


class GRUTraditionDataset(data_utils.Dataset):
    def __init__(self, data, max_length=200, pad_token=0):
        self.data = data
        self.users = sorted(self.data.keys())
        self.max_len = max_length
        self.pad_token = pad_token

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)
        label = [seq[-1]]

        tokens = seq[:-1]
        tokens = tokens[-self.max_len:]
        padding_len = self.max_len - len(tokens)
        tokens = [0] * padding_len + tokens

        return torch.LongTensor(tokens), torch.LongTensor(label), torch.LongTensor([user])

    def _getseq(self, user):
        return self.data[user]
