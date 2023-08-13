import torch
import torch.utils.data as data_utils
import numpy as np


class DelDataloader:
    def __init__(self, args, del_train_data, del_u2items):
        super(DelDataloader, self).__init__()
        self.args = args
        self.max_len = args.maxlen
        self.del_train_data = del_train_data
        self.del_u2items = del_u2items

    def get_del_dataloader(self):
        all_loader = []
        for shard in range(self.args.shards):
            dataset = DelDataset(self.del_train_data[shard], self.del_u2items[shard], self.args.num_items, self.max_len)
            dataloader = data_utils.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True)
            all_loader.append(dataloader)
        return all_loader


class DelDataset(data_utils.Dataset):
    def __init__(self, del_train_data, del_u2items, num_items, max_length=200, pad_token=0):
        self.u2seq = del_train_data
        self.users = sorted(del_u2items.keys())
        self.max_len = max_length
        self.pad_token = pad_token
        self.u2answer = del_u2items
        self.num_items = num_items

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        candidates = []

        if len(seq) == 0:
            seq = [0]

        candidates.append(self.u2answer[user])
        for _ in range(500):
            item = np.random.choice(self.num_items) + 1
            while item in seq or item in candidates:
                item = np.random.choice(self.num_items) + 1
            candidates.append(item)

        tokens = seq[:]
        answer = self.u2answer[user]
        tokens = tokens[-self.max_len:]

        x_mask_len = self.max_len - len(tokens)

        tokens = [self.pad_token] * x_mask_len + tokens

        return torch.LongTensor(tokens), torch.LongTensor([answer]), torch.LongTensor([user]), torch.LongTensor(candidates)

