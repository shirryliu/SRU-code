import copy
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from utils import *
from config import RAW_DATASET_ROOT_FOLDER

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

from abc import *
from pathlib import Path
import os
import tempfile
import shutil
import pickle


class MyDataset:
    def __init__(self, args):
        self.num_items = 0
        self.num_users = 0
        self.args = args

    def data_load(self, f):
        # assume user/item index starting from 1
        User = defaultdict(list)
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            self.num_users = max(u, self.num_users)
            self.num_items = max(i, self.num_items)
            User[u].append(i)
        return User

    def data_partition(self):
        f_train = open('data/%s_train.txt' % self.args.dataset, 'r')
        f_valid = open('data/%s_valid.txt' % self.args.dataset, 'r')
        f_test = open('data/%s_test.txt' % self.args.dataset, 'r')

        user_train = self.data_load(f_train)
        user_valid = self.data_load(f_valid)
        user_test = self.data_load(f_test)

        return [user_train, user_valid, user_test, self.num_users, self.num_items]

    def get_data_path(self):
        path = self.args.dataset + '_default/' + str(self.args.shards) + '/' \
               + self.args.partition + '_' + str(self.num_users) + '_'
        if not os.path.exists(self.args.dataset + '_default/' + str(self.args.shards) + '/'):
            os.mkdir(self.args.dataset + '_default/' + str(self.args.shards) + '/')
        return path

    def random_data_partition(self, user_train):
        path = self.get_data_path()
        if os.path.exists(path + 'train_data.pk'):
            with open(path + 'train_data.pk', 'rb') as fh:
                train_data = pickle.load(fh)
            return train_data
        u_list = list(user_train.keys())
        shard_size = int(len(u_list) / self.args.shards) + 1
        uid = torch.randperm(len(u_list))
        train_data = []
        last = 0
        for shard in range(self.args.shards):
            train_data.append({})
            for i in range(shard_size):
                id = i + last
                if id >= len(u_list):
                    break
                u = u_list[int(uid[id])]
                train_data[shard][u] = user_train[u]
            last = (shard + 1) * shard_size

        with open(path + 'train_data.pk', 'wb+') as fh:
            pickle.dump(train_data, fh)
        return train_data

    def k_means_data_partition(self, user_train, T):
        path = self.get_data_path()
        if os.path.exists(path + 'users_partition.pk'):
            with open(path + 'users_partition.pk', 'rb') as fh:
                users_partition = pickle.load(fh)
            return users_partition

        data = user_train

        path = self.args.dataset + '_default/'
        with open(path + self.args.sub_model + '_hidden_state.pk', 'rb') as fh:
            hidden_state = pickle.load(fh)

        # Randomly select k centroids
        max_data = len(data) / self.args.shards + 5

        centroids = random.sample(data.keys(), self.args.shards)
        centroembs = []
        for i in range(self.args.shards):
            temp_u = hidden_state[centroids[i]].cpu().numpy()
            centroembs.append(temp_u)

        # K-means
        for _ in range(T):
            C = [{} for _ in range(self.args.shards)]
            Scores = {}
            for i in data.keys():
                for j in range(self.args.shards):
                    score_u = self.E_score2(hidden_state[i].cpu().numpy(), centroembs[j])

                    Scores[i, j] = -score_u

            Scores = sorted(Scores.items(), key=lambda x: x[1], reverse=True)  # [((i,j),dis),((i,j),dis)...]

            fl = set()
            for i in range(len(Scores)):
                if Scores[i][0][0] not in fl:  # i
                    if len(C[Scores[i][0][1]]) <= max_data:
                        C[Scores[i][0][1]][Scores[i][0][0]] = data[Scores[i][0][0]]
                        fl.add(Scores[i][0][0])

            centroembs_next = []
            for i in range(self.args.shards):
                temp_u = []
                for u in C[i].keys():
                    temp_u.append(hidden_state[u].cpu().numpy())
                centroembs_next.append(np.mean(temp_u))

            loss = 0.0

            for i in range(self.args.shards):
                score_u = self.E_score2(centroembs_next[i], centroembs[i])
                loss += score_u

            centroembs = centroembs_next
            # print _, loss

        users_partition = [[] for _ in range(self.args.shards)]
        for i in range(self.args.shards):
            users_partition[i] = list(C[i].keys())

        path = self.get_data_path()
        with open(path + 'users_partition.pk', 'wb+') as fh:
            pickle.dump(users_partition, fh)
        return users_partition

    def E_score2(self, a, b):
        return np.sum(np.power(a - b, 2))

    def get_shard_data(self, user_train, user_partition, shards_num):
        train_data = []

        for shard in range(shards_num):
            train_data.append({})

            for u in user_partition[shard]:
                train_data[shard][u] = user_train[u]

        return train_data

    def subdata_delete(self, train_data, args):  # 每个shard删100条数据
        path = self.get_data_path()
        prefix = args.del_way + '_' + str(args.del_num) + '_'
        if not os.path.exists(path + prefix + 'del_train_data.pkl'):
            del_train_data = copy.deepcopy(train_data)
            del_u2items = []
            for i in range(args.shards):
                del_u2items.append({})
                del_id = torch.randperm(len(train_data[i]))
                del_len = int(len(train_data[i]) / 5)
                del_id = del_id[:del_len]

                k = 0
                for u in train_data[i]:
                    if k not in del_id:
                        k += 1
                        continue

                    len1 = len(train_data[i][u])
                    pos = torch.randint(int(len1 * 0.5), int(len1), (1,))
                    label = del_train_data[i][u][pos[0]]
                    del_train_data[i][u] = del_train_data[i][u][:pos - 1]
                    del_train_data[args.shards][u] = del_train_data[args.shards][u][:pos - 1]

                    del_u2items[i][u] = label
                    k += 1

            pickle.dump(del_train_data, open(path + prefix + 'del_train_data.pkl', 'wb'))
            pickle.dump(del_u2items, open(path + prefix + 'del_u2items.pkl', 'wb'))

        del_train_data = pickle.load(open(path + prefix + 'del_train_data.pkl', 'rb'))
        del_u2items = pickle.load(open(path + prefix + 'del_u2items.pkl', 'rb'))

        path = self.args.dataset + '_default/'
        with open(path + self.args.sub_model + '_item_emb.pk', 'rb') as fi:
            item_embedding = pickle.load(fi)

        for i in range(args.shards):
            if args.del_num == 0:
                continue
            for u in del_u2items[i].keys():

                if args.del_way == 'random':  # random
                    random_id = torch.randperm(len(del_train_data[i][u]))
                    random_id = random_id[:args.del_num]
                    tmp = []
                    for j in range(len(del_train_data[i][u])):
                        if j not in random_id:
                            tmp.append(del_train_data[i][u][j])

                    del_train_data[i][u] = copy.deepcopy(tmp)
                    del_train_data[args.shards][u] = copy.deepcopy(tmp)

                elif args.del_way == 'similarity':  # similarity
                    sim = {}
                    for j in range(len(del_train_data[i][u])):  # 计算余弦相似度
                        item = del_u2items[i][u]
                        sim[j] = F.cosine_similarity(item_embedding[item],
                                                     item_embedding[del_train_data[i][u][j]],
                                                     dim=0).cpu().numpy()
                        sim[j] = abs(sim[j])
                    simf = sorted(sim.items(), key=lambda x: x[1], reverse=True)  # 排序
                    random_id = []
                    for (k, v) in simf:
                        random_id.append(k)
                    random_id = random_id[:args.del_num]
                    tmp = []
                    for j in range(len(del_train_data[i][u])):
                        if j not in random_id:
                            tmp.append(del_train_data[i][u][j])

                    del_train_data[i][u] = copy.deepcopy(tmp)
                    del_train_data[args.shards][u] = copy.deepcopy(tmp)

                else:  # neighbor
                    del_train_data[i][u] = copy.deepcopy(del_train_data[i][u][:-args.del_num])  # 最相近的
                    del_train_data[args.shards][u] = copy.deepcopy(del_train_data[i][u][:-args.del_num])

        return del_train_data

    def load_dataset(self):
        dataset = self.data_partition()
        [user_train, user_valid, user_test, num_users, num_items] = dataset
        if self.args.partition == 'random':
            random_train_data = self.random_data_partition(user_train)
            random_train_data.append(user_train)
            dataset = [random_train_data, user_valid, user_test, num_users, num_items]
            if self.args.del_num != -1:
                train_data1 = self.subdata_delete(random_train_data, self.args)
                dataset = [train_data1, user_valid, user_test, num_users, num_items]

        else:
            user_partition = self.k_means_data_partition(user_train, 50)
            k_means_train_data = self.get_shard_data(user_train, user_partition, self.args.shards)
            k_means_train_data.append(user_train)
            dataset = [k_means_train_data, user_valid, user_test, num_users, num_items]
            if self.args.del_num != -1:
                train_data1 = self.subdata_delete(k_means_train_data, self.args)
                dataset = [train_data1, user_valid, user_test, num_users, num_items]
        return dataset
