import copy
import os.path
import pickle
import time

import torch
from tqdm import tqdm

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *


def get_path(args):
    path = 'experiments/' + args.dataset + '/' + args.sub_model + '/' + str(args.shards) + '/' + args.partition + '_' + \
           str(args.hidden_units) + '_' + str(args.lr) + '_' + str(args.num_epochs) + '_' + str(args.maxlen) + '_'
    if args.del_num != -1:
        path = path + args.del_way + '_' + str(args.del_num) + '_'
    return path


def train_sub_model(args):
    print("HELLO!!!!############################################################")
    args.sub_model = args.model
    train_loader, val_loader, test_loader, agg_loader, tradition_loader, num_users, num_items = dataloader_factory(args)
    centroids_hidden_state = []
    max_hidden_state = []

    t = time.time()
    for shard in range(args.shards):
        print('Start train shard %d' % shard)
        export_root = setup_train(args, shard)
        model = model_factory(args)
        trainer = trainer_factory(args, model, train_loader[shard], val_loader, test_loader, export_root)
        trainer.train()
        trainer.test()
        shard_hidden_state, shard_max_hidden_state = model.centroids_hidden_state()
        centroids_hidden_state.append(copy.deepcopy(shard_hidden_state.detach()))
        max_hidden_state.append(copy.deepcopy(shard_max_hidden_state.detach()))
    T = time.time()
    print('ALL TIME: ', T - t)
    path = get_path(args)
    with open(path + 'centroid_hidden_state.pk', 'wb+') as fh:
        pickle.dump(centroids_hidden_state, fh)


def train_aggregation_model(args):
    print("Start Aggregation")
    train_loader, val_loader, test_loader, agg_loader, tradition_loader, num_users, num_items = dataloader_factory(args)
    print(num_users, num_items)
    t = time.time()
    args = change_args_mode(args, 'aggregation')
    args.experiment_description = 'aggregation'
    args.sub_model = args.model
    args.model = 'aggregation'

    export_root = setup_train(args, args.shards)
    print(export_root)
    model = model_factory(args)
    trainer = trainer_factory(args, model, agg_loader[args.shards], val_loader, test_loader, export_root)
    trainer.train()

    T = time.time()
    print('ALL TIME: ', T - t)

    print("Start Test")
    trainer.test()
    trainer.test_last()

    if args.del_num != -1:
        trainer.test_del_candidates()


def make_path(args):
    if not os.path.isdir(args.dataset + '_default'):
        os.makedirs(args.dataset + '_default')
    path = os.path.join(args.dataset + '_default', str(args.shards))
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    make_path(args)
    train_sub_model(args)
    train_aggregation_model(args)
