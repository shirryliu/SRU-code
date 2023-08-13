import copy
import json
import os
import pickle
import time
from pathlib import Path

from tqdm import tqdm
import numpy as np

from models import model_factory
from utils import AverageMeterSet, change_args_mode
from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks
from dataloaders.del_data import DelDataloader

import torch
import torch.nn as nn


class AggTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.Model = self.Load_Model(args)

    @classmethod
    def code(cls):
        return 'aggregation'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def train(self):
        T = 0.0
        accum_iter = 0
        for epoch in range(self.args.epoch_agg):
            t0 = time.time()
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            t1 = time.time() - t0
            T += t1
            self.validate(epoch, accum_iter)
        print("Training Time: ", T)

        self.logger_service.complete({
            'state_dict': (self._create_state_dict()),
        })
        self.writer.close()

    def train_one_epoch(self, epoch, accum_iter):
        self.model.train()
        self.lr_scheduler.step()

        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.train_loader)

        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch_size = batch[0].size(0)
            batch = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)
            loss.backward()

            self.optimizer.step()

            average_meter_set.update('loss', loss.item())
            if batch_idx % 10 == 0:
                tqdm_dataloader.set_description(
                    'Epoch {}, loss {:.3f} '.format(epoch + 1, average_meter_set['loss'].avg))

            accum_iter += batch_size

            if self._needs_to_log(accum_iter):
                # tqdm_dataloader.set_description('Logging to Tensorboard')
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                self.log_extra_train_info(log_data)
                self.logger_service.log_train(log_data)

        return accum_iter

    def validate(self, epoch, accum_iter):
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] + \
                                      ['Recall@%d' % k for k in self.metric_ks[:3]]
                description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch,
                'accum_iter': accum_iter,
            }
            log_data.update(average_meter_set.averages())
            self.logger_service.log_val(log_data)

    def test(self):
        print('Test best model with best set!')
        best_model = torch.load(os.path.join(self.export_root, 'models', 'best_acc_model.pth')).get('model_state_dict')
        self.model.load_state_dict(best_model)
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] + \
                                      ['Recall@%d' % k for k in self.metric_ks[:3]]
                description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            average_metrics = average_meter_set.averages()
            # for k, v in average_metrics.items():
            #     if k.startswith('NDCG'):
            #         average_metrics[k] = float(v.numpy())

            filepath = Path(self.export_root).joinpath('test_result.txt')
            with filepath.open('w') as f:
                json.dump(average_metrics, f, indent=2)
            print(average_metrics)
    def calculate_loss(self, batch):
        seqs, labels, u = batch
        logits, p = self.model(seqs, self.Model)  # B x T x V
        # print("logits shape: ", logits.shape)  # 128*200*3706 Batch * Maxlen * Itemnum

        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch):
        seqs, labels, u = batch
        scores, p = self.model(seqs, self.Model)

        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics

    def Load_Model(self, args):
        Model = []
        path = 'experiments/' + args.dataset + '/' + args.sub_model + '/' + str(args.shards)
        prefix = 'test_shard_'
        args = change_args_mode(args, args.model)
        args.bert_dropout = 0
        args.model = args.sub_model
        sub_model = model_factory(args).to(args.device)
        for i in range(args.shards):
            tmp = prefix + args.partition + '_' + str(i) + '_' + str(args.lr) + '_' + \
                  str(args.hidden_units) + '_' + str(args.maxlen) + '_' + str(args.num_epochs)
            if args.del_num != -1:
                tmp = tmp + '_' + args.del_way + '_' + str(args.del_num)
            model_path = os.path.join(path, tmp)
            print(model_path)
            best_model = torch.load(os.path.join(model_path, 'models', 'best_acc_model.pth'),
                                    map_location=torch.device(args.device)).get('model_state_dict')
            sub_model.load_state_dict(best_model)
            Model.append(copy.deepcopy(sub_model))

        args = change_args_mode(args, 'aggregation')
        args.model = 'aggregation'
        return Model

    def get_data_path(self):
        path = self.args.dataset + '_default/' + str(self.args.shards) + '/' \
               + self.args.partition + '_' + str(self.args.num_users) + '_'
        return path

    def test_del_candidates(self):
        best_model = torch.load(os.path.join(self.export_root, 'models', 'best_acc_model.pth')).get('model_state_dict')
        self.model.load_state_dict(best_model)
        path = self.get_data_path()
        prefix = self.args.del_way + '_' + str(self.args.del_num) + '_'
        del_train_data = pickle.load(open(path + prefix + 'del_train_data.pkl', 'rb'))
        del_u2items = pickle.load(open(path + prefix + 'del_u2items.pkl', 'rb'))

        HT1, HT5, HT10, HT20 = 0.0, 0.0, 0.0, 0.0
        valid_user = 0.0
        del_dataloader = DelDataloader(self.args, del_train_data, del_u2items)
        del_dataloaders = del_dataloader.get_del_dataloader()
        for shard in range(self.args.shards):
            del_loader = del_dataloaders[shard]
            # print(self.args.shard_num)
            for batch in del_loader:
                batch = [x.to(self.device) for x in batch]
                seqs, labels, u, candidates = batch
                scores, p = self.model(seqs, self.Model)

                scores = scores.gather(1, candidates)

                scores = scores.cpu()
                labels = labels.cpu()
                rank = (-scores).argsort(dim=1)
                for i in range(labels.size(0)):
                    for j in range(20):
                        if rank[i][j] == 0:
                            if j < 1:
                                HT1 += 1
                            if j < 5:
                                HT5 += 1
                            if j < 10:
                                HT10 += 1
                            if j < 20:
                                HT20 += 1
                    valid_user += 1
        print("CAN.HT@1: %.4f, HT@5: %.4f, HT@10: %.4f, HT@20: %.4f" %
              (HT1 / valid_user, HT5 / valid_user, HT10 / valid_user, HT20 / valid_user))


