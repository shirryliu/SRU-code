from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks

import torch.nn as nn
import torch
import os
import pickle
from dataloaders.del_data import DelDataloader


class Trainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

    @classmethod
    def code(cls):
        return 'sgb'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        seqs, labels, u = batch
        logits = self.model(seqs)  # B x T x V

        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # Batch * Itemnum

        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch):
        seqs, labels, u = batch
        scores = self.model(seqs)  # B x T x V
        scores = scores[:, -1, :]  # B x V

        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics, labels.size(0)

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
            for batch in del_loader:
                batch = [x.to(self.device) for x in batch]
                seqs, labels, u, candidates = batch
                scores = self.model(seqs)

                scores = scores[:, -1, :]  # B x V
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


