import argparse


parser = argparse.ArgumentParser(description='RecPlay')

################
# Top Level
################
parser.add_argument('--mode', type=str, default='train', choices=['train'])
parser.add_argument('--template', type=str, default='train_bert')
################
# Test
################
parser.add_argument('--test_model_path', type=str, default=None)

################
# Dataset
################
parser.add_argument('--dataset', type=str, default='ml1m')
# parser.add_argument('--min_rating', type=int, default=4, help='Only keep ratings greater than equal to this value')
# parser.add_argument('--min_uc', type=int, default=5, help='Only keep users with more than min_uc ratings')
# parser.add_argument('--min_sc', type=int, default=0, help='Only keep items with more than min_sc ratings')
# parser.add_argument('--split', type=str, default='leave_one_out', help='How to split the datasets')

################
# Dataloader
################
# parser.add_argument('--dataloader_code', type=str, default='bert', choices=['bert', 'sg'])
parser.add_argument('--dataloader_random_seed', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--val_batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=2048)

################
# Trainer
################
parser.add_argument('--trainer_code', type=str, default='sgb', choices=['sgb', 'aggregation'])
# device #
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--device_idx', type=str, default='1')
# optimizer #
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
# lr scheduler #
parser.add_argument('--decay_step', type=int, default=25, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=1.0, help='Gamma for StepLR')
# epochs #
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
# logger #
parser.add_argument('--log_period_as_iter', type=int, default=12800)
# evaluation #
parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20, 50], help='ks for Metric@k')
parser.add_argument('--best_metric', type=str, default='NDCG@10', help='Metric for determining the best model')

################
# Model
################
parser.add_argument('--model', type=str, default='bert4rec')
parser.add_argument('--sub_model', type=str, default='bert4rec')
parser.add_argument('--model_init_seed', type=int, default=0)
# BERT #
parser.add_argument('--maxlen', type=int, default=10, help='Length of sequence for bert')
parser.add_argument('--num_items', type=int, default=3706, help='Number of total items')
parser.add_argument('--num_users', type=int, default=6040, help='Number of total users')
parser.add_argument('--bert_num_blocks', type=int, default=2, help='Number of transformer layers')
parser.add_argument('--bert_num_heads', type=int, default=4, help='Number of heads for multi-attention')
parser.add_argument('--bert_dropout', type=float, default=0.2, help='Dropout probability to use throughout the model')
parser.add_argument('--bert_mask_prob', type=float, default=0.3, help='Probability for masking items in the training sequence')

################
# Experiment
################
parser.add_argument('--experiment_dir', type=str, default='experiments')
parser.add_argument('--experiment_description', type=str, default='test')

################
# Eraser
################
parser.add_argument('--shards', type=int, default=1, help='Number of shards')
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--epoch_agg', default=10, type=int)
parser.add_argument('--agg_lr', default=0.001, type=float)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--partition', type=str, default='random')
parser.add_argument('--del_way', type=str, default='random')
parser.add_argument('--del_num', default=-1, type=int)
parser.add_argument('--sisa', default=False, type=str)
parser.add_argument('--maxpool', default=False, type=str)
parser.add_argument('--mean', default=True, type=str)
################

parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_heads', default=1, type=int)

parser.add_argument('--shard_num', default=0, type=int)

################

args = parser.parse_args()
