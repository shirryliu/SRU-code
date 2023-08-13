from abc import *
import random


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        dataset = dataset.load_dataset()
        [user_train, user_valid, user_test, num_users, num_items] = dataset
        self.train = user_train
        self.val = user_valid
        self.test = user_test
        self.user_count = num_users
        self.item_count = num_items

        args.num_items = num_items
        args.num_users = num_users

        print(num_users, num_items)

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass
