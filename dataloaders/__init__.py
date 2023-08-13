from dataloaders.bert4rec import BERTDataloader
from dataloaders.gru4rec import GRUDataloader
from dataloaders.load_dataset import MyDataset
from dataloaders.sasrec import SASDataloader

DATALOADERS = {
    BERTDataloader.code(): BERTDataloader,
    SASDataloader.code(): SASDataloader,
    GRUDataloader.code(): GRUDataloader,
}


def dataloader_factory(args):
    dataset = MyDataset(args)
    dataloader = DATALOADERS[args.sub_model]
    dataloader = dataloader(args, dataset)
    train, val, test, agg, tradition, num_users, num_items = dataloader.get_pytorch_dataloaders()
    return train, val, test, agg, tradition, num_users, num_items
