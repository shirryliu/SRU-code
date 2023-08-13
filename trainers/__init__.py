from .aggregation import AggTrainer
from .sgb import Trainer


TRAINERS = {
    Trainer.code(): Trainer,
    AggTrainer.code(): AggTrainer,
}


def trainer_factory(args, model, train_loader, val_loader, test_loader, export_root):
    trainer = TRAINERS[args.trainer_code]
    return trainer(args, model, train_loader, val_loader, test_loader, export_root)
