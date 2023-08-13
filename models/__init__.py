from models.aggregation import Aggregation
from models.bert4rec import BERT4Rec
from models.gru4rec import GRU4Rec
from models.sasrec import SASRec

MODELS = {
    BERT4Rec.code(): BERT4Rec,
    GRU4Rec.code(): GRU4Rec,
    SASRec.code(): SASRec,
    Aggregation.code(): Aggregation,
}


def model_factory(args):
    model = MODELS[args.model]
    return model(args).to(args.device)
