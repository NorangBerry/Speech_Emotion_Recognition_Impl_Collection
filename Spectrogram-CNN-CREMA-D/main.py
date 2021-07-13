import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from crema import CremaDataset
from torch import nn

from setting import LABEL, MAX_EPOCH
from torch import nn, optim

from model import CNN_LSTM
from iemocap import IEMOCAP
from lib.train import Trainer

if __name__ == '__main__':
    model = CNN_LSTM()

    trainer = Trainer(CremaDataset(),model, max_epoch = MAX_EPOCH, labels = LABEL)
    trainer.set_optimizer(optim.Adam(model.parameters(), lr=1e-3))
    trainer.set_loss_function(nn.CrossEntropyLoss())

    trainer.train()
    trainer.test()