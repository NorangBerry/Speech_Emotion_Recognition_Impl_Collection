import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from model import CNN_LSTM
from iemocap import IEMOCAP
from lib.train import Trainer

if __name__ == '__main__':
    trainer = Trainer(IEMOCAP(),CNN_LSTM())
    trainer.train()
    trainer.test()