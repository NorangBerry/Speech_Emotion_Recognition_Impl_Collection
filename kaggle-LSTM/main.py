import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from lib.train import Trainer
from crema import CremaDataset
from model import StackLSTM

import os
import sys

if __name__ == '__main__':
    trainer = Trainer(CremaDataset(22050),StackLSTM(13))
    trainer.train()
    trainer.test()