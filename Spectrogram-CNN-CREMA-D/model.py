import torch as th
import torch.nn as nn

from torch.nn import Conv2d,MaxPool2d,LSTM

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM,self).__init__()
        self._build()

    def _build(self):
        self.CNN = nn.Sequential(
            Conv2d(1, 16, (12,16), padding="same"),
            nn.ReLU(),
            MaxPool2d(2,2),

            Conv2d(16, 24, (8,12), padding="same"),
            nn.ReLU(),
            MaxPool2d(2,2),

            Conv2d(24, 32, (5,7), padding="same"),
            nn.ReLU(),
            MaxPool2d(2,2)
        )
        self.LSTM = LSTM(input_size=25*32,hidden_size=128, bidirectional=True)
        # First fully connected layer
        self.dense = nn.Linear(37*256, 64)
        self.dropout = nn.Dropout(0.5)
        self.activate = nn.Sequential(
            nn.Linear(64, 4),
            nn.Softmax(dim=0)
        )

    def forward(self,input):
        cnn = self.CNN(input)
        cnn = th.flatten(cnn.transpose(1,2), 1,2).transpose(1,2)
        lstm = self.LSTM(cnn)[0]
        dense = self.dense(th.flatten(lstm,1))
        dropout = self.dropout(dense)
        activated = self.activate(dropout)
        return activated