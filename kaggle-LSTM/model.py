import torch.nn as nn

from torch.nn import LSTM

class StackLSTM(nn.Module):
    def __init__(self,input_shape):
        super(StackLSTM,self).__init__()
        self._build(input_shape)

    def _build(self,input_shape):
        self.lstm1 = LSTM(input_size=input_shape,hidden_size=128,batch_first=True)
        self.lstm2 = LSTM(input_size=128,hidden_size=64,batch_first=True)
        self.activation = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 6)
        )

    def forward(self,input):
        output = self.lstm1(input)[0]
        output = self.lstm2(output)[1][0]
        output = self.activation(output)
        return output.squeeze(0)