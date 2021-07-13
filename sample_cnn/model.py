import torch as th
import torch.nn as nn

from torch.nn import Conv2d,MaxPool2d

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()
        self._build()

    def _build(self):
        self.conv1a = Conv2d(in_channels=1,out_channels=8, kernel_size=(10,2), padding="same")
        self.conv1b = Conv2d(1, 8, (2,8), padding="same")
        self.conv_serise = nn.Sequential(
            Conv2d(16, 32, 3, padding="same"),
            MaxPool2d(2,2),
            Conv2d(32, 48, 3, padding="same"),
            MaxPool2d(2,2),
            Conv2d(48, 64, 3, padding="same"),
            Conv2d(64, 80, 3, padding="same")
        )
        
        # First fully connected layer
        self.fc1 = nn.Linear(256000, 6)

    def forward(self,input):
        layer1 = th.cat((self.conv1a(input),self.conv1b(input)),1)
        layer2 = self.conv_serise(layer1)
        layer2 = th.flatten(layer2, 1)
        layer_final = self.fc1(layer2)
        return layer_final