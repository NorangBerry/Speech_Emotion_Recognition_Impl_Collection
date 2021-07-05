from .utils import get_accuracy
from torch import nn, optim
from model import CNN_LSTM

from .abstact_dataset import DataType
from tqdm import tqdm
import torch
import numpy as np
from .const import MAX_EPOCH, LABEL

class Trainer():
    def __init__(self, dataloader, model):
        self.dataloader = dataloader
        self.model = model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        dataloader = self.dataloader[DataType.TRAIN]
        self.model.train(True)
        for epoch in range(MAX_EPOCH):
            print(f'\n{epoch+1} Train proceeding')

            self.start_accuracy_calc()
            loss_sum = 0

            for i, (feed, label, filename) in tqdm(enumerate(dataloader), total=len(dataloader)):
                feed = feed.cuda()  
                label = label.cuda()

                outputs = self.model(feed)
                loss = self.criterion(outputs, label)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                loss_sum += loss.data

                self.add_accuracy_data(outputs, label)
            self.print_accuracy_data()
            print(f'Loss: {loss_sum/(i+1)}')
            
            if epoch%100 == 0:
                self.test()
                self.model.train(True)

    def start_accuracy_calc(self):
        self.acc_sum = [0 for i in range(len(LABEL))]
        self.acc_len = [0 for i in range(len(LABEL))]

    def add_accuracy_data(self, result, labels):
        _, predictions = torch.max(result, dim=1)
        for prediction,label in zip(predictions,labels):
            self.acc_sum[label] += 1 if prediction == label else 0
            self.acc_len[label] += 1
        # self.acc_sum += torch.tensor(torch.sum(preds == label).item())
        # self.acc_len += len(preds)

    def print_accuracy_data(self):
        print(f'Weight Accuracy: {np.sum(self.acc_sum)/np.sum(self.acc_len)}')
        unweight_accuracy_sum = np.sum([self.acc_sum[i]/self.acc_len[i] for i in range(len(LABEL))])
        print(f'Unweight Accuracy: {unweight_accuracy_sum/len(LABEL)}')
        class_accuracy = ''
        for key, i in LABEL.items():
            class_accuracy += f'[{key}]: {self.acc_sum[i]/self.acc_len[i]} '
        print(class_accuracy)

    def test(self):
        self.model.eval()  # Setting model to test
        dataloader = self.dataloader[DataType.TEST]
        with torch.no_grad():
            print('\nTest Start!')
            self.start_accuracy_calc()
            for (tester, label, filename) in dataloader:
                tester = tester.cuda()
                label = label.cuda()
                outputs = self.model(tester)
                self.add_accuracy_data(outputs, label)
            self.print_accuracy_data()

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    trainer.test()