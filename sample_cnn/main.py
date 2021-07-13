from utils import get_accuracy
from torch import nn, optim
from sample_cnn import SimpleCNN
from data import CremaDataset, DataManager, DataType
from tqdm import tqdm
import torch
from setting import BATCH_SIZE, MAX_EPOCH
import torch.nn.functional as F

CREMA = 'CREMA'

class Trainer():
    def __init__(self):
        dataset = CremaDataset()
        self.manager = DataManager()
        self.manager.add_data(CREMA,dataset,BATCH_SIZE)
        self.model = SimpleCNN().cuda()

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(MAX_EPOCH):
            loss_sum = 0
            data = self.manager[CREMA, DataType.TRAIN]
            acc_acc = 0
            acc_len = 0
            for i, (x_train, label, filename) in tqdm(enumerate(data), total=len(data)):
                
                x_train = x_train.cuda()
                label = label.cuda()
                outputs = self.model(x_train)
                loss = criterion(outputs, label)
                _, preds = torch.max(outputs, dim=1)
                acc_acc += torch.tensor(torch.sum(preds == label).item())
                acc_len += len(preds)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_sum += loss.data
            print(f'Acc: {acc_acc/acc_len}')
            print(f'Loss: {loss_sum/(i+1)}')

    def test(self):
        self.model.eval()  # Setting model to test
        test_loss = 0
        dataset = self.manager[CREMA, DataType.TEST]
        with torch.no_grad():
            acc = 0
            for (data, target, filename) in dataset:
                data = data.cuda()
                target = target.cuda()
                output = self.model(data)
                # test_loss = F.cross_entropy(output, target)
                acc += get_accuracy(output, target)
                # print(f"Val loss: {test_loss.detach()}, Val acc : {acc}")
            print(f"Acc: {acc/len(dataset)}")
        self.model.eval()  # Setting model to test
        test_loss = 0
        with torch.no_grad():
            acc = 0
            for (data, target, filename) in dataset:
                data, target = data.cuda(), target.cuda()
                output = self.model(data)
                # test_loss = F.cross_entropy(output, target)
                acc += get_accuracy(output, target)
                # print(f"Val loss: {test_loss.detach()}, Val acc : {acc}")
            print(f"Acc: {acc/len(dataset)}")

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    trainer.test()