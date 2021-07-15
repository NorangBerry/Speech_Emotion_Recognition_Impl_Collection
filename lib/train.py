from .abstact_dataset import DataType
from tqdm import tqdm
import torch
import numpy as np

class Trainer():
    def __init__(self, dataloader, model, max_epoch, labels):
        self.dataloader = dataloader
        self.model = model.cuda()
        self.max_epoch = max_epoch
        self.labels = labels
        
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_loss_function(self, criterion):
        self.criterion = criterion

    def cross_validation(self, fold_size):
        results = [0] * fold_size
        for fold in range(fold_size):
            self.dataloader.set_next_step(fold+1)
            print(f'FOLD {fold+1}')
            print('--------------------------------')
            self.model.apply(self.reset_weights)
            self.model.initialize_weights()
            self.train()
            save_path = f'./model-fold-{fold}.pth'
            torch.save(self.model.state_dict(), save_path)

            
            weight_accuracy, unweight_accuracy = self.test()
            results[fold] = [weight_accuracy, unweight_accuracy]
        results = np.mean(np.array(results), axis=0)
        print(f"Total Weighted Accuracy: {results[0]}")
        print(f"Total UnWeighted Accuracy: {results[1]}")
    def reset_weights(self, m):
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()


    def train(self):
        loss = 1e+10
        early_stopping_rounds = 10
        loss_not_decrease_round = 0
        for epoch in range(self.max_epoch):
            print(f"Traing Epoch: {epoch} ")
            self._train_epoch()
            current_loss = self._validate_epoch()
            if current_loss > loss:
                loss_not_decrease_round += 1
                if loss_not_decrease_round >= early_stopping_rounds:
                    break
            else:
                loss = current_loss
                loss_not_decrease_round = 0

    def _train_epoch(self):
        self.model.train(True)
        dataloader = self.dataloader[DataType.TRAIN]
        loss_sum = 0

        for i, (feed, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
            feed = feed.cuda()  
            label = label.cuda()
            outputs = self.model(feed)
            loss = self.criterion(outputs, label)
            loss_sum += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print(f'Training Loss: {loss_sum/len(dataloader):.6f}')

    def _validate_epoch(self):
        self.model.train(False)
        dataloader = self.dataloader[DataType.VALIDATION]
        self.start_accuracy_calc()
        loss_sum = 0

        for (feed, label) in dataloader:
            feed = feed.cuda()  
            label = label.cuda()

            outputs = self.model(feed)
            loss = self.criterion(outputs, label)
            self.add_accuracy_data(outputs,label)
            loss_sum += loss.item()
        
        loss_mean = loss_sum/len(dataloader)
        self.print_accuracy_data()
        print(f'Validation Loss: {loss_mean:.6f}')
        return loss_mean

    def start_accuracy_calc(self):
        self.acc_sum = [0] * len(self.labels)
        self.acc_len = [0] * len(self.labels)

    def add_accuracy_data(self, result, labels):
        _, predictions = torch.max(result, dim=1)
        for prediction,label in zip(predictions,labels):
            self.acc_sum[label] += 1 if prediction == label else 0
            self.acc_len[label] += 1

    def print_accuracy_data(self):
        print(f'Weight Accuracy: {100*np.sum(self.acc_sum)/np.sum(self.acc_len):2.2f}%')
        unweight_accuracy_sum = np.sum([self.acc_sum[i]/(self.acc_len[i]+1e-8) for i in range(len(self.labels))])
        print(f'Unweight Accuracy: {100*unweight_accuracy_sum/len(self.labels):2.2f}%')
        class_accuracy = ''
        for key, i in self.labels.items():
            class_accuracy += f'[{key}]: {100*self.acc_sum[i]/(self.acc_len[i]+1e-8):2.2f}% '
        print(class_accuracy)

    def get_accuray(self):
        unweight_accuracy_sum = np.sum([self.acc_sum[i]/(self.acc_len[i]+1e-8) for i in range(len(self.labels))])
        unweight_accuracy = 100*unweight_accuracy_sum/len(self.labels)
        weight_accuracy = 100*np.sum(self.acc_sum)/np.sum(self.acc_len)
        return weight_accuracy, unweight_accuracy

    def test(self):
        self.model.eval()  # Setting model to test
        dataloader = self.dataloader[DataType.TEST]
        with torch.no_grad():
            print('\nTest Start!')
            self.start_accuracy_calc()
            for (tester, label) in dataloader:
                tester = tester.cuda()
                label = label.cuda()
                outputs = self.model(tester)
                self.add_accuracy_data(outputs, label)
            self.print_accuracy_data()
            return self.get_accuray()

if __name__ == '__main__':
    pass