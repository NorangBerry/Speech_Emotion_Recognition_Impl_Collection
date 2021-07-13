import numpy as np
from torch.functional import Tensor
from torch.utils.data.dataset import random_split
from lib.utils import do_mfcc
from tqdm.std import tqdm
from torch.utils.data import DataLoader
from lib.abstact_dataset import ICremaDataset,DataType
from setting import BATCH_SIZE
from sklearn.model_selection import train_test_split


class CremaDataset(ICremaDataset):
    def __init__(self, sample_rate = None):
        super(CremaDataset,self).__init__(sample_rate)
        self.pre_processing()
        self.load_data()

    def pre_processing(self):
        new_data = []
        max_size = 0
        for (wave, label, sample_rate) in tqdm(self.dataset):
            #(freq,time)
            # spectrogram = self._stft(wave,sample_rate, window_ms=100, window_type='HAMMING', hop_ms=40)
            mfcc = do_mfcc(wave,sample_rate)
            max_size = max(max_size,mfcc.shape[0])
            new_data.append((mfcc,label))
        
        final_data = []
        for iter in new_data:
            mfcc,label = iter
            mfcc = self._padding(mfcc,max_size)
            final_data.append((mfcc.squeeze(2),label))

        self.dataset = final_data
    
    def load_data(self):
        # test_size = round(0.1*len(self.dataset))
        # train_data, test_data = random_split(self.dataset, [len(self.dataset) - test_size, test_size])

        # val_size = round(0.2*len(train_data))
        # train_data, val_data = random_split(train_data, [len(train_data) - val_size, val_size])
        data, label = self.split_data_label(self.dataset)
        train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.1)
        train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.2)

        train_data = self.merge_data_label(train_data,train_label)
        test_data = self.merge_data_label(test_data,test_label)
        val_data = self.merge_data_label(val_data,val_label)


        train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data)
        test_loader = DataLoader(test_data)

        self.dataloader = {}
        self.dataloader[DataType.TRAIN] = train_loader
        self.dataloader[DataType.TEST] = test_loader
        self.dataloader[DataType.VALIDATION] = val_loader

    def split_data_label(self,dataset):
        ret_data = []
        ret_label = []
        for (data,label) in dataset:
            ret_data.append(data)
            ret_label.append(label)
        return ret_data,ret_label

    def merge_data_label(self,data,label):
        return [iter for iter in  zip(data,label)]

    def get_test_speaker(self,speakers):
        test_size = int(0.1*len(speakers))
        return speakers[test_size:test_size*2]