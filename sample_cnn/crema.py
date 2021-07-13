from torch.utils.data.dataset import random_split
from setting import LABEL, BATCH_SIZE
from lib.const import BASE_PATH
import torch.nn as nn
import torchaudio as ta
import os
from torch.utils.data import DataLoader
from lib.abstact_dataset import ICremaDataset
from tqdm import tqdm


#Data 불러오기
class CremaDataset(ICremaDataset):
    def __init__(self):
        super(CremaDataset,self).__init__(LABEL,BATCH_SIZE)
        self.pre_processing()
        self.load_data()

    def pre_processing(self):
        new_data = []
        for (wave, label, filename, sample_rate) in tqdm(self.dataset):
            #(freq,time)
            spectrogram = ta.transforms.MelSpectrogram(sample_rate)(wave.view(1, -1))
            new_data.append((spectrogram,label,filename))

        for (spectrogram, label, filename) in self.dataset:
            self.__padding(spectrogram)
        self.dataset = new_data


    def split_dataset(self):
        speakers = []
        for (spectrogram, label, filename) in self.dataset:
            speaker = filename.split('_')[0]
            if speaker not in speakers:
                speakers.append(speaker)
        
        test_len = int(len(speakers)*0.1)
        test_speakers, train_speakers = speakers[:test_len], speakers[test_len:]

        val_len = int(len(train_speakers)*0.2)
        val_speakers, train_speakers = train_speakers[:val_len], train_speakers[val_len:]

        train_data = []
        val_data = []
        test_data = []
        for (spectrogram, label, filename) in self.dataset:
            speaker = filename.split('_')[0]
            if speaker in test_speakers:
                test_data.append((spectrogram, label))
            elif speaker in val_speakers:
                val_data.append((spectrogram, label))
            else:
                train_data.append((spectrogram, label))

        return train_data, val_data, test_data


    def __padding(self):
        for iter in self.data:
            datum, _, _ = iter
            width = len(datum[0][0])
            iter[0] = nn.ZeroPad2d((0,self.max_width - width,0,0,0,0))(datum)

class DataManager():
    def __init__(self):
        self.data = {}
    def add_data(self,name,dataset,shuffle=True,num_workers=0):
        val_size = int(0.1*len(dataset))
        train_size = len(dataset) - val_size

        train_data, test_data = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle=shuffle, num_workers=num_workers)
        test_loader = DataLoader(test_data)
        self.data[name] = {}
        self.data[name][DataType.TRAIN] = train_loader
        self.data[name][DataType.TEST] = test_loader
    def __getitem__(self, key):
        name, type = key
        return self.data[name][type]