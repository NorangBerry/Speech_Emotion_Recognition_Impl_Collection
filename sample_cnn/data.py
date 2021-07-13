from torch.utils.data.dataset import random_split
from sample_cnn import SimpleCNN
from setting import LABEL, BASE_PATH, BATCH_SIZE
import torch.nn as nn
import torch.nn.functional as F
import torchaudio as ta
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from enum import Enum


class DataType(Enum):
    TRAIN = 1
    TEST = 2

#Data 불러오기
class CremaDataset(Dataset):
    def __init__(self,max_width = 512, max_file_num = -1):
        self.data = []
        self.max_width = max_width

        self.__load_file(max_file_num)
        self.__padding()

    def __load_file(self,max_len=-1):
        directory = os.path.join(BASE_PATH,"CREMA-D")
        for filename in tqdm(os.listdir(directory)[:max_len]):
            ext = os.path.splitext(filename)[-1]
            if ext != '.wav':
                continue

            file_path = os.path.join(directory, filename)
            data, sample_rate = ta.load(file_path)
            data = ta.transforms.MelSpectrogram(sample_rate)(data.view(1, -1))
            label = LABEL[filename.split('_')[-2]]
            self.data.append([data, label, filename])

    def __padding(self):
        for iter in self.data:
            datum, _, _ = iter
            width = len(datum[0][0])
            iter[0] = nn.ZeroPad2d((0,self.max_width - width,0,0,0,0))(datum)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_data = self.data[idx][0]
        label = self.data[idx][1]
        filename = self.data[idx][2]
        return x_data, label, filename

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