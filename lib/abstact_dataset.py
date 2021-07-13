from abc import abstractmethod
from enum import Enum
from lib.utils import change_sample_rate
import os
from .const import BASE_PATH
from tqdm import tqdm
from torch import torch
from torch.utils.data import Dataset, DataLoader
import re
import torchaudio as ta
from torch import nn

class DataType(Enum):
    TRAIN = 1
    TEST = 2
    VALIDATION = 3

class WrapDataset(Dataset):
    def __init__(self,labels,batch_size,sample_rate=None):
        self.sample_rate = sample_rate
        self.labels = labels
        self.dataset = []
        self.dataloader = {}
        self.batch_size = batch_size

    @abstractmethod
    def __load_file(self):
        pass
    @abstractmethod
    def pre_processing(self):
        pass

    @abstractmethod
    def split_dataset(self):
        pass

    def load_data(self):
        train_data, val_data, test_data = self.split_dataset()
        self.dataloader = {
            DataType.TRAIN : DataLoader(train_data, batch_size = self.batch_size, shuffle=True),
            DataType.TEST : DataLoader(test_data),
            DataType.VALIDATION : DataLoader(val_data)
        }
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,key):
        return self.dataloader[key]

#Data 불러오기
class ICremaDataset(WrapDataset):
    def __init__(self,labels,batch_size,sample_rate=None):
        super(ICremaDataset,self).__init__(labels,batch_size,sample_rate)
        self.__load_file()

    def __load_file(self):
        directory = os.path.join(BASE_PATH,"CREMA-D")
        for filename in tqdm(os.listdir(directory)):
            ext = os.path.splitext(filename)[-1]
            if ext != '.wav':
                continue
            key = filename.split('_')[-2]
            if key not in self.labels.keys():
                continue
            label = self.labels[key]

            file_path = os.path.join(directory, filename)
            data, sample_rate = ta.load(file_path)
            if self.sample_rate and self.sample_rate != sample_rate:
                data, sample_rate = change_sample_rate(data,sample_rate,self.sample_rate)
            self.dataset.append((data, label,filename, sample_rate))

    def _padding(self,spectrogram, time = 0):
        width = len(spectrogram[0])
        spectrogram = nn.ZeroPad2d((0,time - width,0,0))(spectrogram)
        return spectrogram.unsqueeze(0)

#Data 불러오기
class IIEMOCAP(WrapDataset):
    def __init__(self, labels, batch_size):
        super(IIEMOCAP,self).__init__(labels,batch_size)
        self.__load_file()

    def __load_file(self):
        directory = os.path.join(BASE_PATH,"IEMOCAP")
        print("\nSession data loading")
        for i in tqdm(range(5)):
            sesssion_dir = os.path.join(directory,f"Session{i+1}")
            wav_dir = os.path.join(sesssion_dir,"sentences\\wav")
            label_dir = os.path.join(sesssion_dir,"dialog\\EmoEvaluation")
            for dialog_dir in os.listdir(wav_dir):
                if not os.path.isdir(os.path.join(wav_dir,dialog_dir)):
                    continue
                labels = self.__parse_label_file(label_dir,dialog_dir)
                for file_name, label in labels:
                    file_path = os.path.join(os.path.join(wav_dir,dialog_dir),f"{file_name}.wav")
                    data, sample_rate = ta.load(file_path)
                    self.dataset.append((data, label, file_name, sample_rate))

    def __parse_label_file(self,label_dir,dialog_dir):
        label_file = open(os.path.join(label_dir,f"{dialog_dir}.txt"), "rb")
        labels = []
        for line in label_file.readlines():
            line = line.decode("utf-8")
            if not self.__is_label_txt(line):
                continue
            start_time, end_time, file_name, label = self.__parse_label_txt(line)
            if label in self.labels:
                labels.append((file_name, self.labels[label]))
        return labels

    def __is_label_txt(self,line):
        time = "\[[0-9]+.[0-9]+ - [0-9]+.[0-9]+\]"
        folder = "Ses[a-zA-Z0-9_.-]+"
        label = "[a-z]{3}"
        pattern = re.compile('\s+'.join([time,folder,label]))
        return re.match(pattern,line)

    def __parse_label_txt(self,line):
        time,file_name,label,_ = line.split('\t')
        start_time, end_time =  time[1:-1].split(' - ')
        label = label.upper()
        return start_time, end_time, file_name, label
    
    def _padding(self,spectrogram, time = 0, freq = 0):
        width = len(spectrogram[0])
        spectrogram = nn.ZeroPad2d((0,time - width,0,freq))(spectrogram)
        return spectrogram.unsqueeze(0)