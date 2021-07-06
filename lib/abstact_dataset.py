from enum import Enum
from lib.utils import change_sample_rate
import os
from .const import BASE_PATH, LABEL
from tqdm import tqdm
from torch import torch
from torch.utils.data import Dataset
import re
import torchaudio as ta
from torch import nn

class DataType(Enum):
    TRAIN = 1
    TEST = 2
    VALIDATION = 3

#Data 불러오기
class ICremaDataset(Dataset):
    def __init__(self,sample_rate=None):
        super(ICremaDataset,self).__init__()
        self.sample_rate = sample_rate
        self.dataset = []
        self.dataloader = {}
        self.__load_file()

    def __load_file(self):
        directory = os.path.join(BASE_PATH,"CREMA-D")
        for filename in tqdm(os.listdir(directory)):
            ext = os.path.splitext(filename)[-1]
            if ext != '.wav':
                continue
            key = filename.split('_')[-2]
            if key not in LABEL.keys():
                continue
            label = LABEL[key]

            file_path = os.path.join(directory, filename)
            data, sample_rate = ta.load(file_path)
            if self.sample_rate and self.sample_rate != sample_rate:
                data, sample_rate = change_sample_rate(data,sample_rate,self.sample_rate)
            self.dataset.append((data, label, sample_rate))

    def pre_processing(self):
        pass

    def load_data(self):
        pass
    
    def __get_window(self, window_type, window_size):
        if window_type == 'HAMMING':
            return torch.hamming_window(window_size)
        else:
            return None
    
    def _padding(self,spectrogram, time = 0):
        width = len(spectrogram)
        spectrogram = nn.ZeroPad2d((0,0,0,0,time - width,0))(spectrogram)
        return spectrogram

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,key):
        return self.dataloader[key]

#Data 불러오기
class IIEMOCAP(Dataset):
    def __init__(self):
        super(IIEMOCAP,self).__init__()
        self.dataset = []
        self.dataloader = {}
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
            if label in LABEL:
                labels.append((file_name, LABEL[label]))
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

    def pre_processing(self):
        pass

    def feed_data(self):
        pass
    
    def _stft(self, data, sample_rate, window_ms, window_type=None, hop_ms=None):
        window_size = (sample_rate * window_ms)//100
        hop_length = (sample_rate * hop_ms)//1000
        window = self.__get_window(window_type, window_size) 
        stft = torch.stft(data, n_fft=window_size, hop_length=hop_length, window=window)
        #(frequency, time)
        stft = stft.permute(0,3,1,2)[0][0]
        return stft

    def __get_window(self, window_type, window_size):
        if window_type == 'HAMMING':
            return torch.hamming_window(window_size)
        else:
            return None
    
    def _padding(self,spectrogram, time = 0, freq = 0):
        width = len(spectrogram[0])
        spectrogram = nn.ZeroPad2d((0,time - width,0,freq))(spectrogram)
        return spectrogram.unsqueeze(0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,key):
        return self.dataloader[key]
