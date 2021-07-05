from enum import Enum
import os

import librosa
from . import const
from .const import BASE_PATH, LABEL
from tqdm import tqdm
from torch import torch
from torch.utils.data import Dataset
import re
import torchaudio as ta
import numpy as np
import matplotlib.pyplot  as plt
from torch import nn
import librosa.display

class DataType(Enum):
    TRAIN = 1
    TEST = 2

#Data 불러오기
class ICremaDataset(Dataset):
    def __init__(self):
        super(ICremaDataset,self).__init__()
        self.dataset = []
        self.dataloader = {}
        self.__load_file()
        self.__show_sample('FEA')

    def __load_file(self):
        directory = os.path.join(BASE_PATH,"CREMA-D")
        for filename in tqdm(os.listdir(directory)):
            ext = os.path.splitext(filename)[-1]
            if ext != '.wav':
                continue
            key = filename.split('_')[-2]
            if key not in const.LABEL.keys():
                continue
            label = const.LABEL[key]

            file_path = os.path.join(directory, filename)
            data, sample_rate = ta.load(file_path)
            self.dataset.append((data, label, filename, sample_rate))
            # self.__show_spectrogram(stft)

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
    
    def _mfcc(self, data, sample_rate):
        mfcc = librosa.feature.mfcc(np.array(data), sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
        mfcc = mfcc.T
        return mfcc
    
    def __get_window(self, window_type, window_size):
        if window_type == 'HAMMING':
            return torch.hamming_window(window_size)
        else:
            return None

    def __show_sample(self,emotion):
        for (data, label, filename, sample_rate) in self.dataset:
            if label == const.LABEL[emotion]:
                self.__show_waveplot(data[0],emotion,sample_rate)
                self.__show_spectrogram(data[0],emotion,sample_rate)
                break
    
    def __show_waveplot(self,data,emotion,sample_rate):
        plt.figure(figsize=(10, 3))
        plt.title('Waveplot for {} emotion'.format(emotion), size=15)
        librosa.display.waveplot(np.array(data), sr=sample_rate)
        plt.show(block=False)

    def __show_spectrogram(self,data,emotion,sample_rate):
        X = librosa.stft(np.array(data))
        Xdb = librosa.amplitude_to_db(np.abs(X))
        plt.figure(figsize=(12, 3))
        plt.title('Spectrogram for {} emotion'.format(emotion), size=15)
        # plt.xlabel("Time")
        # plt.ylabel("Hz")
        librosa.display.specshow(Xdb, sr=sample_rate, x_axis='time', y_axis='hz')  
        plt.colorbar()
        plt.title("Spectrogram (dB)")
        plt.show()
    
    def __show__emotion_distribution(self):
        labels = [0 for i in range(len(const.LABEL))]
        for (data, label, filename, sample_rate) in self.dataset:
            labels[label] += 1
        plt.title('Count of Emotions', size=16)
        x = np.arange(len(labels))
        plt.xticks(x,const.LABEL.keys())
        plt.bar(x,labels)
        plt.ylabel('Count', size=12)
        plt.xlabel('Emotions', size=12)
        plt.show()
    
    
    def _padding(self,spectrogram, time = 0, freq = 0):
        width = len(spectrogram[0])
        spectrogram = nn.ZeroPad2d((0,time - width,0,freq))(spectrogram)
        return spectrogram.unsqueeze(0)

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
