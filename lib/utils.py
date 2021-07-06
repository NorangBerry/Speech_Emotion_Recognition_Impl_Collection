import torchaudio
from lib.const import LABEL
import torch
import matplotlib.pyplot  as plt
import librosa.display
import librosa
import numpy as np

def get_accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)

def get_param_num(nnet, Mb=True):
    """
    Return number parameters(not bytes) in nnet
    """
    neles = sum([param.nelement() for param in nnet.parameters()])
    return neles / 10**6 if Mb else neles

def show_spectrogram(data,emotion,sample_rate):
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

def show_waveplot(data,emotion,sample_rate):
    plt.figure(figsize=(10, 3))
    plt.title('Waveplot for {} emotion'.format(emotion), size=15)
    librosa.display.waveplot(np.array(data), sr=sample_rate)
    plt.show(block=False)

def show_emotion_distribution(label_data):
    labels = [0 for i in range(len(LABEL))]
    for label in label_data:
        labels[label] += 1
    plt.title('Count of Emotions', size=16)
    x = np.arange(len(labels))
    plt.xticks(x,LABEL.keys())
    plt.bar(x,labels)
    plt.ylabel('Count', size=12)
    plt.xlabel('Emotions', size=12)
    plt.show()


def do_stft(data, sample_rate, window_ms, window_type=None, hop_ms=None):
    window_size = (sample_rate * window_ms)//100
    hop_length = (sample_rate * hop_ms)//1000
    window = self.__get_window(window_type, window_size) 
    stft = torch.stft(data, n_fft=window_size, hop_length=hop_length, window=window)
    #(frequency, time)
    stft = stft.permute(0,3,1,2)[0][0]
    return stft

def do_mfcc(data, sample_rate):
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate, n_mfcc=13,
        melkwargs={
        'n_fft': 2048,
        'hop_length': 512,
    })
    mfcc = mfcc_transform(data).T
    return mfcc

def change_sample_rate(wav: torch.tensor, input_rate, output_rate) -> torch.tensor:
    # Only accepts 1-channel waveform input
    wav = wav.view(wav.size(0), -1)
    new_length = wav.size(-1) * output_rate // input_rate
    indices = (torch.arange(new_length) * (input_rate / output_rate))
    round_down = wav[:, indices.long()]
    round_up = wav[:, (indices.long() + 1).clamp(max=wav.size(-1) - 1)]
    output = round_down * (1. - indices.fmod(1.)).unsqueeze(0) + round_up * indices.fmod(1.).unsqueeze(0)
    return output, output_rate
