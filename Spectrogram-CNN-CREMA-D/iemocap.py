import torch
from torch.utils.data.dataset import random_split
from setting import BATCH_SIZE
from lib.abstact_dataset import IIEMOCAP, DataType
from torch.utils.data import DataLoader
from setting import LABEL
from tqdm import tqdm
import lib.utils as utils

class IEMOCAP(IIEMOCAP):
    def __init__(self):
        super(IEMOCAP,self).__init__(LABEL,BATCH_SIZE)
        self.pre_processing()
        self.load_data()

    def pre_processing(self):
        new_data = []
        print("\nPre_processiong")
        # self.check_label_num()
        for (wave, label, filename, sample_rate) in tqdm(self.dataset):
            waves = self.split_wave(wave,sample_rate,time = 3000)

            for partial_wave in waves:
                #(freq,time)
                window_ms = 100
                spectrogram = utils.stft(partial_wave,sample_rate, window_ms=window_ms, window_type='HAMMING', hop_ms=40, onesided=False)
                spectrogram = self.split_spectrogram(spectrogram, sample_rate, window_ms, frequency=4000)[0]
                spectrogram = self._padding(spectrogram, time = 300)
                new_data.append((spectrogram,label,filename))

        self.dataset = new_data

    def split_wave(self,wave,sample_rate,time):
        wave_len = wave.shape[1]
        limit_len = sample_rate * time

        waves = []
        if wave_len > limit_len:
            part_num = (wave_len//limit_len)+1
            wave_part_len = wave_len//part_num
            parts = torch.split(wave,wave_part_len,1)
            waves = [part for part in parts]
            if waves[-1].shape[1] <= part_num:
                waves[-2] = torch.cat((waves[-2], waves[-1]), dim=1)
                waves.pop()
        else:
            waves = [wave]
        return waves

    def split_spectrogram(self,spectrogram, sample_rate, window_ms, frequency = 4000):
        freq_resolution = (sample_rate*window_ms)//1000
        freq_unit = sample_rate//freq_resolution
        freq_len = frequency//freq_unit
        return [spectrogram[:freq_len,:], spectrogram[freq_len:,:]]

    def check_label_num(self):
        ret = [0] * len(LABEL)
        for (wave, label, filename, sample_rate) in tqdm(self.dataset):
            ret[label] += 1
        for key in LABEL:
            print(key,ret[LABEL[key]])

    def split_dataset(self):
        train_data = []
        val_data = []
        test_data = []
        for (spectrogram,label,filename) in self.dataset:
            session_num = int(filename[4])
            if session_num < 5:
                train_data.append((spectrogram,label))
            else:
                if filename[5] == 'F':
                    val_data.append((spectrogram,label))
                else:
                    test_data.append((spectrogram,label))
        return train_data, val_data, test_data

    def get_test_speaker(self,speakers):
        test_size = int(0.1*len(speakers))
        return speakers[test_size:test_size*2]