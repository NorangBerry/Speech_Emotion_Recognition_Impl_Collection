from setting import LABEL
from tqdm.std import tqdm
from setting import BATCH_SIZE
from lib.abstact_dataset import ICremaDataset
import lib.utils as utils


class CremaDataset(ICremaDataset):
    def __init__(self):
        super(CremaDataset,self).__init__(LABEL,BATCH_SIZE)
        self.pre_processing()
        self.load_data()


    def pre_processing(self):
        new_data = []
        for (wave, label, filename, sample_rate) in tqdm(self.dataset):
            #(freq,time)
            spectrogram = utils.stft(wave,sample_rate, window_ms=100, window_type='HAMMING', hop_ms=40, onesided=False)
            spectrogram = spectrogram[:400,:]
            spectrogram = self._padding(spectrogram, time = 300)
            new_data.append((spectrogram,label,filename))
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