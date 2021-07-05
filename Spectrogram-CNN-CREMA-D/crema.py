from const import BATCH_SIZE
from torch.utils.data import DataLoader
from abstact_dataset import ICremaDataset,DataType


class CremaDataset(ICremaDataset):
    def __init__(self):
        super(CremaDataset,self).__init__()
        self.pre_processing()
        self.feed_data()


    def pre_processing(self):
        new_data = []
        for (wave, label, filename, sample_rate) in self.dataset:
            #(freq,time)
            spectrogram = self._stft(wave,sample_rate, window_ms=100, window_type='HAMMING', hop_ms=40)
            spectrogram = spectrogram[:200,:]
            spectrogram = self._padding(spectrogram, time = 300)
            new_data.append((spectrogram,label,filename))
        self.dataset = new_data
    
    def feed_data(self):
        speakers = []

        for (spectrogram, label, filename) in self.dataset:
            speaker = filename.split('_')[0]
            if speaker not in speakers:
                speakers.append(speaker)

        test_speakers = self.get_test_speaker(speakers)
        
        train_data = []
        test_data = []
        for iter in self.dataset:
            speaker = iter[2].split('_')[0]
            if speaker in test_speakers:
                test_data.append(iter)
            else:
                train_data.append(iter)

        train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size = BATCH_SIZE)
        
        self.dataloader[DataType.TRAIN] = train_loader
        self.dataloader[DataType.TEST] = test_loader
        return train_data, test_data

    def get_test_speaker(self,speakers):
        test_size = int(0.1*len(speakers))
        return speakers[test_size:test_size*2]