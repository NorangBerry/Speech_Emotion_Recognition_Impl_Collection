from torch.utils.data.dataset import random_split
from lib.const import BATCH_SIZE
from lib.abstact_dataset import IIEMOCAP, DataType
from torch.utils.data import DataLoader
from lib.const import LABEL
from tqdm import tqdm

class IEMOCAP(IIEMOCAP):
    def __init__(self):
        super(IEMOCAP,self).__init__()
        self.pre_processing()
        self.feed_data()

    def pre_processing(self):
        new_data = []
        print("\nPre_processiong")
        self.check_label_num()
        for (wave, label, filename, sample_rate) in tqdm(self.dataset):
            #(freq,time)
            spectrogram = self._stft(wave,sample_rate, window_ms=100, window_type='HAMMING', hop_ms=40)
            spectrogram = spectrogram[:200,:]
            spectrogram = self._padding(spectrogram, time = 300)
            new_data.append((spectrogram,label,filename))
        self.dataset = new_data

    def check_label_num(self):
        ret = [0 for i in range(len(LABEL))]
        for (wave, label, filename, sample_rate) in tqdm(self.dataset):
            ret[label] += 1
        for key in LABEL:
            print(key,ret[LABEL[key]])


    def feed_data(self):
        train_len = int(len(self.dataset)*0.9)
        train_data, test_data = random_split(self.dataset,[train_len, len(self.dataset)-train_len])

        train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size = BATCH_SIZE)
        
        self.dataloader[DataType.TRAIN] = train_loader
        self.dataloader[DataType.TEST] = test_loader
        return train_data, test_data

    def get_test_speaker(self,speakers):
        test_size = int(0.1*len(speakers))
        return speakers[test_size:test_size*2]