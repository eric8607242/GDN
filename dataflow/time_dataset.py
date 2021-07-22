import numpy as np

import torch

from torch.utils.data import Dataset

class TimeDataset(Dataset):
    def __init__(self, raw_data, slide_win, slide_stride=1):
        self.raw_data = raw_data.astype(np.float32)

        self.slide_win = slide_win
        self.slide_stride = slide_stride

        data, labels = raw_data[:, :-1], raw_data[:, -1]

        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        self.x, self.y, self.labels = self._process_data(data, labels)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        feature = self.x[idx]
        y = self.y[idx]

        label = self.labels[idx]

        return feature, y, label


    def _process_data(self, data, labels):
        x, y, attack_label = [], [], []

        total_time_len, node_num = data.shape

        index_arange = range(self.slide_win, total_time_len, self.slide_stride)

        for i in index_arange:
            feature = data[i-self.slide_win:i]
            target = data[i]

            x.append(feature.T)
            y.append(target.T)

            attack_label.append(labels[i])

        x = torch.stack(x)
        y = torch.stack(y)
        labels = torch.Tensor(attack_label).long()
        
        return x, y, labels




        

        
        

        
