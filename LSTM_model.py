import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import os.path as op
import sys
import time


class CurvesDataset(Dataset):
    def __init__(self, dir_csv, hlen=2, prelen=1):
        super(CurvesDataset, self).__init__()
        self.csvdata = pd.read_csv(dir_csv).values
        self.csvdata = self.csvdata.astype('float32')
        # max_value = np.max(self.csvdata, axis=0)
        # min_value = np.min(self.csvdata, axis=0)
        self.scalar = np.max(self.csvdata)-np.min(self.csvdata)
        self.csvdata = self.csvdata/self.scalar
        self.csv_size = self.csvdata.shape
        self.h_len = hlen
        self.pre_len = prelen

    def __len__(self):
        return (self.csv_size[0]-(self.h_len+self.pre_len)+1)*self.csv_size[1]

    def __getitem__(self, idx):
        index_column = idx//(self.csv_size[0]-(self.h_len+self.pre_len)+1)
        index_row = idx % (self.csv_size[0]-(self.h_len+self.pre_len)+1)
        history = self.csvdata[index_row:(index_row+self.h_len), index_column]
        prediction = self.csvdata[(
            index_row+self.h_len):(index_row+self.h_len+self.pre_len), index_column]

        history = torch.from_numpy(history).unsqueeze(0)
        prediction = torch.from_numpy(prediction).unsqueeze(0)
        return Variable(history), Variable(prediction), self.scalar


class lstm(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, output_size=1, num_layer=2):
        super(lstm, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s*b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)
        return x
