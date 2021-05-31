import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm


class CurvesDataset(Dataset):
    def __init__(self, dir_csv, hlen=2, prelen=1):
        super(CurvesDataset, self).__init__()
        self.csvdata = pd.read_csv(dir_csv).values

        self.csvdata = self.csvdata.astype('float32')

        scalar = np.max(self.csvdata)-np.min(self.csvdata)
        print('scalar:', scalar)
        self.csvdata = self.csvdata/scalar
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

        return Variable(history), Variable(prediction)


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


input_size, output_size = 40, 40
cData = CurvesDataset('expCamera.csv', input_size, output_size)
data_iter = DataLoader(cData, batch_size=10, shuffle=True)

print(cData.__len__())


# create lstm
lstm_layers = 2
model = lstm(input_size, input_size*2, output_size, lstm_layers)

# loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=50, gamma=0.5)

# train
epochs = 200
for e in range(epochs):
    total_loss, cnt = 0, 0
    for var_x, var_y in tqdm(data_iter):

        out = model(var_x)
        loss = criterion(out, var_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        cnt += out.shape[0]
    scheduler.step()
    if e % 10 == 0:
        torch.save(model.state_dict(), 'pth_lib/lstm_exp' +
                   str(e).zfill(3)+'.pth')
    print('Epoch:' + str(e+1)+'/'+str(epochs) +
          ', Loss: ' + str(total_loss/cnt))
    # print(cnt)
