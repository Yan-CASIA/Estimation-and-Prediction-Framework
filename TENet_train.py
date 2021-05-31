import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from matplotlib import pyplot as plt
# import skvideo.io
from TENet_model import *
import numpy as np
import glob
import sys
import os.path as op
from tqdm import tqdm
# import pytorch_ssim


imglist = glob.glob(op.join(sys.path[0], 'ground/*.jpg'))
#biasdir = 'ground/fsbias.npy'
eval_imglist = glob.glob(op.join(sys.path[0], 'water/*6.jpg'))
#eval_biasdir = 'water/fsbias.npy'

batch_size = 10
# iData = NeighborDataset('fullimu.mp4')
iData = efsImgsDataset(imglist)
# print(iData.__len__)
data_iter = DataLoader(iData, batch_size=batch_size,
                       shuffle=True)

print('train dataset load cmp.')
eval_batch_size = 10
# iData = NeighborDataset('fullimu.mp4')
eval_iData = efsImgsDataset(eval_imglist)
# print(iData.__len__)
eval_data_iter = DataLoader(
    eval_iData, batch_size=eval_batch_size, shuffle=False)

print('test dataset load cmp.')
model = STN()
print(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# model = torch.nn.DataParallel(model, device_ids=[0, 3])
print('device on ', device)
# model.load_state_dict(torch.load(
#     op.join(sys.path[0], 'mfs499.pth'), map_location=torch.device('cuda')))

criterion = nn.MSELoss()
# criterion = pytorch_ssim.SSIM()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=20, gamma=0.8)

epochs = 500
loss_list = []
for e in range(epochs):

    model.train()
    total_loss, cnt = 0, 0
    for imgcat, prev, curr in tqdm(data_iter):

        if torch.cuda.is_available():
            imgcat = imgcat.cuda()
            prev = prev.cuda()
            curr = curr.cuda()
        out = model(imgcat)

        grid = F.affine_grid(out, prev.size())
        output = F.grid_sample(curr, grid)

        o1 = output[:, :, 200:568, 200:824]

        p1 = prev[:, :, 200:568, 200:824]

        loss = criterion(o1, p1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        cnt += out.shape[0]

    torch.save(model.state_dict(), 'pth_lib/new-'+str(e).zfill(3)+'.pth')
    model.eval()
    eval_total_loss, eval_cnt = 0, 0

    with torch.no_grad():
        for eval_imgcat, eval_prev, eval_curr in tqdm(eval_data_iter):
            if torch.cuda.is_available():
                eval_imgcat = eval_imgcat.cuda()
                eval_prev = eval_prev.cuda()
                eval_curr = eval_curr.cuda()
            eval_out = model(eval_imgcat)
            eval_grid = F.affine_grid(eval_out, eval_prev.size())
            eval_output = F.grid_sample(eval_curr, eval_grid)
            eo1 = eval_output[:, :, 200:568, 200:824]
            ep1 = eval_prev[:, :, 200:568, 200:824]
            #eval_loss = criterion(eo1, ep1)+criterion(eo2, ep2)+criterion(eo3, ep3)+criterion(eo4, ep4)
            eval_loss = criterion(eo1, ep1)
            eval_total_loss += eval_loss.item()
            eval_cnt += eval_out.shape[0]

    loss_list.append([total_loss/cnt, eval_total_loss/eval_cnt])
    scheduler.step()
    print('Epoch:' + str(e+1)+'/'+str(epochs) +
          ', Loss_train: ' + str(total_loss/cnt) +
          ', Loss_eval: ' + str(eval_total_loss/eval_cnt))

loss_array = np.array(loss_list)
np.save('loss.npy', loss_array)
plt.figure()
plt.plot(loss_array)
plt.legend(['loss_train', 'loss_eval'])
plt.show()
