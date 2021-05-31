# Import numpy and OpenCV
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import os.path as op
import sys
import time
from LSTM_model import lstm
from TENet_model import *
from torchvision import transforms as trans
from scipy import interpolate as intp
import color_transfer


# input
vds_dir = '02.mp4'
cap = cv2.VideoCapture('Demo/'+vds_dir)

# motion parameters of the robot
period_frames = 43  # 01:29 02:43
frequency = 10  # 01:15 02:10

imu_bias = 18  # 01:2 02:18

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('f,w,h = %d, %d, %d', (n_frames, w, h))

# Read first frame
_, prev = cap.read()

# Convert frame to grayscale
# prev = color_transfer.color_transfer(imgtgt, prev)
prev_lab = cv2.cvtColor(prev, cv2.COLOR_BGR2LAB).astype("float32")
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# Pre-define transformation-store array
transforms = np.zeros((n_frames-1, 3), np.float32)
trajectory_curr = np.zeros((1, 3), np.float32)
trajectory_pre = np.zeros((1, 3), np.float32)
trajectory_array = []

# cropping
borderup, borderdown, borderleft, borderright = 0, h, 0, w

# create TENet
TENet_model = STN()

# read pth
TENet_model.eval()
TENet_model.load_state_dict(torch.load(
    op.join(sys.path[0], 'pth_lib/tenet.pth'), map_location=lambda storage, loc: storage))

transform = trans.Compose([trans.ToTensor()])
normtrans = trans.Compose([trans.ToTensor(), trans.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# create lstm
input_size, output_size, lstm_layers = 40, 40, 2
lstm_model = lstm(input_size, input_size*2, output_size, lstm_layers)

# read pth
lstm_model.eval()
lstm_model.load_state_dict(torch.load(
    op.join(sys.path[0], 'pth_lib/lstm.pth'), map_location=torch.device('cpu')))


pre_scal = 3677.2366

array_length = int(15*input_size/frequency)
space_input = np.linspace(0, array_length, array_length)


start_point = array_length
Homo = np.array([[0.9993, -0.0868, 30.5632],
                 [0.0382, 0.9538, 99.9099], [-0.0000, -0.0001, 1.0356]])

# KF
unsmooth_pre = []
smooth_pre = []
# Q, R = 1e-6, 1e-3
Q, R = 1e-6, 4e-5
K, X, P, P_pre = 0, 0, 0, 0

unsmooth_prex = []
smooth_prex = []
Qx, Rx = 1e-6, 2e-4
Kx, Xx, Px, P_prex = 0, 0, 0, 0

imgs_store = []


t_start = time.clock()

for i in range(n_frames):

    bx, by, bw, bh = 100, 150, 900, 600
    success, curr = cap.read()

    if not success:
        break
    frame_test = cv2.warpPerspective(curr, Homo, (1024, 768))

    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    pre_downsample = cv2.equalizeHist(
        cv2.resize(prev_gray[by:bh, bx:bw], (256, 256)))
    curr_downsample = cv2.equalizeHist(
        cv2.resize(curr_gray[by:bh, bx:bw], (256, 256)))
    imgs = transform(
        np.array([pre_downsample, curr_downsample]).transpose(1, 2, 0))

    # Extract traslation
    theta = TENet_model(imgs.unsqueeze(0))
    dd = theta.data[0, :, 2].numpy()
    # dx = -w/2*dd[0]
    # dy = -h/2*dd[1]
    dx = -1024/2*dd[0]
    dy = -768/2*dd[1]

    da = 0

    # Store transformation
    transforms[i] = [dx, dy, da]

    # trajectory_pre = trajectory_curr
    trajectory_curr = trajectory_curr + transforms[i]

    trajectory_array.append(trajectory_curr[0, :])

    if i >= start_point:

        history_array = np.array(trajectory_array)
        history_array = history_array.astype('float32')/pre_scal
        history_points = history_array[-(array_length+1):-1, 1]
        history_pointsx = history_array[-(array_length+1):-1, 0]

        input_ready = torch.from_numpy(history_points.reshape(1, 1, -1))
        input_readyx = torch.from_numpy(history_pointsx.reshape(1, 1, -1))
        input_normalized = F.interpolate(
            input_ready, scale_factor=input_size/array_length, mode='linear')
        input_normalizedx = F.interpolate(
            input_readyx, scale_factor=input_size/array_length, mode='linear')
        var_input = Variable(input_normalized)
        var_inputx = Variable(input_normalizedx)
        # f1 = intp.interp1d(x, y, kind='linear')
        # print(var_input.size())
        # break
        output = lstm_model(var_input)
        outputx = lstm_model(var_inputx)
        output = F.interpolate(
            output, scale_factor=array_length/input_size, mode='linear')
        outputx = F.interpolate(
            outputx, scale_factor=array_length/input_size, mode='linear')
        output = output.view(-1).data.numpy()
        outputx = outputx.view(-1).data.numpy()

        index_curr = i
        index_history = ((index_curr-imu_bias)//period_frames) * \
            period_frames-1+imu_bias
        d_index = index_curr - index_history
        # if frames == 0 or index_curr % period_frames == 0:
        #     value_prediction = output[period_frames-d_index]
        value_history = history_array[index_history, 1]
        value_historyx = history_array[index_history, 0]
        # print(value_history)
        value_prediction = output[period_frames-d_index]
        value_predictionx = outputx[period_frames-d_index]
        # print(value_prediction)
        value_inter = value_history + d_index * \
            (value_prediction-value_history)/period_frames
        value_interx = value_historyx + d_index * \
            (value_predictionx-value_historyx)/period_frames
        # print(value_inter)
        scaled_value = pre_scal*value_inter
        scaled_valuex = pre_scal*value_interx
        unsmooth_pre.append(scaled_value)
        unsmooth_prex.append(scaled_valuex)

        if i-start_point < 2:
            smooth_pre.append(scaled_value)
            X_pre = smooth_pre[-1]
            P_pre = smooth_pre[-1]-smooth_pre[-1]
            smooth_prex.append(scaled_valuex)
            X_prex = smooth_prex[-1]
            P_prex = smooth_prex[-1]-smooth_prex[-1]

        if i-start_point >= 2:
            K = P_pre / (P_pre + R)
            X = X_pre + K * (scaled_value-X_pre)
            P = P_pre - K * P_pre + Q
            P_pre = P
            X_pre = X
            smooth_pre.append(X)
            Kx = P_prex / (P_prex + Rx)
            Xx = X_prex + Kx * (scaled_valuex-X_prex)
            Px = P_prex - Kx * P_prex + Qx
            P_prex = Px
            X_prex = Xx
            smooth_prex.append(Xx)

        x_modify = pre_scal*history_array[-1, 0] - smooth_prex[-1]
        y_modify = pre_scal*history_array[-1, 1] - smooth_pre[-1]

        t = np.zeros((2, 3), np.float32)
        t[0, 0] = 1
        t[0, 1] = 0
        t[1, 0] = 0
        t[1, 1] = 1
        t[0, 2] = x_modify
        t[1, 2] = y_modify

        # theta = torch.tensor(t).unsqueeze(0)
        # prev_t = transform(prev).unsqueeze(0)
        # curr_t = transform(curr).unsqueeze(0)
        # grid = F.affine_grid(theta/768, prev_t.size())
        # output = F.grid_sample(curr_t, grid)

        # frame_stabilized = 255*output[0].detach().numpy().transpose(1, 2, 0)
        # cv2.imwrite('output/'+'ttt'+str(i+1)+'.jpg', frame_stabilized)
        frame_stabilized = cv2.warpAffine(curr, t, (w, h))
        imgs_store.append(frame_stabilized)

    prev_gray = curr_gray

    # print("Frame: " + str(i+1) + "/" + str(n_frames-1))

t_end = time.clock()
print('time:', (t_end-t_start)/n_frames*1000)


fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('Results/ITLK-'+vds_dir, fourcc, fps, (w, h))

for img in imgs_store:
    out.write(img)
