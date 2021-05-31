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
import torch.nn.functional as F


# 导入原始视频
# input
vds_dir = '02.mp4'
cap = cv2.VideoCapture('Demo/'+vds_dir)

# motion parameters of the robot
period_frames = 43  # 01:29 02:43
frequency = 10  # 01:15 02:10

imu_bias = 18  # 01:2 02:18

maxcners = 200
qualityLe = 0.01

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('f,w,h = %d, %d, %d', (n_frames, w, h))


# Read first frame
_, prev = cap.read()

# Convert frame to grayscale
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)


# Pre-define transformation-store array
transforms = np.zeros((n_frames-1, 3), np.float32)
trajectory_curr = np.zeros((1, 3), np.float32)
trajectory_pre = np.zeros((1, 3), np.float32)
trajectory_array = []

# cropping
borderup, borderdown, borderleft, borderright = 0, h, 0, w

# create lstm
input_size, output_size, lstm_layers = 40, 40, 2
model = lstm(input_size, input_size*2, output_size, lstm_layers)

# read
model.eval()
model.load_state_dict(torch.load(
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
Q, R = 1e-6, 4e-5
K, X, P, P_pre = 0, 0, 0, 0

unsmooth_prex = []
smooth_prex = []
Qx, Rx = 1e-6, 2e-4
Kx, Xx, Px, P_prex = 0, 0, 0, 0

imgs_store = []
t_start = time.clock()
bx, by, bw, bh = 100, 150, 900, 600

for i in range(n_frames):
    # Calculate the interest region, the goodfeature detector fails with black borders
    mask_use = np.zeros(prev_gray.shape, np.uint8)
    # bx, by, bw, bh = round(max(borderleft+50, 0)), round(max(borderup+50, 0)
    bx, by, bw, bh = 100, 150, 900, 600

    mask_use[by:bh, bx:bw] = prev_gray[by:bh, bx:bw]

    # good featrures detect feature points in previous frame
    goodfeatures = cv2.goodFeaturesToTrack(prev_gray,
                                           mask=mask_use,
                                           maxCorners=maxcners,
                                           qualityLevel=qualityLe,
                                           minDistance=30,
                                           blockSize=3)

    prev_pts = goodfeatures

    # Read next frame
    success, curr = cap.read()

    if not success:
        break
    # cv2.imwrite('ground/'+'prev'+str(i+2)+'.jpg', curr)
    frame_test = cv2.warpPerspective(curr, Homo, (1024, 768))
    # Convert to grayscale
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow (i.e. track feature points)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None)

    # Sanity check
    assert prev_pts.shape == curr_pts.shape

    # Filter only valid points
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    # rigid transformation
    # Find transformation matrix
    # will only work with OpenCV4 or more
    m, cc = cv2.estimateAffinePartial2D(prev_pts, curr_pts)

    # Extract traslation
    dx = -m[0, 2]
    dy = -m[1, 2]

    # Extract rotation angle
    da = -np.arctan2(m[1, 0], m[0, 0])

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

        # print(history_points)
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
        output = model(var_input)
        outputx = model(var_inputx)
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

        frame_stabilized = cv2.warpAffine(curr, t, (w, h))
        imgs_store.append(frame_stabilized)

    prev_gray = curr_gray

    # print("Frame: " + str(i+1) + "/" + str(n_frames-1) +
    #       " -  Tracked points : " + str(len(prev_pts)))

t_end = time.clock()
print('time:', (t_end-t_start)/n_frames*1000)


fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('results/IOLK-'+vds_dir+'.mp4', fourcc, fps, (w, h))

for img in imgs_store:
    out.write(img)
