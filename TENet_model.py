import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision
from torchvision import transforms, models
import numpy as np
# import skvideo.io
# import imageio
import cv2


class NeighborDataset(Dataset):
    def __init__(self, file_dir):
        super(NeighborDataset, self).__init__()
        self.imgs = skvideo.io.vread(file_dir)
        # self.imgs = imageio.get_reader(file_dir, 'ffmpeg')

    def __len__(self):
        return self.imgs.shape[0]-1

    def __getitem__(self, idx):
        transform = transforms.Compose([transforms.ToTensor()])

        # last_frame = self.imgs[idx].transpose((2,0,1))
        pre = cv2.cvtColor(self.imgs[idx], cv2.COLOR_BGR2GRAY)
        pre_resize = cv2.resize(pre[150:600, 100:900], (256, 256))
        curr = cv2.cvtColor(self.imgs[idx+1], cv2.COLOR_BGR2GRAY)
        curr_resize = cv2.resize(curr[150:600, 100:900], (256, 256))
        return transform(np.array([pre_resize, curr_resize]).transpose(1, 2, 0)), transform(pre), transform(curr)


class vNeighborDataset(Dataset):
    def __init__(self, videos):
        super(vNeighborDataset, self).__init__()
        # self.videos = videos
        # self.frame_num = []
        self.imgs = skvideo.io.vread(videos[0])
        self.frame_num = [skvideo.io.vread(videos[0]).shape[0]-1]
        for video in videos[1:]:
            self.frame_num.append(skvideo.io.vread(video).shape[0]-1)
            self.imgs = np.concatenate(
                (self.imgs, skvideo.io.vread(video)), axis=0)
        # self.videos = skvideo.io.vread(file_dir)
        # self.imgs = imageio.get_reader(file_dir, 'ffmpeg')

    def __len__(self):
        return np.sum(np.array(self.frame_num))

    def __getitem__(self, idx):
        transform = transforms.Compose([transforms.ToTensor()])
        cnt, cnt_frames = 0, 0
        # print(idx)
        for video_frames in self.frame_num:
            cnt_frames += video_frames
            if idx < cnt_frames:
                break
            cnt += 1
        # print(cnt)
        # imgs = skvideo.io.vread(self.videos[cnt])
        index = idx + cnt

        # last_frame = self.imgs[idx].transpose((2,0,1))
        pre = cv2.cvtColor(self.imgs[index], cv2.COLOR_BGR2GRAY)
        pre_resize = cv2.resize(pre[150:600, 100:900], (256, 256))
        curr = cv2.cvtColor(self.imgs[index+1], cv2.COLOR_BGR2GRAY)
        curr_resize = cv2.resize(curr[150:600, 100:900], (256, 256))
        return transform(np.array([pre_resize, curr_resize]).transpose(1, 2, 0)), transform(pre), transform(curr)


class ImgsDataset(Dataset):
    def __init__(self, imglist, lables):
        super(ImgsDataset, self).__init__()
        # self.videos = videos
        # self.frame_num = []
        imglist.sort()
        self.ilist = imglist
        self.imgs = []
        for addr in imglist:
            self.imgs.append(cv2.imread(addr, flags=cv2.IMREAD_GRAYSCALE))
        self.label = np.load(lables)
        # self.videos = skvideo.io.vread(file_dir)
        # self.imgs = imageio.get_reader(file_dir, 'ffmpeg')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        transform = transforms.Compose([transforms.ToTensor()])
        img_cat = self.imgs[idx]
        # print(img_cat.shape)
        img_input = np.array(
            [img_cat[:, :256], img_cat[:, 256:]]).transpose(1, 2, 0)

        # bias = self.label[:, idx]
        bias = self.label[idx, :]

        return transform(img_input), bias, self.ilist[idx]


class fsImgsDataset(Dataset):
    def __init__(self, imglist):
        super(fsImgsDataset, self).__init__()
        # self.videos = videos
        # self.frame_num = []
        self.imgs = []
        for addr in imglist:
            self.imgs.append(cv2.imread(addr, flags=cv2.IMREAD_GRAYSCALE))

        # self.videos = skvideo.io.vread(file_dir)
        # self.imgs = imageio.get_reader(file_dir, 'ffmpeg')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        transform = transforms.Compose([transforms.ToTensor()])
        img_cat = self.imgs[idx]
        prev = img_cat[:, :1024]
        curr = img_cat[:, 1024:]
        img_input = np.array(
            [cv2.resize(prev[150:600, 100:900], (256, 256)), cv2.resize(curr[150:600, 100:900], (256, 256))]).transpose(1, 2, 0)
        # bias = self.label[idx, :]

        return transform(img_input), transform(prev), transform(curr)

class efsImgsDataset(Dataset):
    def __init__(self, imglist):
        super(efsImgsDataset, self).__init__()
        # self.videos = videos
        # self.frame_num = []
        self.imgs = []
        for addr in imglist:
            self.imgs.append(cv2.imread(addr, flags=cv2.IMREAD_GRAYSCALE))

        # self.videos = skvideo.io.vread(file_dir)
        # self.imgs = imageio.get_reader(file_dir, 'ffmpeg')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        transform = transforms.Compose([transforms.ToTensor()])
        img_cat = self.imgs[idx]
        prev = img_cat[:, :1024]
        curr = img_cat[:, 1024:]
        img_input = np.array(
            [cv2.equalizeHist(cv2.resize(prev[150:600, 100:900], (256, 256))), cv2.equalizeHist(cv2.resize(curr[150:600, 100:900], (256, 256)))]).transpose(1, 2, 0)
        # bias = self.label[idx, :]

        return transform(img_input), transform(prev), transform(curr)

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()

        self.localization0 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=7, stride=3, padding=3),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=5, stride=3, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.fc_loc0 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 1 * 2, bias=False)
        )
        # self.batchsize = batch_size

    def forward(self, input):
        x = self.localization0(input)
        x = x.view(-1, 16 * 4 * 4)
        theta = self.fc_loc0(x)
        # theta = theta.view(-1, 2, 3)
        # theta = theta + self.bias
        # grid = F.affine_grid(theta, input.size())
        # output = F.grid_sample(input[:,5:8,:,:], grid)
        theta = theta.view(-1, 2, 1)
        bias = torch.Tensor([[1, 0], [0, 1]]).expand(theta.shape[0], 2, 2)
        if torch.cuda.is_available():
            bias = bias.cuda()

        affine = torch.cat([bias, theta], dim=2)
        return affine

class SSTN(nn.Module):
    def __init__(self):
        super(SSTN, self).__init__()

        self.localization0 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=7, stride=3, padding=3),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=5, stride=3, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.fc_loc0 = nn.Sequential(
            nn.Linear(16 * 4 * 4 * 2, 1 * 2, bias=False)
        )
        # self.batchsize = batch_size

    def forward(self, input):
        input_d = F.interpolate(input[:,:,64:192,64:192], scale_factor=2)
        x0 = self.localization0(input)
        x1 = self.localization0(input_d)
        x2 = torch.cat([x0.view(-1, 16 * 4 * 4),x1.view(-1, 16 * 4 * 4)],1)
        theta = self.fc_loc0(x2)
        theta = theta.view(-1, 2, 1)
        bias = torch.Tensor([[1, 0], [0, 1]]).expand(theta.shape[0], 2, 2)
        if torch.cuda.is_available():
            bias = bias.cuda()

        affine = torch.cat([bias, theta], dim=2)
        return affine
class RSSTN(nn.Module):
    def __init__(self):
        super(RSSTN, self).__init__()

        self.localization0 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=7, stride=3, padding=3),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=5, stride=3, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        )
        self.localization1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=7, stride=3, padding=3),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=5, stride=3, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.fc_loc0 = nn.Sequential(
            nn.Linear(16 * 4 * 4 * 2, 1 * 2, bias=False)
        )
        # self.batchsize = batch_size

    def forward(self, input):
        input_d = F.interpolate(input[:,:,64:192,64:192], scale_factor=2)
        x0 = self.localization0(input)
        x1 = self.localization1(input_d)
        x2 = torch.cat([x0.view(-1, 16 * 4 * 4),x1.view(-1, 16 * 4 * 4)],1)
        theta = self.fc_loc0(x2)
        theta = theta.view(-1, 2, 1)
        bias = torch.Tensor([[1, 0], [0, 1]]).expand(theta.shape[0], 2, 2)
        if torch.cuda.is_available():
            bias = bias.cuda()

        affine = torch.cat([bias, theta], dim=2)
        return affine
        
class fSTN(nn.Module):
    def __init__(self):
        super(fSTN, self).__init__()

        self.localization0 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=7, stride=3, padding=3),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=5, stride=3, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.fc_loc0 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 4 * 4 * 4, bias=False),
            nn.Linear(4 * 4 * 4, 1 * 2, bias=False)
        )
        # self.batchsize = batch_size

    def forward(self, input):
        x = self.localization0(input)
        x = x.view(-1, 16 * 4 * 4)
        theta = self.fc_loc0(x)
        # theta = theta.view(-1, 2, 3)
        # theta = theta + self.bias
        # grid = F.affine_grid(theta, input.size())
        # output = F.grid_sample(input[:,5:8,:,:], grid)
        theta = theta.view(-1, 2, 1)
        bias = torch.Tensor([[1, 0], [0, 1]]).expand(theta.shape[0], 2, 2)
        if torch.cuda.is_available():
            bias = bias.cuda()

        affine = torch.cat([bias, theta], dim=2)
        return affine

class STNi(nn.Module):
    def __init__(self):
        super(STNi, self).__init__()

        self.localization0 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=7, stride=3, padding=3),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=5, stride=3, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.fc_loc0 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 1 * 2, bias=False)
        )

    def forward(self, input):
        x = self.localization0(input)
        x = x.view(-1, 16 * 4 * 4)
        theta = self.fc_loc0(x)
        theta = theta.view(-1, 2)
        return theta

class fixSTN(nn.Module):
    def __init__(self):
        super(fixSTN, self).__init__()

        self.localization0 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=7, stride=3, padding=3),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=5, stride=3, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        )
        for p in self.parameters():
            p.requires_grad = False
        self.fc_loc0 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 1 * 2, bias=False)
        )
        # self.batchsize = batch_size

    def forward(self, input):
        x = self.localization0(input)
        x = x.view(-1, 16 * 4 * 4)
        theta = self.fc_loc0(x)
        # theta = theta.view(-1, 2, 3)
        # theta = theta + self.bias
        # grid = F.affine_grid(theta, input.size())
        # output = F.grid_sample(input[:,5:8,:,:], grid)
        theta = theta.view(-1, 2, 1)
        bias = torch.Tensor([[1, 0], [0, 1]]).expand(theta.shape[0], 2, 2)
        if torch.cuda.is_available():
            bias = bias.cuda()

        affine = torch.cat([bias, theta], dim=2)
        return affine

class STNres(nn.Module):
    def __init__(self):
        super(STNres, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(in_features=512, out_features=2, bias=True)
        # self.batchsize = batch_size

    def forward(self, input):
        theta = self.backbone(input)
        # theta = theta.view(-1, 2, 3)
        # theta = theta + self.bias
        # grid = F.affine_grid(theta, input.size())
        # output = F.grid_sample(input[:,5:8,:,:], grid)
        theta = theta.view(-1, 2, 1)
        bias = torch.Tensor([[1, 0], [0, 1]]).expand(theta.shape[0], 2, 2)
        if torch.cuda.is_available():
            bias = bias.cuda()

        affine = torch.cat([bias, theta], dim=2)
        return affine

class STNresi(nn.Module):
    def __init__(self):
        super(STNresi, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(in_features=512, out_features=2, bias=True)
        # self.batchsize = batch_size

    def forward(self, input):
        theta = self.backbone(input)
        # theta = theta.view(-1, 2, 3)
        # theta = theta + self.bias
        # grid = F.affine_grid(theta, input.size())
        # output = F.grid_sample(input[:,5:8,:,:], grid)
        theta = theta.view(-1, 2)
        return theta
        
class STNsfi(nn.Module):
    def __init__(self):
        super(STNsfi, self).__init__()
        self.backbone = models.shufflenet_v2_x0_5()
        self.backbone.conv1[0] = nn.Conv2d(2, 24, kernel_size=(
            3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.backbone.fc = nn.Linear(
            in_features=1024, out_features=2, bias=True)
        # self.batchsize = batch_size

    def forward(self, input):
        theta = self.backbone(input)
        theta = theta.view(-1, 2)
        return theta
        
class STNsf(nn.Module):
    def __init__(self):
        super(STNsf, self).__init__()
        self.backbone = models.shufflenet_v2_x0_5()
        self.backbone.conv1[0] = nn.Conv2d(2, 24, kernel_size=(
            3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.backbone.fc = nn.Linear(
            in_features=1024, out_features=2, bias=False)
        # self.batchsize = batch_size

    def forward(self, input):
        theta = self.backbone(input)
        theta = theta.view(-1, 2, 1)
        bias = torch.Tensor([[1, 0], [0, 1]]).expand(theta.shape[0], 2, 2)
        if torch.cuda.is_available():
            bias = bias.cuda()
        affine = torch.cat([bias, theta], dim=2)
        return affine