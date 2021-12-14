import os
import sys

import torch
import torch.nn
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from partialconv.models.partialconv2d import PartialConv2d

# We need to create a UNet as the Generator for the GAN as to span feature space
# USE the normal partial resnet pretrained w/ transfer learning to discriminate
# generator should just encode, then noise, then generate.

class UNetPconv(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.down1 = downStep(1, 64)
        self.down2 = downStep(64, 128)
        self.down3 = downStep(128, 256)
        # self.down4 = downStep(256, 512)
        # self.down5 = downStep(512, 1024)
        # self.up1 = upStep(1536, 512)
        # self.up2 = upStep(768, 256)
        self.up3 = upStep(384, 128)
        self.up4 = upStep(192, 64, withReLU = False)

    def forward(self, x, m):
        x1, m1 = self.down1(x, m)
        x2, m2 = self.down2(x1, m1)
        x3, m3 = self.down3(x2, m2)
        # x4, m4 = self.down4(x3, m3)
        # x5, m5 = self.down5(x4, m4)

        # x, m = self.up1(x5, x4, m5, m4)
        # x, m = self.up2(x, x3, m, m3)
        x, m = self.up3(x3, x2, m3, m2)
        x, m = self.up4(x, x1, m, m1)

        return x, m


class downStep(nn.Module):
    def __init__(self, inC, outC):
        super(downStep, self).__init__()
        self.convLayer1 = PartialConv2d(inC, outC, kernel_size = 3, padding= 1, bias=False, multi_channel=True, return_mask=True)
        self.convLayer2 = PartialConv2d(outC, outC, kernel_size = 3, padding= 1, bias=False, multi_channel=True, return_mask=True)
        #self.batchNorm = nn.BatchNorm2d(outC)
        self.relU = nn.ReLU(inplace=True)

    def forward(self, x, m):
        # todo
        x, m = self.convLayer1(x, m)
        x = self.relU(x)
        #x = self.batchNorm(x)
        x, m = self.convLayer2(x, m)
        x = self.relU(x)
        #x = self.batchNorm(x)
        return x, m


class upStep(nn.Module):
    def __init__(self, inC, outC, withReLU=True):
        super(upStep, self).__init__()
        self.convLayer1 = PartialConv2d(inC, outC, kernel_size = 3, padding= 1, bias=False, multi_channel=True, return_mask=True)
        self.convLayer2 = PartialConv2d(outC, outC, kernel_size = 3, padding= 1, bias=False, multi_channel=True, return_mask=True)
        self.convLayer3 = PartialConv2d(outC, 1, kernel_size = 3, padding= 1, bias=False, multi_channel=True, return_mask=True)
        self.sig = nn.Sigmoid()
        self.relU = nn.ReLU(inplace=True)
        self.withReLU = withReLU
        self.up = nn.Upsample(scale_factor = 2, mode = 'nearest')

    def checkAndPadding(self, var1, var2):
        if var1.size(2) > var2.size(2) or var1.size(3) > var2.size(3):
            var1 = var1[:, :, :var2.size(2), :var2.size(3)]
        else:
            pad = [0, 0, int(var2.size(2) - var1.size(2)), int(var2.size(3) - var1.size(3))]
            var1 = F.pad(var1, pad)
        return var1, var2

    def forward(self, x, x_down, m, m_down ):
        x = self.up(x)
        m = self.up(m.float())
        x, x_down = self.checkAndPadding(x, x_down)
        m, m_down = self.checkAndPadding(m, m_down)
        x = torch.cat([x, x_down], 1)
        m = torch.cat([m, m_down], 1)
        x, m = self.convLayer1(x, m)
        if self.withReLU:
            x = self.relU(x)
            #x = self.batchNorm(x)
        x, m = self.convLayer2(x, m)
        if self.withReLU:
            x = self.relU(x)
            #x = self.batchNorm(x)
        else:
            x, m = self.convLayer3(x, m)
            x = self.sig(x)
        return x, m
