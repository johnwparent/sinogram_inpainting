import os
import sys

import torch
from torch.nn import functional as F
from torch import nn

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from partialconv.models.partialconv2d import PartialConv2d


class UNetPconv(nn.Module):
    def __init__(self, *args, **kwargs):
        super(UNetPconv, self).__init__()
        self.l1 = PCONVLayer(1, 64)
        self.l3 = PCONVLayer(64, 128)
        self.l4 = PCONVLayer(128, 256)
        # self.l5 = PCONVLayer(256, 512)      # Encoding Section
        # self.l6 = PCONVLayer(512, 512)
        # self.l7 = PCONVLayer(512, 512)
        # self.l8 = PCONVLayer(512, 512)
        # self.l9 = PCONVLayer(512, 512)

        # self.l10 = PCONVLayer(2 * 512, 512, mc=True)
        # self.l11 = PCONVLayer(2 * 512, 512, mc=True)
        # self.l12 = PCONVLayer(2 * 512, 512, mc=True)
        # self.l13 = PCONVLayer(2 * 512, 512, mc=True)
        # self.l14 = PCONVLayer(512 + 256, 256, mc=True)     # Decoding Section
        self.l15 = PCONVLayer(256 + 128, 128, mc=True)
        self.l16 = PCONVLayer(128 + 64, 64, mc=True)
        self.l17 = PCONVLayer(64 + 1, 1, mc=True, bn=False)

    def forward(self, x, mask):
        x1, m1 = self.l1(x, mask)
        x3, m3 = self.l3(x1, m1)
        x4, m4 = self.l4(x3, m3)
        # x5, m5 = self.l5(x4, m4)
        # x6, m6 = self.l6(x5, m5)
        # x7, m7 = self.l7(x6, m6)
        # x8, m8 = self.l8(x7, m7)
        # x9, m9 = self.l9(x8, m8)

        def tcat(m1, m2):
            return torch.cat([F.interpolate(m1, m2.shape[2:]), m2], dim=1)

        def trep(m1, sin, sout):
            return torch.cat(
                             [m1[:,0].unsqueeze(1).repeat(1, sin, 1, 1),
                              m1[:,1].unsqueeze(1).repeat(1, sout, 1, 1)],
                             dim=1)

        # x10, m10 = self.l10(tcat(x9, x8),  trep(tcat(m9, m8), 512, 512))
        # x11, m11 = self.l11(tcat(x10, x7), trep(tcat(m10, m7), 512, 512))
        # x12, m12 = self.l12(tcat(x11, x6), trep(tcat(m11, m6), 512, 512))
        # x13, m13 = self.l13(tcat(x6, x5), trep(tcat(m6, m5), 512, 512))
        # x14, m14 = self.l14(tcat(x13, x4), trep(tcat(m13, m4), 512, 256))
        x15, m15 = self.l15(tcat(x4, x3), trep(tcat(m4, m3), 256, 128))
        x16, m16 = self.l16(tcat(x15, x1), trep(tcat(m15, m1), 128, 64))
        out, _   = self.l17(tcat(x16, x),  trep(tcat(m16, mask), 64, 1))

        return out


class PCONVLayer(nn.Module):
    def __init__(self, in_shape: int, out_shape: int, activation: nn.Module = None, bn:bool = True, mc: bool = False):
        super(PCONVLayer, self).__init__()
        self.cov = PartialConv2d(in_shape, out_shape, 3, 1, padding=1, return_mask=True, multi_channel=mc)
        self.act = activation if activation else nn.LeakyReLU(negative_slope=0.2)
        self.batchnorm = nn.BatchNorm2d(out_shape) if bn else lambda x: x

    def forward(self, x, mask):
        x, mask = self.cov(x, mask_in=mask)
        x = self.batchnorm(x)
        x = self.act(x)
        return x, mask
