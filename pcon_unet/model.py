import os
import sys
import torch
from torch import nn

# Sources:
# smp for collapsing first channel
# https://github.com/NVIDIA/partialconv/blob/master/models/pd_resnet.py

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from partialconv.models.pd_resnet import pdresnet50, Bottleneck
import partialconv.models.pd_resnet


class myresnet(partialconv.models.PDResNet):
    def __init__(self):
        super(myresnet, self).__init__(Bottleneck, [3, 4, 6, 3])

    # Just chops the last few layers off (fc layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


class UNetPconv(nn.Module):
    def __init__(self, *args, **kwargs):
        super(UNetPconv, self).__init__()
        self.encoder = self._load_encoder()
        print(self.encoder)

    def _load_encoder(self):
        m = myresnet()

        # Load the state dict
        # https://www.dropbox.com/sh/t6flbuoipyzqid8/AAAKQPHacndqTga_mqN4GwyEa/pdresnet50?dl=0&subfolder_nav_tracking=1
        sd = torch.load("model_best.pth.tar")["state_dict"]

        # Remove the "module" prefix from the keys, otherwise loading will fail!
        sd = {k[7:]: v for k, v in sd.items()}

        m.load_state_dict(sd)
        return m

    def forward(self, x, mask):
        pass