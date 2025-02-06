""" Full assembly of the parts to form the complete network """


# Taken from repo online
# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

import torch
import torch.nn as nn


from unet_parts import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x) # 64 -> 64
        x2 = self.down1(x1) # 64 -> 32
        x3 = self.down2(x2) # 32 -> 16
        x4 = self.down3(x3) # 16 -> 8
        x5 = self.down4(x4) # 8 -> 4
        x = self.up1(x5, x4) # 4 -> 8
        x = self.up2(x, x3) # 8 -> 16
        x = self.up3(x, x2) # 16 -> 32
        x = self.up4(x, x1) # 32 -> 64
        logits = self.outc(x) # 64 -> 64
        return logits