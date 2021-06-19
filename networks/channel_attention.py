""" -*- coding: utf-8 -*-
@ Time: 2021/3/9 16:14
@ author: Zhang Chi
@ e-mail: 220200785@mail.seu.edu.cn
@ file: unet_se.py
@ project: UV_Net_paper
"""


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()

        # encoder
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = DoubleConv(512, 1024)

        # decoder
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out


class SE_Module(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SE_Module, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_features=channel, out_features=channel // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel // ratio, out_features=channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        # expand_as 输出相同尺寸的tensor
        return x * z.expand_as(x)


class SE_ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsampling=False, expansion=4):
        super(SE_ResNetBlock, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch * self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_ch * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class UNet_SE(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNet_SE, self).__init__()

        # encoder
        self.conv1 = DoubleConv(in_ch, 64)
        self.se1 = SE_Module(64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(64, 128)
        self.se2 = SE_Module(128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(128, 256)
        self.se3 = SE_Module(256)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(256, 512)
        self.se4 = SE_Module(512)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = DoubleConv(512, 1024)

        # decoder
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        conv1_1 = self.conv1(x)
        # tensor不能随意赋值，张量先求道，求导后被修改，然后等到计算的时候，张量发生变化，就会报错
        conv1 = conv1_1 + self.se1(conv1_1)
        pool1 = self.pool1(conv1)

        conv2_1 = self.conv2(pool1)
        conv2 = conv2_1 + self.se2(conv2_1)
        pool2 = self.pool2(conv2)

        conv3_1 = self.conv3(pool2)
        conv3 = conv3_1 + self.se3(conv3_1)
        pool3 = self.pool3(conv3)

        conv4_1 = self.conv4(pool3)
        conv4 = conv4_1 + self.se4(conv4_1)
        pool4 = self.pool4(conv4)

        conv5 = self.conv5(pool4)

        up_6 = self.up6(conv5)
        merge6 = torch.cat([up_6, conv4], dim=1)
        conv6 = self.conv6(merge6)

        up_7 = self.up7(conv6)
        merge7 = torch.cat([up_7, conv3], dim=1)
        conv7 = self.conv7(merge7)

        up_8 = self.up8(conv7)
        merge8 = torch.cat([up_8, conv2], dim=1)
        conv8 = self.conv8(merge8)

        up_9 = self.up9(conv8)
        merge9 = torch.cat([up_9, conv1], dim=1)
        c9 = self.conv9(merge9)

        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out
