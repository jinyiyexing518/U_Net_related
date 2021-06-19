""" -*- coding: utf-8 -*-
@ Time: 2021/3/22 14:12
@ author: Zhang Chi
@ e-mail: zhang_chi@seu.edu.cn
@ file: unet_bam.py
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


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class UNet_BAM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNet_BAM, self).__init__()

        # encoder
        self.conv1 = DoubleConv(in_ch, 64)
        self.bam1_1 = ChannelAttention(64)
        self.bam1_2 = SpatialAttention()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(64, 128)
        self.bam2_1 = ChannelAttention(128)
        self.bam2_2 = SpatialAttention()
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(128, 256)
        self.bam3_1 = ChannelAttention(256)
        self.bam3_2 = SpatialAttention()
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(256, 512)
        self.bam4_1 = ChannelAttention(512)
        self.bam4_2 = SpatialAttention()
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
        conv1 = self.bam1_1(conv1_1) * conv1_1 + self.bam1_2(conv1_1) * conv1_1
        pool1 = self.pool1(conv1)

        conv2_1 = self.conv2(pool1)
        conv2 = self.bam2_1(conv2_1) * conv2_1 + self.bam2_2(conv2_1) * conv2_1
        pool2 = self.pool2(conv2)

        conv3_1 = self.conv3(pool2)
        conv3 = self.bam3_1(conv3_1) * conv3_1 + self.bam3_2(conv3_1) * conv3_1
        pool3 = self.pool3(conv3)

        conv4_1 = self.conv4(pool3)
        conv4 = self.bam4_1(conv4_1) * conv4_1 + self.bam4_2(conv4_1) * conv4_1
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
