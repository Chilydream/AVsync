import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb
from yolo_models.vlad_standard import VladPooling


def conv_block(input_dim, filters, strides):
    return nn.Sequential(
        nn.Conv2d(input_dim, filters[0], kernel_size=1, bias=False, stride=strides),
        nn.BatchNorm2d(filters[0]),
        nn.ReLU(True),
        nn.Conv2d(filters[0], filters[1], kernel_size=3, bias=False, padding=1),
        nn.BatchNorm2d(filters[1]),
        nn.ReLU(True),
        nn.Conv2d(filters[1], filters[2], kernel_size=1, bias=False),
        nn.BatchNorm2d(filters[2])
    )


def shortcut(input_dim, filters, strides):
    return nn.Sequential(
        nn.Conv2d(input_dim, filters[2], kernel_size=1, bias=False, stride=strides),
        nn.BatchNorm2d(filters[2])
    )


def identity_block(input_dim, filters):
    return nn.Sequential(
        nn.Conv2d(input_dim, filters[0], kernel_size=1, bias=False),
        nn.BatchNorm2d(filters[0]),
        nn.ReLU(True),
        nn.Conv2d(filters[0], filters[1], kernel_size=3, bias=False, padding=1),
        nn.BatchNorm2d(filters[1]),
        nn.ReLU(True),
        nn.Conv2d(filters[1], filters[2], kernel_size=1, bias=False),
        nn.BatchNorm2d(filters[2])
    )


input_dim = 1
block1_input = 64
filters = [[48, 48, 96], [96, 96, 128], [128, 128, 256], [256, 256, 512]]
conv_block_input = 32

ghost_centers = 0

vlad_centers = 10


class Resnet34(nn.Module):
    def __init__(self):
        super(Resnet34, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, block1_input, kernel_size=7, bias=False, padding=3)
        self.norm_layer = nn.BatchNorm2d(block1_input)
        self.relu = nn.ReLU(True)
        self.max_pool1 = nn.MaxPool2d((2, 2))

        self.conv_block1 = conv_block(64, filters[0], (1, 1))
        self.shortcut1 = shortcut(64, filters[0], (1, 1))
        self.identity_block1 = identity_block(filters[0][2], filters[0])

        self.conv_block2 = conv_block(filters[0][2], filters[1], (2, 2))
        self.shortcut2 = shortcut(filters[0][2], filters[1], (2, 2))
        self.identity_block2 = identity_block(filters[1][2], filters[1])

        self.conv_block3 = conv_block(filters[1][2], filters[2], (2, 2))
        self.shortcut3 = shortcut(filters[1][2], filters[2], (2, 2))
        self.identity_block3 = identity_block(filters[2][2], filters[2])

        self.conv_block4 = conv_block(filters[2][2], filters[3], (2, 2))
        self.shortcut4 = shortcut(filters[2][2], filters[3], (2, 2))
        self.identity_block4 = identity_block(filters[3][2], filters[3])
        self.max_pool2 = nn.MaxPool2d((3, 1), stride=(2, 1))

        self.conv2 = nn.Conv2d(filters[3][2], filters[3][2], kernel_size=(7, 1), bias=True)
        self.conv3 = nn.Conv2d(filters[3][2], vlad_centers, kernel_size=(7, 1), bias=True)

        # 对角初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Parameter):
                nn.init.orthogonal_(m.weight)

    def forward(self, x):

        x1 = self.conv1(x)
        x1 = self.norm_layer(x1)
        x1 = self.relu(x1)
        x1 = self.max_pool1(x1)

        # 两层
        x2 = self.relu(self.conv_block1(x1).add(self.shortcut1(x1)))
        x2 = self.relu(self.identity_block1(x2).add(x2))

        # 三层:
        x3 = self.relu(self.conv_block2(x2).add(self.shortcut2(x2)))
        x3 = self.relu(self.identity_block2(x3).add(x3))
        x3 = self.relu(self.identity_block2(x3).add(x3))

        # 三层:
        x4 = self.relu(self.conv_block3(x3).add(self.shortcut3(x3)))
        x4 = self.relu(self.identity_block3(x4).add(x4))
        x4 = self.relu(self.identity_block3(x4).add(x4))

        # 三层:
        x5 = self.relu(self.conv_block4(x4).add(self.shortcut4(x4)))
        x5 = self.relu(self.identity_block4(x5).add(x5))
        x5 = self.relu(self.identity_block4(x5).add(x5))
        x5 = self.max_pool2(x5)
        # [batch, 512, 7, 16]

        # ============================
        #   Fully Connected Block 1
        # ============================
        feat_broadcast = self.relu(self.conv2(x5))
        # (batch_size, 512, 1, 16)

        x_center = self.conv3(x5)

        return feat_broadcast, x_center


class Resnet34NetVLAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_resnet = Resnet34()
        self.vlad_layer = VladPooling(k_centers=10, feature_size=512)
        self.dense = torch.nn.Linear(5120, 512)

    def forward(self, x):
        feat_broadcast, x_center = self.encoder_resnet(x)
        # feat_broadcast:(batch, 512, 1, 16)

        vlad_output = self.vlad_layer(feat_broadcast, x_center)

        final_output = torch.nn.functional.relu(self.dense(vlad_output))

        final_output_l2 = torch.nn.functional.normalize(final_output, dim=1, p=2)
        return final_output_l2


if __name__ == "__main__":
    input_value = torch.rand((4, 1, 257, 250))
    # (batch,channel,257,时间序列长度)

    model = Resnet34NetVLAD()

    out = model(input_value)
    # (batch,512)
    print(out)
    print(out.shape)
