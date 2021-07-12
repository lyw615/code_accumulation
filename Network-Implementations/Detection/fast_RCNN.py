import torch.nn as nn
import torch

import torchvision
import torch
import torch.nn as nn


class Fast_RCNN(nn.Module):
    def __init__(self, num_cls):
        super(Fast_RCNN, self).__init__()
        self.num_cls = num_cls

        self.conv1 = nn.Conv2d(3, 64, (3, 3), padding=1)
        self.relu1 = self.Relu()
        self.conv2 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.relu2 = self.Relu()
        self.pool1 = nn.MaxPool2d((2, 2), 2)  # 1/2

        self.conv3 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.relu3 = self.Relu()
        self.conv4 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.relu4 = self.Relu()
        self.pool2 = nn.MaxPool2d((2, 2), 2)  # 1/4

        self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.relu5 = self.Relu()
        self.conv6 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.relu6 = self.Relu()
        self.conv7 = nn.Conv2d(256, 256, (1, 1))
        self.relu7 = self.Relu()
        self.pool3 = nn.MaxPool2d((2, 2), 2)  # 1/8

        self.conv8 = nn.Conv2d(256, 512, (3, 3), padding=1)
        self.relu8 = self.Relu()
        self.conv9 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.relu9 = self.Relu()
        self.conv10 = nn.Conv2d(512, 512, (1, 1))
        self.relu10 = self.Relu()
        self.pool4 = nn.MaxPool2d((2, 2), 2)  # 1/16

        self.conv11 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.relu11 = self.Relu()
        self.conv12 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.relu12 = self.Relu()
        self.conv13 = nn.Conv2d(512, 512, (1, 1))
        self.relu13 = self.Relu()
        self.pool5 = nn.MaxPool2d((2, 2), 2)  # 1/32

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        pool1 = self.pool1(x)

        x = self.relu3(self.conv3(pool1))
        x = self.relu4(self.conv4(x))
        pool2 = self.pool2(x)

        x = self.relu5(self.conv5(pool2))
        x = self.relu6(self.conv6(x))
        x = self.relu7(self.conv7(x))
        pool3 = self.pool3(x)

        x = self.relu8(self.conv8(pool3))
        x = self.relu9(self.conv9(x))
        x = self.relu10(self.conv10(x))
        pool4 = self.pool4(x)

        x = self.relu11(self.conv11(pool4))
        x = self.relu12(self.conv12(x))
        x = self.relu13(self.conv13(x))
        pool5 = self.pool5(x)

        bbox_reg, cls = self.RPN(pool5)

        # extract roi

        # roi pooling
        # fast rcnn head

    def ROI_extract(self):
        # roi corresponding to class
        # feature resize to 7*7
        pass

    def RPN(self, x):
        x = nn.Conv2d(512, 512, (3, 3), padding=1)(x)

        # bbox regression
        bbox_reg = nn.Conv2d(512, 4, (1, 1))

        # cls
        cls = nn.Conv2d(512, 2, (1, 1))

        return bbox_reg, cls

    def Relu(self):
        return nn.ReLU(inplace=True)
