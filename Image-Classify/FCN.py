import torchvision
import torch
import torch.nn as nn


class FCN_8S(nn.Module):
    def __init__(self, num_cls):
        super(FCN_8S, self).__init__()
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

        self.conv14 = nn.Conv2d(512, 4096, (7, 7), padding=3)
        self.relu14 = self.Relu()
        self.drop1 = nn.Dropout2d()

        self.conv15 = nn.Conv2d(4096, 4096, (1, 1))
        self.relu15 = self.Relu()
        self.drop2 = nn.Dropout2d()

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

        pred_p3 = nn.Conv2d(256, self.num_cls, (1, 1))

        pred_p4 = nn.Conv2d(512, self.num_cls, (1, 1))

        conv7 = self.relu14(self.conv14(pool5))
        conv7 = self.drop1(conv7)

        conv7 = self.relu15(self.conv15(conv7))
        conv7 = self.drop2(conv7)

        conv7 = nn.Conv2d(4096, self.num_cls, (1, 1))(conv7)

        d_conv7 = nn.ConvTranspose2d(self.num_cls, self.num_cls, (4, 4), 2)(conv7)
        # d_conv7=nn.ReLU(d_conv7)
        fus_16 = pred_p4 + d_conv7

        d_fus_16 = nn.ConvTranspose2d(self.num_cls, self.num_cls, (2, 2), 2)(fus_16)
        # d_fus_16=nn.ReLU(d_fus_16)

        fus8 = pred_p3 + d_fus_16

    def Relu(self):
        return nn.ReLU(inplace=True)
