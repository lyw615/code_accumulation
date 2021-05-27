import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch


class ResNet(nn.Module):
    # todo 根据官方实现，里面的卷积操作都改为 bias=False，据说是为了训练的更快。
    # todo 没有把conv_block 和identity_block分开写，
    def __init__(self, num_class):
        self.num_class = num_class
        super(ResNet, self).__init__()

        # feature=self.feature_init(x)

    def forward(self, x):
        output = nn.Conv2d(3, 64, 7, stride=2)(x)
        output = nn.BatchNorm2d(64)(output)
        output = nn.ReLU(inplace=True)(output)

        # con2_x
        output = nn.MaxPool2d(3, 2)(output)
        output = self.stack_bottles(3, 64, (64, 64, 256), output)
        # con3_x
        output = self.stack_bottles(4, 256, (128, 128, 512), output, stride=2)
        # con4_x
        output = self.stack_bottles(6, 512, (256, 256, 1024), output, stride=2)
        # con5_x
        output = self.stack_bottles(3, 1024, (512, 512, 2048), output, stride=2)

        output = nn.AdaptiveAvgPool2d((1, 1))(output)

        # 把output从[batch_size,channels,height,width]的shape转成
        # [batch_size,channels*height*width]的shape，用于全连接的计算
        output = output.view(x.shape[0], -1)
        output = nn.Linear(2048, self.num_class)(output)
        output = nn.ReLU(inplace=True)(output)

        output = nn.Softmax()(output)

        return output

    def compute_same_padding(self, x, kernel_size, stride):
        # 计算same mode的padding需要设置多少pixel
        x_height, x_width = x.shape[:, -2]
        # out_h=np.ceil((x_height-kernel_size)/stride)+1
        # out_w=np.ceil((x_width-kernel_size)/stride)+1

        pad_h = (x_height - 1) * stride + kernel_size

        if (pad_h - kernel_size) % stride:
            pad_h = pad_h + (kernel_size - (pad_h - kernel_size) % stride)

        pad_w = (x_width - 1) * stride + kernel_size

        if (pad_w - kernel_size) % stride:
            pad_w = pad_w + (kernel_size - (pad_w - kernel_size) % stride)

        p_up, p_bot, = pad_h // 2, pad_h // 2
        p_r, p_l = pad_w // 2, pad_w // 2

        if pad_h % 2:
            p_up += 1
        if pad_w % 2:
            p_l += 1

        return [p_l, p_r, p_up, p_bot]

    def bottle_block(self, input_channels, conv_channels, x, stride=1):
        short_cut = x

        output = nn.Conv2d(input_channels, conv_channels[0], 1)(x)
        output = nn.BatchNorm2d(conv_channels[0])(output)
        output = nn.ReLU(inplace=True)(output)

        # padding=self.compute_same_padding(output,3,2)
        # #left ,right,top,bot
        # output=F.pad(x,padding)

        output = nn.Conv2d(conv_channels[0], conv_channels[1], 3, stride=stride, padding=1)(output)
        output = nn.BatchNorm2d(conv_channels[1])(output)
        output = nn.ReLU(inplace=True)(output)

        output = nn.Conv2d(conv_channels[1], conv_channels[2], 1)(output)
        output = nn.BatchNorm2d(conv_channels[2])(output)

        if input_channels != conv_channels[2]:
            _stride = 1
            if not x.shape[-1] == output.shape[-1]:
                _stride = 2
            # 此时output的channel数量会是input的两倍，此时input的feature size要减半后才能跟output相加
            short_cut = nn.Conv2d(input_channels, conv_channels[2], 1, stride=_stride)(x)
            short_cut = nn.BatchNorm2d(conv_channels[2])(short_cut)

            output = short_cut + output

        else:
            output = output + short_cut

        output = nn.ReLU()(output)

        return output

    def stack_bottles(self, num, input_channels, conv_channels, x, stride=1):
        ##这里需要完成feature size减半,1*1一般stride就是1 ，所以一般会在3*3卷积里设置stride=2降采样
        output = self.bottle_block(input_channels, conv_channels, x, stride)

        for _ in range(num - 1):
            output = self.bottle_block(conv_channels[-1], conv_channels, output)

        return output


if __name__ == "__main__":
    res50 = ResNet(1000)
    x = torch.randn(1, 3, 224, 224)
    output = res50.forward(x)
    target_class = torch.argmax(output)
    print(target_class)

# from torchvision.models.resnet import resnet50
# res50=resnet50()
# print(res50)
