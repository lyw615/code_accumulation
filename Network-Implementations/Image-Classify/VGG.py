import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super(VGG, self).__init__()

        self.feature = self.feature_init()
        self.classifier = self.classifier_init()
        self._initialize_weights()

    def conv3x3(self, input_channels, out_channels, stride=1, padding=1):
        return nn.Conv2d(input_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=padding)

    def conv1x1(self, input_channels, out_channels, stride=1):
        return nn.Conv1d(input_channels, out_channels, kernel_size=(1, 1), stride=stride)

    def max_pooling(self, kernel_size, stride):
        return nn.MaxPool2d(kernel_size, stride)

    def relu(self):
        return nn.ReLU(True)

    def feature_init(self):
        feature = nn.Sequential(
            self.conv3x3(3, 64),
            self.relu(),
            self.conv3x3(64, 64),
            self.relu(),
            self.max_pooling((2, 2), 2),

            self.conv3x3(64, 128),
            self.relu(),

            self.conv3x3(128, 128),
            self.relu(),
            self.max_pooling((2, 2), 2),

            self.conv3x3(128, 256),
            self.relu(),

            self.conv3x3(256, 256),
            self.relu(),

            self.conv3x3(256, 256),
            self.relu(),
            self.max_pooling((2, 2), 2),

            self.conv3x3(256, 512),
            self.relu(),

            self.conv3x3(512, 512),
            self.relu(),

            self.conv3x3(512, 512),
            self.relu(),
            self.max_pooling((2, 2), 2),

            self.conv3x3(512, 512),
            self.relu(),

            self.conv3x3(512, 512),
            self.relu(),

            self.conv3x3(512, 512),
            self.relu(),
            self.max_pooling((2, 2), 2),

        )

        return feature

    def classifier_init(self):
        classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            self.relu(),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            self.relu(),
            nn.Dropout(0.5),

            nn.Linear(4096, self.num_classes),
            self.relu(),
            nn.Softmax()
        )

        return classifier

    def forward(self, x):
        feature = self.feature(x)
        # view equal to resize, view(-1) get one-d, x.size(0) mean batch size
        feature = feature.view(x.size(0), -1)

        output = self.classifier(feature)

        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward_by_line(self, x):
        "x.shape=[batch_size,h,w,channels ] "
        output = self.conv3x3(3, 64)(x)
        output = self.relu()(output)

        output = self.conv3x3(64, 64)(output)
        output = self.relu()(output)
        output = self.max_pooling((2, 2), 2)(output)

        output = self.conv3x3(64, 128)(output)
        output = self.relu()(output)

        output = self.conv3x3(128, 128)(output)
        output = self.relu()(output)
        output = self.max_pooling((2, 2), 2)(output)

        output = self.conv3x3(128, 256)(output)
        output = self.relu()(output)

        output = self.conv3x3(256, 256)(output)
        output = self.relu()(output)

        output = self.conv3x3(256, 256)(output)
        output = self.relu()(output)
        output = self.max_pooling((2, 2), 2)(output)

        output = self.conv3x3(256, 512)(output)
        output = self.relu()(output)

        output = self.conv3x3(512, 512)(output)
        output = self.relu()(output)

        output = self.conv3x3(512, 512)(output)
        output = self.relu()(output)
        output = self.max_pooling((2, 2), 2)(output)

        output = self.conv3x3(512, 512)(output)
        output = self.relu()(output)

        output = self.conv3x3(512, 512)(output)
        output = self.relu()(output)

        output = self.conv3x3(512, 512)(output)
        output = self.relu()(output)
        output = self.max_pooling((2, 2), 2)(output)

        output = torch.flatten(output)
        # fully connected layers
        output = nn.Linear(7 * 7 * 512, 4096)(output)
        output = self.relu()(output)
        output = nn.Dropout(0.5)(output)

        output = nn.Linear(4096, 4096)(output)
        output = self.relu()(output)
        output = nn.Dropout(0.5)(output)

        output = nn.Linear(4096, self.num_classes)(output)
        output = self.relu()(output)

        output = nn.Softmax()(output)

        return output


vgg16 = VGG(3)
print(vgg16)
