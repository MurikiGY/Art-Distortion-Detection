from torch.nn import Module, Conv2d, Linear, MaxPool2d, ReLU, LogSoftmax
from torch import flatten

class Neural_Network(Module):
    def __init__(self, numChannels, classes):
        super(Neural_Network, self).__init__()

        ch1 = 20
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=ch1, kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        ch2 = 50
        self.conv2 = Conv2d(in_channels=ch1, out_channels=ch2, kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        ch3 = 50
        self.conv3 = Conv2d(in_channels=ch2, out_channels=ch3, kernel_size=(5, 5))
        self.relu3 = ReLU()
        self.maxpool3 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        ch4 = 50
        self.conv4 = Conv2d(in_channels=ch3, out_channels=ch4, kernel_size=(5, 5))
        self.relu4 = ReLU()
        self.maxpool4 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        fc_ch1 = 1000
        self.fc1 = Linear(in_features=7200, out_features=fc_ch1)
        self.relu5 = ReLU()

        self.fc2 = Linear(in_features=fc_ch1, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu5(x)

        x = self.fc2(x)
        out = self.logSoftmax(x)

        return out
