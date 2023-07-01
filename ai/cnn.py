from torch.nn import Module, Conv2d, Linear, MaxPool2d, ReLU, LogSoftmax, ModuleList
from torch import flatten

class Neural_Network(Module):
    def __init__(self, numChannels, classes):
        super(Neural_Network, self).__init__()

        conv_layer = [20, 40, 60, 80]

        self.conv = ModuleList()
        self.conv_relu = ModuleList()
        self.maxpool = ModuleList()

        for i in range(len(conv_layer)):
            if i == 0:
                aux = Conv2d(in_channels=numChannels, out_channels=conv_layer[i], kernel_size=(5, 5))
            else:
                aux = Conv2d(in_channels=conv_layer[i-1], out_channels=conv_layer[i], kernel_size=(5, 5))

            self.conv.append(aux)
            self.conv_relu.append(ReLU())
            self.maxpool.append(MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))


        in_feat = 11520
        fc_layer = [2000, 1000, 500]

        self.fc = ModuleList()
        self.fc_relu = ModuleList()

        for i in range(len(fc_layer)):
            if i == 0:
                aux = Linear(in_features=in_feat, out_features=fc_layer[i])
            else:
                aux = Linear(in_features=fc_layer[i-1], out_features=fc_layer[i])

            self.fc.append(aux)
            self.fc_relu.append(ReLU())

        self.final_layer = Linear(in_features=fc_layer[-1], out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):

        for i in range(len(self.conv)):
            x = self.conv[i](x)
            x = self.conv_relu[i](x)
            x = self.maxpool[i](x)

        x = flatten(x, 1)

        for i in range(len(self.fc)):
            x = self.fc[i](x)
            x = self.fc_relu[i](x)

        x = self.final_layer(x)
        out = self.logSoftmax(x)

        return out
