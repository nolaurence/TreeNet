from torch import nn
from torch.nn.functional import relu

# computation graph
class Net(nn.Module):
    def __init__(self, n_resident, batch_size):
        super().__init__()
        self.batch_size = batch_size

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.conv1_1 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        # self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=1)
        # self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=64 * 37, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=n_resident)


    def forward(self, x):
        # [8, 37]
        # convolution 1 [N, 16, 37]
        # print(x.shape)
        layer1_1 = self.residual_1(relu(self.conv1(x[:, 0].view(-1, 1, 37)))) + self.residual_1(relu(self.conv1(x[:, 1].view(-1, 1, 37))))
        layer1_2 = self.residual_1(relu(self.conv1(x[:, 2].view(-1, 1, 37)))) + self.residual_1(relu(self.conv1(x[:, 3].view(-1, 1, 37))))
        layer1_3 = self.residual_1(relu(self.conv1(x[:, 4].view(-1, 1, 37)))) + self.residual_1(relu(self.conv1(x[:, 5].view(-1, 1, 37))))
        layer1_4 = self.residual_1(relu(self.conv1(x[:, 6].view(-1, 1, 37)))) + self.residual_1(relu(self.conv1(x[:, 7].view(-1, 1, 37))))


        # pooling layer
        # layer1_1 = self.maxpool1(layer1_1)
        # layer1_2 = self.maxpool1(layer1_2)
        # layer1_3 = self.maxpool1(layer1_3)
        # layer1_4 = self.maxpool1(layer1_4)


        # convolution 2 [N, 32, 36]
        layer2_1 = self.residual_2(relu(self.conv2(layer1_1))) + self.residual_2(relu(self.conv2(layer1_2)))
        layer2_2 = self.residual_2(relu(self.conv2(layer1_3))) + self.residual_2(relu(self.conv2(layer1_4)))
        # print(layer2_1.shape)

        # pooling layer
        # layer2_1 = self.maxpool2(layer2_1)
        # layer2_2 = self.maxpool2(layer2_2)
        # print(layer2_1.shape)

        # convolution 3 [33-3+1=31, 64]
        y = self.residual_3(relu(self.conv3(layer2_1))) + self.residual_3(relu(self.conv3(layer2_2)))
        # y = self.maxpool3(y)
        # print(y.shape)

        y = y.view(y.shape[0], -1)

        y = relu(self.fc1(y))
        y = self.fc2(y)

        return y

    def residual_1(self, x):
        output = self.conv1_1(x)
        output = relu(output)
        output = self.conv1_1(output)
        output = relu(output)
        output += x
        return output

    def residual_2(self, x):
        output = self.conv2_1(x)
        output = relu(output)
        output = self.conv2_1(output)
        output = relu(output)
        output += x
        return output

    def residual_3(self, x):
        output = self.conv3_1(x)
        output = relu(output)
        output = self.conv3_1(output)
        output = relu(output)
        output += x
        return output
