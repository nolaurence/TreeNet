from torch import nn
from torch.nn.functional import relu

# computation graph
class Net_R(nn.Module):
    def __init__(self, n_resident):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # input is 37 dimenssion
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=1)  # output is 36 dims
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # output is 18 dims
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # output is 9 dims
        self.fc1 = nn.Linear(in_features=64 * 9, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=n_resident)
        # self.fc_a = nn.Linear(in_features=256, out_features=n_activity)
        # self.relu = nn.ReLU
        # self.sigmoid = nn.Sigmoid()
        self.conv1_1 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # [8, 37]
        # convolution 1 [37-3+1=35, 16]
        layer1_1 = self.residual_1(relu(self.conv1(x[0].view(-1, 1, 37)))) + self.residual_1(relu(self.conv1(x[1].view(-1, 1, 37))))
        layer1_2 = self.residual_1(relu(self.conv1(x[2].view(-1, 1, 37)))) + self.residual_1(relu(self.conv1(x[3].view(-1, 1, 37))))
        layer1_3 = self.residual_1(relu(self.conv1(x[4].view(-1, 1, 37)))) + self.residual_1(relu(self.conv1(x[5].view(-1, 1, 37))))
        layer1_4 = self.residual_1(relu(self.conv1(x[6].view(-1, 1, 37)))) + self.residual_1(relu(self.conv1(x[7].view(-1, 1, 37))))

        # pooling layer
        layer1_1 = self.maxpool1(layer1_1)
        layer1_2 = self.maxpool1(layer1_2)
        layer1_3 = self.maxpool1(layer1_3)
        layer1_4 = self.maxpool1(layer1_4)

        # convolution 2 [35-3+1=33, 32]
        layer2_1 = self.residual_2(relu(self.conv2(layer1_1))) + self.residual_2(relu(self.conv2(layer1_2)))
        layer2_2 = self.residual_2(relu(self.conv2(layer1_3))) + self.residual_2(relu(self.conv2(layer1_4)))

        # pooling layer
        layer2_1 = self.maxpool2(layer2_1)
        layer2_2 = self.maxpool2(layer2_2)

        # convolution 3 [33-3+1=31, 64]
        y = self.residual_3(relu(self.conv3(layer2_1))) + self.residual_3(relu(self.conv3(layer2_2)))
        print(y.shape)
        y = self.maxpool3(y)

        y = y.view(-1)
        # print(y.shape)

        resident = relu(self.fc1(y))
        # activity = relu(self.fc1(y))
        resident = self.fc2(resident)
        # activity = self.fc_a(activity)

        return resident

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


class Net_A(nn.Module):
    def __init__(self, n_activity):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # input is 37 dimenssion
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=1)  # output is 36 dims
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # output is 18 dims
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # output is 9 dims
        self.fc1 = nn.Linear(in_features=64 * 9, out_features=256)
        # self.fc2 = nn.Linear(in_features=256, out_features=n_resident)
        self.fc2 = nn.Linear(in_features=256, out_features=n_activity)
        # self.relu = nn.ReLU
        # self.sigmoid = nn.Sigmoid()
        self.conv1_1 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # [8, 37]
        # convolution 1 [37-3+1=35, 16]
        layer1_1 = self.residual_1(relu(self.conv1(x[0].view(-1, 1, 37)))) + self.residual_1(
            relu(self.conv1(x[1].view(-1, 1, 37))))
        layer1_2 = self.residual_1(relu(self.conv1(x[2].view(-1, 1, 37)))) + self.residual_1(
            relu(self.conv1(x[3].view(-1, 1, 37))))
        layer1_3 = self.residual_1(relu(self.conv1(x[4].view(-1, 1, 37)))) + self.residual_1(
            relu(self.conv1(x[5].view(-1, 1, 37))))
        layer1_4 = self.residual_1(relu(self.conv1(x[6].view(-1, 1, 37)))) + self.residual_1(
            relu(self.conv1(x[7].view(-1, 1, 37))))

        # pooling layer
        layer1_1 = self.maxpool1(layer1_1)
        layer1_2 = self.maxpool1(layer1_2)
        layer1_3 = self.maxpool1(layer1_3)
        layer1_4 = self.maxpool1(layer1_4)

        # convolution 2 [35-3+1=33, 32]
        layer2_1 = self.residual_2(relu(self.conv2(layer1_1))) + self.residual_2(relu(self.conv2(layer1_2)))
        layer2_2 = self.residual_2(relu(self.conv2(layer1_3))) + self.residual_2(relu(self.conv2(layer1_4)))

        # pooling layer
        layer2_1 = self.maxpool2(layer2_1)
        layer2_2 = self.maxpool2(layer2_2)

        # convolution 3 [33-3+1=31, 64]
        y = self.residual_3(relu(self.conv3(layer2_1))) + self.residual_3(relu(self.conv3(layer2_2)))
        print(y.shape)
        y = self.maxpool3(y)

        y = y.view(-1)
        # print(y.shape)

        activity = relu(self.fc1(y))
        # activity = relu(self.fc1(y))
        activity = self.fc2(activity)
        # activity = self.fc_a(activity)

        return activity

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
