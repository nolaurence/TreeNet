from torch import nn
from torch.nn.functional import relu


# computation graph
class Net(nn.Module):
    def __init__(self, n_label):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # input is 37 dimenssion
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc = nn.Linear(in_features=64 * 37, out_features=n_label)
        # self.relu = nn.ReLU
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # [8, 37]
        # convolution 1 [37-3+1=35, 16]
        # layer1_1 = relu(self.conv1(x[0].view(-1, 1, 37)) + self.conv1(x[1]))
        # print(x[7])
        layer1_1 = relu(self.conv1(x[0].view(-1, 1, 37)) + self.conv1(x[1].view(-1, 1, 37)))
        layer1_2 = relu(self.conv1(x[2].view(-1, 1, 37)) + self.conv1(x[3].view(-1, 1, 37)))
        layer1_3 = relu(self.conv1(x[4].view(-1, 1, 37)) + self.conv1(x[5].view(-1, 1, 37)))
        layer1_4 = relu(self.conv1(x[6].view(-1, 1, 37)) + self.conv1(x[7].view(-1, 1, 37)))

        # convolution 2 [35-3+1=33, 32]
        layer2_1 = relu(self.conv2(layer1_1) + self.conv2(layer1_2))
        layer2_2 = relu(self.conv2(layer1_3) + self.conv2(layer1_4))

        # convolution 3 [33-3+1=31, 64]
        y = relu(self.conv3(layer2_1) + self.conv3(layer2_2))
        y = y.view(-1)

        y = self.softmax(self.fc(y))
        return y
