import torch
import time
from torch import nn, optim
import numpy as np
from network import Net
from data_utils import load_data


# Hyper parameters
SAMPLE_LENGTH = 8  # this parameters shouldn't be modified.
NUM_MULTI_LABEL = 15 + 2
LEARNING_RATE = 1e-3
L2_WEIGHT = 1e-2
N_EPOCHS = 10


def train(data):
    train_x, train_y = data[0], data[1]
    test_x, test_y = data[2], data[3]

    model = Net(NUM_MULTI_LABEL)
    # if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_WEIGHT)
    loss_fn = nn.BCELoss()

    # training stage
    for epoch in range(N_EPOCHS):

        running_loss = 0

        for i in range(len(train_x)):

            x = torch.from_numpy(data_slice(train_x[i])).to(device)
            y = torch.from_numpy(train_y[i]).to(device)
            # print(train_y[i])

            for j in range(x.shape[0]):
                optimizer.zero_grad()
                outputs = model(x[j].view(8, 37))
                loss = loss_fn(outputs, y[j])
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print('epoch {}, file {} loss: {:.4f}'.format(epoch + 1, i + 1, running_loss / x.shape[0]))

    checkpoint = {'model': model,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()}

    torch.save(checkpoint, '../weights/parameters' + time.strftime('%Y-%m-%d_%H:%M:%S',
                                                                   time.localtime(time.time())) + '.pkl')


def front_padding(data, sample_length):
    # padding the input data to fit the training method
    padding_matrix = np.zeros([sample_length - 1, data.shape[1]])
    added_data = np.vstack((padding_matrix, data))
    return added_data


def data_slice(x):
    n_instances = x.shape[0]
    x = front_padding(x, SAMPLE_LENGTH)
    output = []
    for i in range(n_instances):
        instance = x[i:i+SAMPLE_LENGTH]
        output.append(instance)
    output = np.array(output, dtype=np.float32)
    # output.astype(np.float32)
    return output


data = load_data()
train(data)
