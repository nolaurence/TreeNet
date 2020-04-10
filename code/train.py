import torch
import time
import numpy as np
from torch import nn, optim
from network import Net
from preprocess import load_data
# from eval import evaluation


# Hyper parameters
# BATCH_SIZE = 1
SAMPLE_LENGTH = 8  # this parameters shouldn't be modified.
NUM_RESIDENT = 2
NUM_ACTIVITY = 15
LEARNING_RATE = 0.01
L2_WEIGHT = 1e-3
N_EPOCHS = 10


def train(data):
    train_x, train_y = data[0], data[1]
    test_x, test_y = data[2], data[3]

    model = Net(NUM_RESIDENT, NUM_ACTIVITY)
    # if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_WEIGHT)
    loss_fn1 = nn.CrossEntropyLoss()
    loss_fn2 = nn.CrossEntropyLoss()
    # loss_2 = nn.CrossEntropyLoss()

    # training stage
    for epoch in range(N_EPOCHS):

        running_loss1, running_loss2 = 0, 0

        for i in range(len(train_x)):
            length = len(train_x[i])
            start = 0
            print(length)
            print(train_x[i].shape)
            x = torch.from_numpy(data_slice(train_x[i], SAMPLE_LENGTH)).to(device)
            y = torch.from_numpy(train_y[i]).to(device)
            for j in range(len(x)):
                optimizer.zero_grad()

                # label_r, label_a = , y[j, 1]

                x = x[j]
                label_r = y[j, 0].unsqueeze(0)
                label_a = y[j, 1].unsqueeze(0)
                # print(label_a)

                print(label_r.shape)
                # print(x.shape)
                # for i in range(BATCH_SIZE):
                #     feed_x, feed_r, feed_a = x[i], label_r[i], label_a[i]
                resident, activity = model(x)
                # print(resident)
                resident = resident.unsqueeze(0)
                activity = activity.unsqueeze(0)

                loss1 = loss_fn1(resident, label_r)
                loss2 = loss_fn2(activity, label_a)
                loss1.backward(retain_graph=True)
                loss2.backward()
                running_loss1 += loss1.item()
                running_loss2 += loss2.item()

                optimizer.step()

                # start += BATCH_SIZE

            print('epoch {}, file {} resident loss: {:.4f}, activity loss: {:.4f}'.format(
                epoch + 1, i + 1, running_loss1 / len(x), running_loss2 / len(x)))

        # evaluation(model, SAMPLE_LENGTH, train_x, train_y)

    torch.save(model, '../weights/parameters' + time.strftime('%Y-%m-%d_%H:%M:%S',
                                                              time.localtime(time.time())) + '.pkl')


def data_slice(x, sample_length):
    n_instances = x.shape[0] - (sample_length - 1)
    # x = front_padding(x, sample_length)
    output = []
    for i in range(n_instances):
        instance = x[i:i+sample_length]
        output.append(instance)
    nice = np.array(output, dtype=np.float32)
    # output.astype(np.float32)
    return nice

data = load_data(SAMPLE_LENGTH)
train(data)
