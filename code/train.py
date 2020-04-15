import torch
import time
import numpy as np
from torch import nn, optim
from torch.utils import data as Data
from network import Net_R, Net_A
from preprocess import load_data
# from eval import evaluation


# Hyper parameters
BATCH_SIZE = 16
SAMPLE_LENGTH = 8  # this parameters shouldn't be modified.
NUM_RESIDENT = 2
NUM_ACTIVITY = 15
LEARNING_RATE = 0.01
L2_WEIGHT = 1e-3
N_EPOCHS = 10


def train(data):
    train_x, train_y = data[0], data[1]
    test_x, test_y = data[2], data[3]

    # x = train_x[0]
    # print(x.shape)

    # model = Net(NUM_RESIDENT, NUM_ACTIVITY)
    # if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    modelR = Net_R(NUM_RESIDENT)
    modelR.to(device)

    # define the optimizer and loss function
    optimizer = optim.Adam(modelR.parameters(), lr=LEARNING_RATE, weight_decay=L2_WEIGHT)
    loss_fn = nn.CrossEntropyLoss()

    # training stage for resident
    for epoch in range(N_EPOCHS):
        # running_loss = 0
        for i in range(len(train_x)):
            running_loss = 0
            x = train_x[i]
            residents = train_y[i][:, 0]
            dataset = Data.TensorDataset(x, residents)
            train_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

            for _, data in enumerate(train_loader):
                inputs, labels = data
                outputs = modelR(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print('epoch {}, file {} resident loss: {:.4f}'.format(
                epoch + 1, i + 1, running_loss / len(x)))

    torch.save(modelR, '../weights/parameters' + time.strftime('modelR-%Y-%m-%d_%H:%M:%S',
                                                               time.localtime(time.time())) + '.pkl')

    modelA = Net_A(NUM_ACTIVITY)
    modelA.to(device)
    optimizer_A = optim.Adam(modelR.parameters(), lr=LEARNING_RATE, weight_decay=L2_WEIGHT)
    lossA = nn.CrossEntropyLoss()
    # training stage for activity
    for epoch in range(N_EPOCHS):
        for i in range(len(train_x)):
            running_loss = 0
            x = train_x[i]
            activity = train_y[i][:, 1]
            dataset = Data.TensorDataset(x, activity)
            train_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

            for _, data in enumerate(train_loader):
                inputs, labels = data
                outputs = modelA(inputs)
                loss = lossA(outputs, labels)
                loss.backward()
                optimizer_A.step()
                running_loss += loss.item()

             print('epoch {}, file {} activity loss: {:.4f}'.format(epoch + 1, i + 1, running_loss / len(x)))

    torch.save(modelR, '../weights/parameters' + time.strftime('modelA-%Y-%m-%d_%H:%M:%S',
                                                               time.localtime(time.time())) + '.pkl')


data = load_data(SAMPLE_LENGTH)
train(data)
