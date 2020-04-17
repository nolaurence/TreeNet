import torch
import time
from torch import nn, optim
from torch.utils import data as Data
import matplotlib.pyplot as plt
from network import Net
from preprocess import trainset


# Hyper parameters
BATCH_SIZE = 32
SAMPLE_LENGTH = 8
NUM_RESIDENT = 2
NUM_ACTIVITY = 15
LEARNING_RATE = 1e-4
L2_WEIGHT = 1e-4
N_EPOCHS = 10


def train():
    # determine the device to run the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data
    train_data = trainset(SAMPLE_LENGTH)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    modelR = Net(NUM_RESIDENT, BATCH_SIZE)
    modelR.to(device)

    # define the optimizer and loss function
    optimizer = optim.Adam(modelR.parameters(), lr=LEARNING_RATE, weight_decay=L2_WEIGHT)
    loss_fn = nn.CrossEntropyLoss()

    running_loss = []

    # training stage for resident
    for epoch in range(N_EPOCHS):

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()  # reset the optimizer
            inputs, labels = data

            # jump off the last batch (batch size is not scalable in training)
            if inputs.shape[0] < BATCH_SIZE:
                continue
            # estimate whether GPU exists
            if torch.cuda.is_available():
                outputs = modelR(inputs.cuda())
                loss = loss_fn(outputs, labels[:, 0].cuda())
            else:
                outputs = modelR(inputs)
                loss = loss_fn(outputs, labels[:, 0])

            loss.backward()  # back propagation
            optimizer.step()  # weights update
            running_loss.append(loss.item())

            print('epoch {} batch {} resident loss: {:.4f}'.format(epoch + 1, i, loss.item()))

    # plot code
    x = range(len(running_loss))
    plt.plot(x, running_loss)
    plt.show()

    # save the model for resident prediction
    torch.save(modelR, '../weights/parameters' + time.strftime('modelR-%Y-%m-%d_%H-%M-%S',
                                                               time.localtime(time.time())) + '.pkl')

    modelA = Net(NUM_ACTIVITY, BATCH_SIZE)
    modelA.to(device)
    optimizer_A = optim.Adam(modelR.parameters(), lr=LEARNING_RATE, weight_decay=L2_WEIGHT)


    running_loss1 = []
    # training stage for activity
    for epoch in range(N_EPOCHS):

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = data
            # jump off the last batch
            if inputs.shape[0] < BATCH_SIZE:
                continue

            # estimate whether GPU exists
            if torch.cuda.is_available():
                outputs = modelA(inputs.cuda())
                loss = loss_fn(outputs, labels[:, 1].cuda())
            else:
                outputs = modelA(inputs)
                loss = loss_fn(outputs, labels)

            loss.backward()  # back propagation
            optimizer_A.step()  # weights update
            running_loss1.append(loss.item())

            print('epoch {}, batch {} activity loss: {:.4f}'.format(epoch + 1, i , loss.item()))

    torch.save(modelR, '../weights/parameters' + time.strftime('modelA-%Y-%m-%d_%H-%M-%S',
                                                               time.localtime(time.time())) + '.pkl')
    print('Done.')

train()
