import torch
import time
from torch import nn, optim
from torch.utils import data as Data
import matplotlib.pyplot as plt
from network import Net
from preprocess import trainset, testset
# from eval import evaluation


# Hyper parameters
BATCH_SIZE = 32
SAMPLE_LENGTH = 8  # this parameters shouldn't be modified.
NUM_RESIDENT = 2
NUM_ACTIVITY = 15
LEARNING_RATE = 1e-4
L2_WEIGHT = 1e-4
N_EPOCHS = 10


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data = trainset(SAMPLE_LENGTH)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # x = train_x[0]
    # print(x.shape)

    # model = Net(NUM_RESIDENT, NUM_ACTIVITY)
    # if torch.cuda.is_available():

    modelR = Net(NUM_RESIDENT, BATCH_SIZE)
    modelR.to(device)


    # define the optimizer and loss function
    optimizer = optim.Adam(modelR.parameters(), lr=LEARNING_RATE, weight_decay=L2_WEIGHT)
    loss_fn = nn.CrossEntropyLoss()

    # training stage for resident
    for epoch in range(N_EPOCHS):
        running_loss = []
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = data
            if inputs.shape[0] < BATCH_SIZE:
                continue
            outputs = modelR(inputs.cuda())
            # print(outputs)
            # print(labels[:, 0])
            loss = loss_fn(outputs, labels[:, 0].cuda())
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

            print('\r', 'epoch {} batch {} resident loss: {:.4f}'.format(epoch + 1, i, loss.item()), end='')

    # plot code
    x = range(len(running_loss))
    plt.plot(x, running_loss)
    plt.show()
    torch.save(modelR, '../weights/parameters' + time.strftime('modelR-%Y-%m-%d_%H-%M-%S',
                                                               time.localtime(time.time())) + '.pkl')

    modelA = Net(NUM_ACTIVITY, BATCH_SIZE)
    modelA.to(device)
    optimizer_A = optim.Adam(modelR.parameters(), lr=LEARNING_RATE, weight_decay=L2_WEIGHT)
    lossA = nn.CrossEntropyLoss()
    # training stage for activity
    for epoch in range(N_EPOCHS):
        running_loss1 = []
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = data
            if inputs.shape[0] < BATCH_SIZE:
                continue
            outputs = modelA(inputs)
            loss = lossA(outputs, labels[:, 1])
            loss.backward()
            optimizer_A.step()
            running_loss1.append(loss.item())

            print('epoch {}, batch {} activity loss: {:.4f}'.format(epoch + 1, i , loss.item()))

    torch.save(modelR, '../weights/parameters' + time.strftime('modelA-%Y-%m-%d_%H-%M-%S',
                                                               time.localtime(time.time())) + '.pkl')
    print('Done.')

# data = load_data(SAMPLE_LENGTH)
train()
