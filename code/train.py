import torch
import time
from torch import nn, optim
from torch.utils import data as Data
from network import Net_R, Net_A
from preprocess import trainset, testset
# from eval import evaluation


# Hyper parameters
BATCH_SIZE = 32
SAMPLE_LENGTH = 8  # this parameters shouldn't be modified.
NUM_RESIDENT = 2
NUM_ACTIVITY = 15
LEARNING_RATE = 0.01
L2_WEIGHT = 1e-3
N_EPOCHS = 10


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data = trainset(SAMPLE_LENGTH)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # x = train_x[0]
    # print(x.shape)

    # model = Net(NUM_RESIDENT, NUM_ACTIVITY)
    # if torch.cuda.is_available():

    modelR = Net_R(NUM_RESIDENT)
    modelR.to(device)


    # define the optimizer and loss function
    optimizer = optim.Adam(modelR.parameters(), lr=LEARNING_RATE, weight_decay=L2_WEIGHT)
    loss_fn = nn.CrossEntropyLoss()

    # training stage for resident
    for epoch in range(N_EPOCHS):
        running_loss = 0
        # for i in range(len(train_x)):
        #     running_loss = 0
        #     x = torch.from_numpy(train_x[i])
        #     residents = torch.from_numpy(train_y[i][:, 0])
        #     print(x.shape, residents.shape)
        #     dataset = Data.TensorDataset(x, residents)
        #     train_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

        for i, data in enumerate(train_loader):
            inputs, labels = data
            # inputs.to(device)
            # labels.to(device)
            outputs = modelR(inputs.cuda())
            loss = loss_fn(outputs, labels[:, 0].cuda())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            print('epoch {} batch{} resident loss: {:.4f}'.format(epoch + 1, i,running_loss / BATCH_SIZE))

    torch.save(modelR, '../weights/parameters' + time.strftime('modelR-%Y-%m-%d_%H:%M:%S',
                                                               time.localtime(time.time())) + '.pkl')

    modelA = Net_A(NUM_ACTIVITY)
    modelA.to(device)
    optimizer_A = optim.Adam(modelR.parameters(), lr=LEARNING_RATE, weight_decay=L2_WEIGHT)
    lossA = nn.CrossEntropyLoss()
    # training stage for activity
    for epoch in range(N_EPOCHS):
        running_loss = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            outputs = modelA(inputs)
            loss = lossA(outputs, labels[:, 1])
            loss.backward()
            optimizer_A.step()
            running_loss += loss.item()

            print('epoch {}, batch {} activity loss: {:.4f}'.format(epoch + 1, i , running_loss / BATCH_SIZE))

    torch.save(modelR, '../weights/parameters' + time.strftime('modelA-%Y-%m-%d_%H:%M:%S',
                                                               time.localtime(time.time())) + '.pkl')
    print('Done.')

# data = load_data(SAMPLE_LENGTH)
train()
