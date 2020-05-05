import torch
import time
from torch import nn, optim
from torch.utils import data as Data
# import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from network3 import Net
from preprocess import trainset, testset
from eval import evaluation


# Hyper parameters
BATCH_SIZE = 256
SAMPLE_LENGTH = 8  # 这个参数不能改
NUM_RESIDENT = 2
NUM_ACTIVITY = 15
LEARNING_RATE = 0.0035961683699580957
L2_WEIGHT = 0.0001328492781537454
N_EPOCHS = 10

LR = 0.005
L2 = 0.0001
BS = 256


def train():
    # determine the device to run the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    use_gpu = True

    # tensorboard initialization
    writer = SummaryWriter('../runs')

    # load data
    train_data = trainset(SAMPLE_LENGTH)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    modelR = Net(NUM_RESIDENT, BATCH_SIZE)
    modelR.to(device)

    # define the optimizer and loss function
    optimizer = optim.Adam(modelR.parameters(), lr=LEARNING_RATE, weight_decay=L2_WEIGHT)
    loss_fn = nn.CrossEntropyLoss()

    # training stage for resident
    idx = 0
    for epoch in range(N_EPOCHS):

        for i, data in enumerate(train_loader):
            idx += 1
            optimizer.zero_grad()  # reset the optimizer
            inputs, labels = data
            if epoch == 0 and i == 0:
                if torch.cuda.is_available() and use_gpu:
                    writer.add_graph(modelR, input_to_model=inputs.cuda(), verbose=False)
                else:
                    writer.add_graph(modelR, input_to_model=inputs, verbose=False)

            # jump off the last batch (batch size is not scalable in training)
            if inputs.shape[0] < BATCH_SIZE:
                continue
            # estimate whether GPU exists
            if torch.cuda.is_available() and use_gpu:
                outputs = modelR(inputs.cuda())
                loss = loss_fn(outputs, labels[:, 0].cuda())
            else:
                outputs = modelR(inputs)
                loss = loss_fn(outputs, labels[:, 0])

            loss.backward()  # back propagation
            optimizer.step()  # weights update

            # loss visualization
            writer.add_scalar('resident loss', loss.item(), global_step=idx)

            print('epoch {} batch {} resident loss: {:.4f}'.format(epoch + 1, i, loss.item()))


    # save the model for resident prediction
    torch.save(modelR, '../weights/parameters' + time.strftime('modelR-%Y-%m-%d_%H-%M-%S',
                                                               time.localtime(time.time())) + '.pkl')

    modelA = Net(NUM_ACTIVITY, BS)
    modelA.to(device)
    optimizer_A = optim.Adam(modelA.parameters(), lr=LR, weight_decay=L2)
    train_loader2 = Data.DataLoader(dataset=train_data, batch_size=BS, shuffle=False, num_workers=0)

    # nni code
    # best_loss = 0
    # training stage for activity
    print('training activity ...')
    idx = 0
    for epoch in range(N_EPOCHS):

        # nni code
        # running_loss = 0
        for i, data in enumerate(train_loader2):
            idx += 1
            optimizer_A.zero_grad()
            inputs, labels = data
            # jump off the last batch
            if inputs.shape[0] < BS:
                continue

            # estimate whether GPU exists
            if torch.cuda.is_available() and use_gpu:
                outputs = modelA(inputs.cuda())
                loss = loss_fn(outputs, labels[:, 1].cuda())
            else:
                outputs = modelA(inputs)
                loss = loss_fn(outputs, labels)

            loss.backward()  # back propagation
            optimizer_A.step()  # weights update

            # nni code
            # running_loss += loss.item()
            # idx += 1

            # loss visualization
            writer.add_scalar('activity loss', loss.item(), global_step=idx)

            print('epoch {}, batch {} activity loss: {:.4f}'.format(epoch + 1, i, loss.item()))

    torch.save(modelR, '../weights/parameters' + time.strftime('modelA-%Y-%m-%d_%H-%M-%S',
                                                               time.localtime(time.time())) + '.pkl')
    print('Done.')

    # evaluation
    test_data = testset(SAMPLE_LENGTH)
    test_loader1 = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader2 = Data.DataLoader(dataset=test_data, batch_size=BS, shuffle=False, num_workers=0)
    evaluation(modelR, modelA, train_loader, train_loader2)


train()
