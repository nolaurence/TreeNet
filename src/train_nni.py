# nni trial
import torch
import time
import nni
from torch import nn, optim
from torch.utils import data as Data
from torch.utils.tensorboard import SummaryWriter
from network import Net
from preprocess import trainset

# nni code
RECEIVED_PARAMS = nni.get_next_parameter()

# Hyper parameters
# BATCH_SIZE = 128
BATCH_SIZE = RECEIVED_PARAMS['batch_size']
SAMPLE_LENGTH = 8  # 这个参数不能改
NUM_RESIDENT = 2
NUM_ACTIVITY = 15
# LEARNING_RATE = 1e-6
LEARNING_RATE = RECEIVED_PARAMS['lr']
L2_WEIGHT = RECEIVED_PARAMS['l2']
N_EPOCHS = 10

LR = 1e-5
L2 = 1e-3


def train():
    # determine the device to run the model
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    use_gpu = False

    # initialize tensorboard
    writer = SummaryWriter('../runs')

    # load data
    train_data = trainset(SAMPLE_LENGTH)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    modelR = Net(NUM_RESIDENT, BATCH_SIZE)
    modelR.to(device)

    # define the optimizer and loss function
    optimizer = optim.Adam(modelR.parameters(), lr=LEARNING_RATE, weight_decay=L2_WEIGHT)
    loss_fn = nn.CrossEntropyLoss()
    best_loss = 0
    # training stage for resident
    for epoch in range(N_EPOCHS):
        # nni code
        running_loss = 0
        batches = 0

        for i, data in enumerate(train_loader):

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

            # nni code
            batches += 1
            running_loss += loss.item()

            # loss visualization
            writer.add_scalar('resident loss for epoch ' + str(epoch + 1), loss.item(), global_step=i + 1)

            print('epoch {} batch {} resident loss: {:.4f}'.format(epoch + 1, i, loss.item()))

        # nni code
        loss_per_epoch = running_loss / batches
        if epoch == 0:
            best_loss = loss_per_epoch
        elif best_loss >= loss_per_epoch:
            best_loss = loss_per_epoch
        nni.report_intermediate_result(loss_per_epoch)
    nni.report_final_result(best_loss)

    # save the model for resident prediction
    torch.save(modelR, '../weights/parameters' + time.strftime('modelR-%Y-%m-%d_%H-%M-%S',
                                                               time.localtime(time.time())) + '.pkl')

    # modelA = Net(NUM_ACTIVITY, BATCH_SIZE)
    # modelA.to(device)
    # optimizer_A = optim.Adam(modelR.parameters(), lr=LR, weight_decay=L2)
    #
    # # training stage for activity
    # for epoch in range(N_EPOCHS):
    #
    #     for i, data in enumerate(train_loader):
    #         optimizer.zero_grad()
    #         inputs, labels = data
    #         # jump off the last batch
    #         if inputs.shape[0] < BATCH_SIZE:
    #             continue
    #
    #         # estimate whether GPU exists
    #         if torch.cuda.is_available():
    #             outputs = modelA(inputs.cuda())
    #             loss = loss_fn(outputs, labels[:, 1].cuda())
    #         else:
    #             outputs = modelA(inputs)
    #             loss = loss_fn(outputs, labels)
    #
    #         loss.backward()  # back propagation
    #         optimizer_A.step()  # weights update
    #
    #         # loss visualization
    #         writer.add_scalar('activity loss for epoch ' + str(epoch + 1), loss.item(), global_step=i + 1)
    #
    #         print('epoch {}, batch {} activity loss: {:.4f}'.format(epoch + 1, i , loss.item()))
    #
    # torch.save(modelR, '../weights/parameters' + time.strftime('modelA-%Y-%m-%d_%H-%M-%S',
    #                                                            time.localtime(time.time())) + '.pkl')
    print('Done.')

train()
