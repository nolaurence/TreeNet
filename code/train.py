import torch
import time
from torch import nn, optim
from network import Net
from preprocess import load_data, data_slice
from eval import evaluation


# Hyper parameters
BATCH_SIZE = 32
SAMPLE_LENGTH = 8  # this parameters shouldn't be modified.
NUM_MULTI_LABEL = 15 + 2
LEARNING_RATE = 0.01
L2_WEIGHT = 1e-3
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
            length = len(train_x[i])
            start = 0
            print(length)
            while start + BATCH_SIZE < length:
                optimizer.zero_grad()
                x = data_slice(train_x[i][start: start + BATCH_SIZE], SAMPLE_LENGTH)
                y = train_y[i][start: start + BATCH_SIZE]
                x = torch.from_numpy(x).to(device)
                y = torch.from_numpy(y).to(device)
                print(x.shape)
                outputs = model(x)
                loss = loss_fn(outputs, y)
                loss.backward()
                optimizer.step()
                start += BATCH_SIZE
                running_loss += loss.item()

            print('epoch {}, file {} loss: {:.4f}'.format(epoch + 1, i + 1, running_loss / start))

            # x = torch.from_numpy(data_slice(train_x[i], SAMPLE_LENGTH)).to(device)
            # y = torch.from_numpy(train_y[i]).to(device)
            # print(train_y[i])

            # for j in range(x.shape[0]):
            #
            #     outputs = model(x[j].view(8, 37))
            #     loss = loss_fn(outputs, y[j])
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()
            #     running_loss += loss.item()
            #
            # print('epoch {}, file {} loss: {:.4f}'.format(epoch + 1, i + 1, running_loss / x.shape[0]))

        # test code
        # inputs = torch.from_numpy(data_slice(train_x[0], SAMPLE_LENGTH)).to(device)
        # labels = train_y[0]
        # label = labels[100]
        # output = model(inputs[100].view(8, 37))
        #
        # print(list(output.cpu().detach().numpy()))
        # print(label)
        ##############################
        evaluation(model, SAMPLE_LENGTH, train_x, train_y)

    torch.save(model, '../weights/parameters' + time.strftime('%Y-%m-%d_%H:%M:%S',
                                                              time.localtime(time.time())) + '.pkl')


data = load_data()
train(data)
