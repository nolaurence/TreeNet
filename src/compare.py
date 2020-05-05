from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from sklearn.preprocessing import OneHotEncoder
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score, precision_score
import nni


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def load_data():
    n_file = 26
    train_ratio = 0.7
    train_indices = int(n_file * train_ratio)
    train_y = []
    x_indices = []
    x_list = []
    start = 0
    x_indices.append(start)
    print('loading raw data ...')
    for i in range(train_indices):
        x, y = load_events('P' + str(i + 1) + '.xlsx')
        # x = one_hot_preprocess(x)
        index = len(x)
        start += index

        x_indices.append(start)
        x_list.extend(x)
        train_y.extend(y)  # [resident, activity]

    train_x = one_hot_preprocess(x_list)

    # preprocess for label
    train_y = label_preprocess(train_y)

    return train_x, train_y


def load_events(filename):
    df = pd.read_excel('../RawData/' + filename)

    sensor_mapping = {'D07': 0, 'D09': 1, 'D10': 2, 'D11': 3, 'D12': 4, 'D13': 5, 'D14': 6, 'D15': 7,
                      'I04': 8, 'I06': 9,
                      'M01': 10, 'M02': 11, 'M03': 12, 'M04': 13, 'M05': 14, 'M06': 15, 'M07': 16, 'M08': 17,
                      'M09': 18, 'M10': 19, 'M11': 20, 'M12': 21, 'M13': 22, 'M14': 23, 'M15': 24, 'M16': 25,
                      'M17': 26, 'M18': 27, 'M19': 28, 'M20': 29, 'M21': 30, 'M22': 31, 'M23': 32, 'M24': 33,
                      'M25': 34, 'M26': 35, 'M51': 36}
    value_mapping = {'ON': 0, 'OFF': 1, 'ABSENT': 0, 'PRESENT': 1, 'OPEN': 1, 'CLOSE': 0}
    df['SensorID'] = df['SensorID'].map(sensor_mapping)
    df['Sensorvalue'] = df['Sensorvalue'].map(value_mapping)

    # df = df[True ^ df['Sensorvalue'].isin([0])]

    x = df['SensorID'].values
    activity = df['ActivityID'].values
    resident = df['ResidentID'].values
    y = np.hstack((np.array(resident).reshape((-1, 1)), np.array(activity).reshape((-1, 1))))
    return x, y


def one_hot_preprocess(data):
    print('preprocessing ...')

    # transform the train data into one hot data
    onehot = OneHotEncoder(sparse=False)
    data = np.array(data).reshape((-1, 1))
    data_onehot = onehot.fit_transform(data)
    return data_onehot


def label_preprocess(input):
    input = np.array(input)
    return input - 1


def load_test():
    n_file = 26
    train_ratio = 0.7
    train_indices = int(n_file * train_ratio)
    # train_y = []
    x_indices = []
    x_list = []
    start = 0
    x_indices.append(start)
    print('loading raw data ...')
    for i in range(train_indices):
        x, y = load_events('P' + str(i + 1) + '.xlsx')
        # x = one_hot_preprocess(x)
        index = len(x)
        start += index

        x_indices.append(start)
        x_list.extend(x)
        # train_y.extend(y)  # [resident, activity]

    test_y = []
    for i in range(train_indices, n_file):
        x, y = load_events('P' + str(i + 1) + '.xlsx')
        # x = one_hot_preprocess(x)
        index = len(x)
        start += index
        x_indices.append(start)
        x_list.extend(x)

        test_y.extend(y)  # [resident, activity]

    # processing on x
    x_list = one_hot_preprocess(x_list)
    train_x = []
    test_x = x_list[x_indices[train_indices]:]

    # preprocess for label
    # train_y = label_preprocess(train_y)
    test_y = label_preprocess(test_y)

    return test_x, test_y

# RECEIVED_PARAMS = nni.get_next_parameter()

Epochs = 25
# Batchsize = RECEIVED_PARAMS['batch_size']
# lr = RECEIVED_PARAMS['lr']
# l2 = RECEIVED_PARAMS['l2']
Batchsize = 64
lr=0.0002
l2 = 0.0002

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1, 37)))
model.add(Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(l2)))
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999))

X_train, Y_train = load_data()
print(X_train.shape)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
labels = Y_train[:, 0]
# print(set(labels))
Y_train = to_categorical(labels, num_classes=None)
# print(Y_train.shape)

# print('test')

history = LossHistory()
model.fit(X_train, Y_train, epochs=Epochs, batch_size=Batchsize, validation_data=None,
          verbose=2, callbacks=[history], shuffle=False)
# print(history.losses)

x_test, y_test = load_test()
y_test = y_test[:, 0]
# X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
print(x_test.shape[0])
# y_train_pred = model.predict_classes(X_train, verbose=0)
y_test_pred = model.predict_classes(x_test, verbose=0)
# print(set(y_train_pred.shape))

p = precision_score(y_test, y_test_pred, average='binary')
f1 = f1_score(y_test, y_test_pred, average='binary')
print('precision: {:.4f}'.format(p))
print('F1 : {:.4f}'.format(f1))
correct_preds = 0
for idx in range(x_test.shape[0]):
    if y_test_pred[idx] == y_test[idx]:
        correct_preds += 1
test_acc = correct_preds / x_test.shape[0]
# print('accuracy:')
# print(train_acc)
print("Trainset accuracy: {:.2f}%".format(test_acc * 100))
# for
# nni.report_final_result(train_acc)
