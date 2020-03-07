import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

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
        train_y.append(y)  # [resident, activity]

    test_y = []
    for i in range(train_indices, n_file):
        x, y = load_events('P' + str(i + 1) + '.xlsx')
        # x = one_hot_preprocess(x)
        index = len(x)
        start += index
        x_indices.append(start)
        x_list.extend(x)

        test_y.append(y)  # [resident, activity]

    # processing on x
    x_list = one_hot_preprocess(x_list)
    train_x = []
    test_x = []
    for i in range(train_indices):
        x = x_list[x_indices[i]: x_indices[i + 1]]
        train_x.append(x)
    for i in range(train_indices, n_file):
        x = x_list[x_indices[i]: x_indices[i + 1]]
        test_x.append(x)

    # preprocess for label
    train_y, test_y = one_hot_preprocess_y(train_y, test_y)

    return train_x, train_y, test_x, test_y


def load_events(filename):

    df = pd.read_excel('../RawData/' + filename)

    sensor_mapping = {'D07': 0, 'D09': 1, 'D10': 2, 'D11': 3, 'D12': 4, 'D13': 5, 'D14': 6, 'D15': 7,
                      'I04': 8, 'I06': 9,
                      'M01': 10, 'M02': 11, 'M03': 12, 'M04': 13, 'M05': 14, 'M06': 15, 'M07': 16, 'M08': 17,
                      'M09': 18, 'M10': 19, 'M11': 20, 'M12': 21, 'M13': 22, 'M14': 23, 'M15': 24, 'M16': 25,
                      'M17': 26, 'M18': 27, 'M19': 28, 'M20': 29, 'M21': 30, 'M22': 31, 'M23': 32, 'M24': 33,
                      'M25': 34, 'M26': 35, 'M51': 36}
    value_mapping = {'ON': 0, 'OFF': 1, 'ABSENT': 2, 'PRESENT': 3, 'OPEN': 4, 'CLOSE': 5}
    df['SensorID'] = df['SensorID'].map(sensor_mapping)
    df['Sensorvalue'] = df['Sensorvalue'].map(value_mapping)

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


def one_hot_preprocess_y(train, test):
    indices = []
    start = 0

    for i in range(len(train)):
        start += train[i].shape[0]
        if i == 0:
            y_list = train[i]
            indices.append(0)
        else:
            y_list = np.vstack((y_list, train[i]))
        indices.append(start)
    for i in range(len(test)):
        y_list = np.vstack((y_list, test[i]))
        start += test[i].shape[0]
        indices.append(start)

    onehot_y = label_transform(y_list)

    outputs_train = []
    outputs_test = []
    for i in range(len(train)):
        outputs_train.append(onehot_y[indices[i]: indices[i + 1]])
    for i in range(len(train), len(train) + len(test)):
        outputs_test.append(onehot_y[indices[i]: indices[i + 1]])

    return outputs_train, outputs_test


def label_transform(y):
    residents = y[:, 0].reshape((-1, 1))
    activities = y[:, 1].reshape((-1, 1))

    onehot_encoder1 = OneHotEncoder(sparse=False)
    residents = onehot_encoder1.fit_transform(residents)
    activities = onehot_encoder1.fit_transform(activities)
    output = np.hstack((residents, activities))  # [resident's onehot, activities' onehot]
    output = output.astype(np.float32)

    return output


def data_slice(x, sample_length):
    n_instances = x.shape[0]
    x = front_padding(x, sample_length)
    output = []
    for i in range(n_instances):
        instance = x[i:i+sample_length]
        output.append(instance)
    output = np.array(output, dtype=np.float32)
    # output.astype(np.float32)
    return output


def front_padding(data, sample_length):
    # padding the input data to fit the training method
    padding_matrix = np.zeros([sample_length - 1, data.shape[1]])
    added_data = np.vstack((padding_matrix, data))
    return added_data
