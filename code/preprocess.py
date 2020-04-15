import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def load_data(sample_len):
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
        x = front_padding(x, sample_len)
        x = data_slice(x, sample_len)
        train_x.append(x)
    for i in range(train_indices, n_file):
        x = x_list[x_indices[i]: x_indices[i + 1]]
        x = front_padding(x, sample_len)
        x = data_slice(x, sample_len)
        test_x.append(x)
    # print(train_y.shape)

    # preprocess for label
    train_y = label_preprocess(train_y)
    test_y = label_preprocess(test_y)

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


def front_padding(data, sample_length):
    # padding the input data to fit the training method
    padding_matrix = np.zeros([sample_length - 1, data.shape[1]])
    added_data = np.vstack((padding_matrix, data))
    return added_data


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

def label_preprocess(input):
    input = np.array(input)
    return input - 1
