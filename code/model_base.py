# -*- coding:utf-8 -*-

from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas

import constants


def listToArray(data_list):
    data = data_list[0]
    if data.ndim < 2:
        data = np.reshape(data, (-1, 1))
    for d in data_list[1:]:
        if d.ndim < 2:
            d = np.reshape(d, (-1, 1))
        data = np.hstack([data, d])
    return data


def flowIgnoreType(dataFLow):
    # the number of data a day is 144(block) * 4(type)* 2(in_out) * 81(station)
    # ignore type
    size = len(dataFLow)
    days = int(size/constants.DATA_A_DAY)
    dataFLow = np.reshape(dataFLow, (days, constants.DATA_A_DAY))
    temp_gen = np.zeros((days, constants.NUM_TIME_BLOCK * 2 * constants.NUM_STATIONS))

    step = constants.NUM_TIME_BLOCK * constants.NUM_TYPES
    # get the in out flow ignore type
    for i in range(days-1, -1, -1):
        for s in range(0, constants.DATA_A_DAY, step):
            temp = np.reshape(dataFLow[i, s: s+step], (constants.NUM_TYPES, constants.NUM_TIME_BLOCK))
            temp_gen[i, int(s/constants.NUM_TYPES): int((s+step)/constants.NUM_TYPES)] = np.sum(temp, axis=0)

    return np.reshape(temp_gen, (days * constants.NUM_TIME_BLOCK * 2 * constants.NUM_STATIONS))


def readTrainData(input_file):
    data = pandas.read_csv(input_file)
    # ['block', 'stationID', 'status', 'payType', 'weekDay', 'holiday', 'flow']
    # ['yesterday', 'lastWeek', 'averWeek', 'averWeekOutIn']

    data_y = data[constants.HEADER_FEAT[-1]].values
    max_label = np.max(data_y)
    train_y = data_y

    train_x = np.zeros((len(data_y), sum(constants.FEAT_LEN)), dtype=np.float32)
    encoder = OneHotEncoder()

    v = 0
    for feat in constants.HEADER_FEAT[:-1]:
        data_feat = encoder.fit_transform(np.reshape(data[feat].values, (-1, 1)))
        data_feat = data_feat.toarray()
        # data_feat = data_feat
        train_x[:, sum(constants.FEAT_LEN[:v]):sum(constants.FEAT_LEN[:v+1])] = data_feat
        v += 1
    for feat in constants.HEADER_HIS:
        data_feat = np.reshape(data[feat].values, (-1, 1))
        train_x[:, sum(constants.FEAT_LEN[:v]):sum(constants.FEAT_LEN[:v + 1])] = data_feat
        v += 1

    # train_x = listToArray(train_x)

    # train_x = pandas.DataFrame(train_x)

    return train_x, train_y

def readTrainPartial(input_file):
    data = pandas.read_csv(input_file)
    # ['block', 'stationID', 'status', 'payType', 'weekDay', 'holiday', 'flow']
    # ['yesterday', 'lastWeek', 'averWeek', 'averWeekOutIn']
    days_left = [2,3,7,8,9,10,14,15,16,17,21,22,23,24]

    data_y = data[constants.HEADER_FEAT[-1]].values
    train_y = np.reshape(data_y, (constants.DAYS, constants.DATA_A_DAY))
    train_y = train_y[days_left, :]
    train_y = train_y.flatten()

    train_x = np.zeros((len(train_y), sum(constants.FEAT_LEN)), dtype=np.float32)
    encoder = OneHotEncoder()

    v = 0
    for feat in constants.HEADER_FEAT[:-1]:
        data_feat = data[feat].values
        data_feat = np.reshape(data_feat, (constants.DAYS, constants.DATA_A_DAY))
        data_feat = data_feat[days_left, :]
        data_feat = data_feat.flatten()

        data_feat = encoder.fit_transform(np.reshape(data_feat, (-1, 1)))
        data_feat = data_feat.toarray()
        # data_feat = data_feat
        train_x[:, sum(constants.FEAT_LEN[:v]):sum(constants.FEAT_LEN[:v+1])] = data_feat
        v += 1
    for feat in constants.HEADER_HIS:
        data_feat = data[feat].values
        data_feat = np.reshape(data_feat, (constants.DAYS, constants.DATA_A_DAY))
        data_feat = data_feat[days_left, :]
        data_feat = data_feat.flatten()

        data_feat = np.reshape(data_feat, (-1, 1))
        train_x[:, sum(constants.FEAT_LEN[:v]):sum(constants.FEAT_LEN[:v + 1])] = data_feat
        v += 1

    # train_x = listToArray(train_x)

    # train_x = pandas.DataFrame(train_x)

    return train_x, train_y


def dataForTree(input_file):
    data = pandas.read_csv(input_file)
    # ['block', 'stationID', 'status', 'payType', 'weekDay', 'holiday', 'flow']
    # ['yesterday', 'lastWeek', 'averWeek', 'averWeekOutIn']

    data_y = data[constants.HEADER_FEAT[-1]].values
    max_label = np.max(data_y)
    train_y = data_y

    train_x = []

    for feat in constants.HEADER_FEAT[:-1]:
        data_feat = data[feat].values
        # data_feat = data_feat
        train_x.append(data_feat)
        print(data_feat.shape)
    for feat in constants.HEADER_HIS:
        data_feat = data[feat].values
        train_x.append(data_feat)
        print(data_feat.shape)

    train_x = listToArray(train_x)

    # train_x = pandas.DataFrame(train_x)


    # train_x, train_y = dataIgnoreType(train_x, train_y)

    return train_x, train_y


def dataIgnoreType(x, y):
    y = flowIgnoreType(y)
    n_samples, n_feats = x.shape
    print(x.shape)
    x_new = np.zeros((int(n_samples/constants.NUM_TYPES), n_feats))
    print(x_new.shape)
    yes_new = flowIgnoreType(x[:, 6])
    last_new = flowIgnoreType(x[:, 7])
    for i in range(0, n_samples, constants.NUM_TIME_BLOCK * constants.NUM_TYPES):
        x_new[int(i/ constants.NUM_TYPES):int(i/ constants.NUM_TYPES)+constants.NUM_TIME_BLOCK, :] = x[i:i+constants.NUM_TIME_BLOCK, :]
    x_new[:, 6] = yes_new
    x_new[:, 7] = last_new

    return x_new[:, [0,1,2,3,4,5,6,7,9]], y



def dataTransform(data):
    # ['block', 'stationID', 'status', 'payType', 'weekDay', 'holiday', 'flow']
    # ['yesterday', 'lastWeek', 'averWeek', 'averWeekOutIn']

    data_y = data[constants.HEADER_FEAT[-1]].values
    train_y = data_y

    train_x = []
    encoder = OneHotEncoder()

    for feat in constants.HEADER_FEAT[:-1]:
        data_feat = encoder.fit_transform(np.reshape(data[feat].values, (-1, 1)))
        data_feat = data_feat.toarray()
        # data_feat = data_feat
        train_x.append(data_feat)
    for feat in constants.HEADER_HIS:
        data_feat = data[feat].values
        train_x.append(data_feat)

    train_x = listToArray(train_x)

    # train_x = pandas.DataFrame(train_x)

    return train_x, train_y


def nextBatch(x, r):
    # ['block', 'stationID', 'status', 'payType', 'weekDay', 'holiday', 'flow']
    # ['yesterday', 'lastWeek', 'averWeek', 'averWeekOutIn']
    m, n = x.shape
    if n < sum(constants.FEAT_LEN):
        data_feat = np.zeros((m, sum(constants.FEAT_LEN)), dtype=np.float32)
        for i in range(m):
            for j in range(n-len(constants.HEADER_HIS)):
                data_feat[i, int(sum(constants.FEAT_LEN[j-1:j])+x[i, j])] = 1
                # print(int(sum(FEAT_LEN[:j])+x[i, j]))
        data_feat[:, -len(constants.HEADER_HIS):] = x[:, -len(constants.HEADER_HIS):]
    else:
        data_feat = np.array(x, dtype=np.float32)

    m, n = data_feat.shape
    data_feat = np.reshape(data_feat, (int(m/r), n * r))
    # print(np.max(data_feat))
    return data_feat


def readTestData(test_file):
    data = pandas.read_csv(test_file)
    # ['block', 'stationID', 'status', 'payType', 'weekDay', 'holiday']
    # ['yesterday', 'lastWeek', 'averWeek', 'averWeekOutIn']

    pred_x = []
    encoder = OneHotEncoder()

    for feat in constants.HEADER_FEAT[:4]:
        data_feat = encoder.fit_transform(np.reshape(data[feat].values, (-1, 1)))
        data_feat = data_feat.toarray()
        pred_x.append(data_feat)

    data_feat = np.zeros((7), dtype=np.int)
    data_feat[int(data[constants.HEADER_WEEK_DAY].values[0])] = 1
    # print(data_feat)
    pred_x.append(np.tile(np.reshape(data_feat, (1, -1)), (constants.DATA_A_DAY, 1)))

    if constants.HEADER_HOLIDAY in constants.HEADER_FEAT:
        data_feat = np.zeros((2), dtype=np.int)
        data_feat[int(data[constants.HEADER_HOLIDAY].values[0])] = 1
        pred_x.append(np.tile(np.reshape(data_feat, (1, -1)), (constants.DATA_A_DAY, 1)))

    for feat in constants.HEADER_HIS:
        data_feat = data[feat].values
        pred_x.append(data_feat)

    pred_x = listToArray(pred_x)

    return pred_x


def readTestDataForTree(test_file):
    data = pandas.read_csv(test_file)
    # ['block', 'stationID', 'status', 'payType', 'weekDay', 'holiday']
    # ['yesterday', 'lastWeek', 'averWeek', 'averWeekOutIn']

    pred_x = []

    for feat in constants.HEADER_FEAT[:-1]:
        data_feat = data[feat].values
        pred_x.append(data_feat)


    for feat in constants.HEADER_HIS:
        data_feat = data[feat].values
        pred_x.append(data_feat)

    pred_x = listToArray(pred_x)

    return pred_x