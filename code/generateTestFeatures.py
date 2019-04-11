# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from constants import *
from analyzeData import getWeekDay

# the number of data a day is 144(block) * 4(type)* 2(in_out) * 81(station)
data_a_day = DATA_A_DAY
LOG_NAME = 'genTestFeatures'

# !!! python actions are performed along line
mode = PRED

def generateXTicks(x):
    ticks = []
    for minu in x:
        hour = str(int(np.floor(minu / 60))).zfill(2)
        minutes = str(int(minu % 60)).zfill(2)
        tick = '%s:%s' % (hour, minutes)
        ticks.append(tick)
    return ticks


def genFlowAverWeekOutIn(dataFLow):
    # the number of data a day is 144(block) * 4(type)* 2(in_out) * 81(station)
    # ignore type
    days = dataFLow.shape[0]
    temp_gen = np.zeros(dataFLow.shape)

    step = NUM_TIME_BLOCK * NUM_TYPES
    # get the in out flow ignore type
    for i in range(days-1, -1, -1):
        for s in range(0, data_a_day, step):
            temp = np.reshape(dataFLow[i, s: s+step], (NUM_TYPES, NUM_TIME_BLOCK))
            temp_gen[i, s: s+step] = np.tile(np.sum(temp, axis=0), (NUM_TYPES))

    # construct as historical feature by using pass weeks' data
    flow_gen = np.mean(temp_gen, axis=0)

    return flow_gen


def genHisData():


    print(LOG_NAME + ': reading data ......')
    data_features = pd.read_csv(DATA_PATH_FEAT+'features.csv')
    data_yesterday = pd.read_csv(DATA_PATH_TEST+'features_yesterday.csv')

    feats = {}

    # ['block', 'stationID', 'status', 'payType', 'weekDay', 'holiday'
    print(LOG_NAME + 'generating basic features ......')
    feats[HEADER_FEAT[0]] = data_yesterday[HEADER_FEAT[0]].values
    feats[HEADER_FEAT[1]] = data_yesterday[HEADER_FEAT[1]].values
    feats[HEADER_FEAT[2]] = data_yesterday[HEADER_FEAT[2]].values
    feats[HEADER_FEAT[3]] = data_yesterday[HEADER_FEAT[3]].values
    week_day = getWeekDay(DATE_TEST)
    feats[HEADER_FEAT[4]] = np.ones((DATA_A_DAY)) * week_day
    holiday = 0
    if DATA_PATH_TEST in HOLIDAY:
        holiday = 1
    feats[HEADER_FEAT[5]] = np.ones((DATA_A_DAY)) * holiday

    if mode == VALID:
        feats[HEADER_FEAT[6]] = data_yesterday[HEADER_FEAT[6]].values

    data_week_day = data_features[HEADER_WEEK_DAY].values
    idx_week_day = data_week_day == week_day

    data_holiday = data_features[HEADER_HOLIDAY].values
    idx_holidays = data_holiday == holiday

    idx = idx_week_day & idx_holidays

    flow_week_day = data_features[HEADER_FLOW].values[idx]
    size = len(flow_week_day)
    days = int(size/data_a_day)
    flow_week_day = np.reshape(flow_week_day, (days, data_a_day))



    print(LOG_NAME + ': genFlowYesterday ......')
    feats[HEADER_HIS[0]] = data_yesterday[HEADER_FLOW].values
    print(LOG_NAME + ': genFlowLastWeek ......')
    feats[HEADER_HIS[1]] = flow_week_day[-1, :]
    print(LOG_NAME + ': genFlowAverWeek ......')
    feats[HEADER_HIS[2]] = np.mean(flow_week_day, axis=0)

    print(LOG_NAME + ': genFlowAverWeekOut ......')
    feats[HEADER_HIS[3]] = genFlowAverWeekOutIn(flow_week_day)  # ignore type

    print(LOG_NAME + ': genFlowYesterdayNormal ......')
    feats[HEADER_HIS[4]] = data_yesterday[HEADER_FLOW].values

    df = pd.DataFrame(feats)
    if mode == PRED:
        df.to_csv(DATA_PATH_TEST+'features_his.csv', index=False)
    elif mode == VALID:
        df.to_csv(DATA_PATH_TEST + 'features_valid.csv', index=False)


if __name__ == '__main__':
    genHisData()

