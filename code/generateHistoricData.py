# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from constants import *

# the number of data a day is 144(block) * 4(type)* 2(in_out) * 81(station)
data_a_day = DATA_A_DAY
LOG_NAME = 'genHisData'
yesterday_type = 1  # 0 normal, 1 Saturday then use last Saturday as input, Monday then use last Friday as input

# !!! python actions are performed along line


def genFlowYesterdayNormal(dataFLow, holidays, weekdays):
    # the number of data a day is 144(block) * 4(type)* 2(in_out) * 81(station)
    size = len(dataFLow)
    days = int(size/data_a_day)
    dataFLow = np.reshape(dataFLow, (days, data_a_day))
    flow_gen = np.zeros(dataFLow.shape)

    # construct as historical feature by using yesterday's data
    for i in range(0, days):
        # if first day, use the average of all days
        idx_holiday = holidays == holidays[i]

        # first day, use mean val of all weekdays
        if (i < 1) or (holidays[i] != holidays[i-1]):
            idx_weekday = weekdays % 7 <= 4
            idx = idx_holiday & idx_weekday
            flow_gen[i, :] = np.mean(dataFLow[idx, :], axis=0)
        else:
            flow_gen[i, :] = dataFLow[i-1, :]

    return np.reshape(flow_gen, (size))


def genFlowYesterday(dataFLow, holidays, weekdays):
    # the number of data a day is 144(block) * 4(type)* 2(in_out) * 81(station)
    size = len(dataFLow)
    days = int(size/data_a_day)
    dataFLow = np.reshape(dataFLow, (days, data_a_day))
    flow_gen = np.zeros(dataFLow.shape)

    # construct as historical feature by using yesterday's data
    for i in range(0, days):
        # if first day, use the average of all days
        idx_holiday = holidays == holidays[i]

        wd = weekdays[i]
        # Saturday then use last Saturday as input, if no last Saturday, use mean val of all weekends
        if wd % 7 == 5:
            if i - 7 < 0:
                idx_weekday = weekdays % 7 > 4
                idx = idx_holiday & idx_weekday
                flow_gen[i, :] = np.mean(dataFLow[idx, :], axis=0)
            else:
                flow_gen[i, :] = dataFLow[i - 7, :]
        # Monday then use last Friday as input, if no last Friday, use mean val of all weekdays
        elif wd % 7 == 0:
            if i - 3 < 0:
                idx_weekday = weekdays % 7 <= 4
                idx = idx_holiday & idx_weekday
                flow_gen[i, :] = np.mean(dataFLow[idx, :], axis=0)
            else:
                flow_gen[i, :] = dataFLow[i - 3, :]
        # first day, use mean val of all weekdays
        elif (i < 1) or (holidays[i] != holidays[i-1]):
            idx_weekday = weekdays % 7 <= 4
            idx = idx_holiday & idx_weekday
            flow_gen[i, :] = np.mean(dataFLow[idx, :], axis=0)
        else:
            flow_gen[i, :] = dataFLow[i-1, :]

    return np.reshape(flow_gen, (size))


def genFlowLastWeek(dataFLow, holidays):
    # the number of data a day is 144(block) * 4(type)* 2(in_out) * 81(station)
    size = len(dataFLow)
    days = int(size/data_a_day)
    dataFLow = np.reshape(dataFLow, (days, data_a_day))
    flow_gen = np.zeros(dataFLow.shape)

    # construct as historical feature by using last weeks' data
    for i in range(0, days):
        # if first week, use the average of all same weekdays
        if (i < 7) or (holidays[i] != holidays[i-7]):
            idx_holiday = holidays == holidays[i]
            weekdays = np.array([False] * days)
            weekdays[np.arange(i % 7, days, 7)] = True

            weekdays = weekdays & idx_holiday
            flow_gen[i, :] = np.mean(dataFLow[weekdays, :], axis=0)
        else:
            flow_gen[i, :] = dataFLow[i-7, :]

    return np.reshape(flow_gen, (size))


def genFlowAverWeek(dataFLow, holidays):
    # the number of data a day is 144(block) * 4(type)* 2(in_out) * 81(station)
    size = len(dataFLow)
    days = int(size/data_a_day)
    dataFLow = np.reshape(dataFLow, (days, data_a_day))
    flow_gen = np.zeros(dataFLow.shape)

    # construct as historical feature by using pass weeks' data
    for i in range(0, days):
        # if first week, use the average of all same weekdays

        idx_holidays = holidays == holidays[i]

        weekdays = np.array([False] * days)

        # only use average of before same weekdays
        weekdays[np.arange(i % 7, i, 7)] = True

        # use the average of all same weekdays
        # weekdays[np.arange(i % 7, days, 7)] = True

        weekdays = weekdays & idx_holidays

        if sum(weekdays) < 1:
            weekdays[np.arange(i % 7, days, 7)] = True
            weekdays = weekdays & idx_holidays

        # print(weekdays)

        flow_gen[i, :] = np.mean(dataFLow[weekdays, :], axis=0)

    return np.reshape(flow_gen, (size))


def genFlowAverWeekOutIn(dataFLow, holidays):
    # the number of data a day is 144(block) * 4(type)* 2(in_out) * 81(station)
    # ignore type
    size = len(dataFLow)
    days = int(size/data_a_day)
    dataFLow = np.reshape(dataFLow, (days, data_a_day))
    temp_gen = np.zeros(dataFLow.shape)
    flow_gen = np.zeros(dataFLow.shape)

    step = NUM_TIME_BLOCK * NUM_TYPES
    # get the in out flow ignore type
    for i in range(days-1, -1, -1):
        for s in range(0, data_a_day, step):
            temp = np.reshape(dataFLow[i, s: s+step], (NUM_TYPES, NUM_TIME_BLOCK))
            temp_gen[i, s: s+step] = np.tile(np.sum(temp, axis=0), (NUM_TYPES))

    # construct as historical feature by using pass weeks' data
    for i in range(0, days):
        # if first week, use the average of all same weekdays
        idx_holidays = holidays == holidays[i]

        weekdays = np.array([False] * days)
        weekdays[np.arange(i % 7, i, 7)] = True

        weekdays = weekdays & idx_holidays

        if sum(weekdays) < 1:
            weekdays[np.arange(i % 7, days, 7)] = True
            weekdays = weekdays & idx_holidays

        # print(weekdays)

        flow_gen[i, :] = np.mean(temp_gen[weekdays, :], axis=0)

    return np.reshape(flow_gen, (size))


def getHolidays(holidays):
    size = len(holidays)
    days = int(size / data_a_day)
    holidays = np.reshape(holidays, (days, data_a_day))
    holidays = holidays[:, 0]
    return holidays


def genHisData():
    print(LOG_NAME + ': reading data ......')
    data_features = pd.read_csv(DATA_PATH_FEAT+'features.csv')

    holidays = getHolidays(data_features[HEADER_HOLIDAY].values)
    weekdays = getHolidays(data_features[HEADER_WEEK_DAY].values)

    print(LOG_NAME + ': genFlowYesterday ......')
    flow_yesterday = genFlowYesterday(data_features[HEADER_FLOW].values, holidays, weekdays)
    print(LOG_NAME + ': genFlowYesterdayNormal ......')
    flow_yesterday_normal = genFlowYesterdayNormal(data_features[HEADER_FLOW].values, holidays, weekdays)
    print(LOG_NAME + ': genFlowLastWeek ......')
    flow_last_week = genFlowLastWeek(data_features[HEADER_FLOW].values, holidays)
    print(LOG_NAME + ': genFlowAverWeek ......')
    flow_aver_week = genFlowAverWeek(data_features[HEADER_FLOW].values, holidays)

    print(LOG_NAME + ': genFlowAverWeekOut ......')
    flow_aver_week_out_in = genFlowAverWeekOutIn(data_features[HEADER_FLOW].values, holidays)  # ignore type

    # we need to forecast in out flow, so it may be no use to ignore in_out

    # print(LOG_NAME + ': genFlowAverWeekALL ......')
    # flow_aver_week_all = genFlowAverWeekALL(data_features[HEADER_FLOW].values)  # ignore time block

    # print(LOG_NAME + ': genFlowAverWeekType ......')
    # flow_aver_week_all = genFlowAverWeekType1(data_features[HEADER_FLOW].values)  # ignore in_out
    # flow_aver_week_all = genFlowAverWeekType2(data_features[HEADER_FLOW].values)  # ignore in_out
    # flow_aver_week_all = genFlowAverWeekType3(data_features[HEADER_FLOW].values)  # ignore in_out
    # flow_aver_week_all = genFlowAverWeekType4(data_features[HEADER_FLOW].values)  # ignore in_out
    #
    # flow_aver_week_all = genFlowAverWeekBlock(data_features[HEADER_FLOW].values)  # ignore in_out and type
    #
    # flow_aver_week_all = genFlowAverWeekBlock(data_features[HEADER_FLOW].values)  # ignore in_out and type and time block
    #
    # flow_aver_week_all = genFlowAverWeekBlock(data_features[HEADER_FLOW].values)  # ignore station

    print(LOG_NAME + ': save file ......')
    data_features[HEADER_HIS[0]] = flow_yesterday
    data_features[HEADER_HIS[1]] = flow_last_week
    data_features[HEADER_HIS[2]] = flow_aver_week
    data_features[HEADER_HIS[3]] = flow_aver_week_out_in
    data_features[HEADER_HIS[4]] = flow_yesterday_normal

    data_features.to_csv(DATA_PATH_FEAT+'features_his.csv', index=False)


if __name__ == '__main__':
    genHisData()

