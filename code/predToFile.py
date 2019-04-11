# -*- coding:utf-8 -*-

import pandas as pd
from constants import *
import numpy as np
import time


def flowIgnoreType(dataFLow):
    # the number of data a day is 144(block) * 4(type)* 2(in_out) * 81(station)
    # ignore type
    size = len(dataFLow)
    days = int(size/DATA_A_DAY)
    dataFLow = np.reshape(dataFLow, (days, DATA_A_DAY))
    temp_gen = np.zeros((days, NUM_TIME_BLOCK * 2 * NUM_STATIONS))

    step = NUM_TIME_BLOCK * NUM_TYPES
    # get the in out flow ignore type
    for i in range(days-1, -1, -1):
        for s in range(0, DATA_A_DAY, step):
            temp = np.reshape(dataFLow[i, s: s+step], (NUM_TYPES, NUM_TIME_BLOCK))
            temp_gen[i, int(s/NUM_TYPES): int((s+step)/NUM_TYPES)] = np.sum(temp, axis=0)

    return np.reshape(temp_gen, (days * NUM_TIME_BLOCK * 2 * NUM_STATIONS))


def writeToSubmit(preds):
    preds = flowIgnoreType(preds)

    submit = pd.read_csv(DATA_PATH_TEST + SUBMIT_FILE)

    in_preds = np.zeros((NUM_TIME_BLOCK * NUM_STATIONS))
    out_preds = np.zeros((NUM_TIME_BLOCK * NUM_STATIONS))
    for i in range(0, len(preds), NUM_TIME_BLOCK * 2):
        # print(int(i/2)+NUM_TIME_BLOCK)
        out_preds[int(i/2): int(i/2)+NUM_TIME_BLOCK] = preds[i: i+NUM_TIME_BLOCK]
        in_preds[int(i/2): int(i/2)+NUM_TIME_BLOCK] = preds[i+NUM_TIME_BLOCK: i+NUM_TIME_BLOCK*2]

    # the result less than 1 set to 0

    in_preds[in_preds < 1] = 0
    out_preds[out_preds < 1] = 0

    submit[HEADER_IN] = in_preds
    submit[HEADER_OUT] = out_preds

    now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    path_save = DATA_PATH_PRED + SUBMIT_FILE[:-4] + '_' +now_time + '.csv'
    submit.to_csv(path_save, index=False)

    return path_save

