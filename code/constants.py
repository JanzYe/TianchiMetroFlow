# -*- coding: utf-8 -*-
import os
from param_models import *

f = open('./testSetting.txt', 'r')
state = f.readline().strip()

EPOCH_EXCEED = 20

DATE_START = '2019-01-01'
# Test A
# DATA_PATH_TEST = '../Metro_testA/'  # file name format: record_2019-01-02.csv
# # DATE_TEST = '2019-01-28'
# DATE_TEST = '2019-01-29'
# YESTERDAY_TEST = 'testA_record_2019-01-28.csv'
# # SUBMIT_FILE = 'testA_record_2019-01-28.csv'
# SUBMIT_FILE = 'testA_submit_2019-01-29.csv'

# Test B
# DATA_PATH_TEST = '../Metro_testB/'  # file name format: record_2019-01-02.csv
# DATE_TEST = '2019-01-27'
# YESTERDAY_TEST = 'testB_record_2019-01-26.csv'
# SUBMIT_FILE = 'testB_submit_2019-01-27.csv'
# SUBMIT_RESULT = '3_Bresult.csv'


# Test A online
if state == 'A':
    DATA_PATH_TEST = '../data/testA/'  # file name format: record_2019-01-02.csv
    DATE_TEST = '2019-01-29'
    YESTERDAY_TEST = 'testA_record_2019-01-28.csv'
    SUBMIT_FILE = 'testA_submit_2019-01-29.csv'
    SUBMIT_RESULT = '../submit/testA/3_testA_results.csv'
    DECAY = 0.98  # with_tol: 1->12.28 no_tol: 1->12.05 0.98->11.88 0.95->12.12
    if not os.path.exists('../submit/testA/'):
        os.mkdir('../submit/testA/')


# Test B online
if state == 'B':
    DATA_PATH_TEST = '../data/testB/'  # file name format: record_2019-01-02.csv
    DATE_TEST = '2019-01-27'
    YESTERDAY_TEST = 'testB_record_2019-01-26.csv'
    SUBMIT_FILE = 'testB_submit_2019-01-27.csv'
    SUBMIT_RESULT = '../submit/testB/3_testB_results.csv'
    DECAY = 1  # with_tol: 1->11.80 no_tol: 1->11.89 
    if not os.path.exists('../submit/testB/'):
        os.mkdir('../submit/testB/')


# Test C online
if state == 'C':
    DATA_PATH_TEST = '../data/testC/'  # file name format: record_2019-01-02.csv
    DATE_TEST = '2019-01-31'
    YESTERDAY_TEST = 'testC_record_2019-01-30.csv'
    SUBMIT_FILE = 'testC_submit_2019-01-31.csv'
    SUBMIT_RESULT = '../submit/testC/3_testC_results.csv'
    DECAY = 0.80  # with_tol: 1->17.13  no_tol: 1->16.51 0.95->13.85 0.90->12.26 0.85->12.00 0.80->13.14
    if not os.path.exists('../submit/testC/'):
        os.mkdir('../submit/testC/')

HEADER_IN = 'inNums'
HEADER_OUT = 'outNums'

if not os.path.exists('../submit/'):
    os.mkdir('../submit/')

DATA_PATH_ORI = '../data/train/'  # file name format: record_2019-01-02.csv
DATA_PATH_FEAT = DATA_PATH_ORI + 'features/'
if not os.path.exists(DATA_PATH_FEAT):
    os.mkdir(DATA_PATH_FEAT)
DATA_PATH_IMG = DATA_PATH_ORI+'images/'
if not os.path.exists(DATA_PATH_IMG):
    os.mkdir(DATA_PATH_IMG)
DATA_PATH_RESULT = DATA_PATH_ORI+'results/'
if not os.path.exists(DATA_PATH_RESULT):
    os.mkdir(DATA_PATH_RESULT)

DATA_PATH_MODELS = DATA_PATH_ORI+'models/'
# DATA_PATH_MODELS = DATA_PATH_TEST+'models/'
if not os.path.exists(DATA_PATH_MODELS):
    os.mkdir(DATA_PATH_MODELS)


DATA_PATH_PRED = '../data/preds/'
if not os.path.exists(DATA_PATH_PRED):
    os.mkdir(DATA_PATH_PRED)

DATA_PATH_SUBMIT = '../data/submit/'
if not os.path.exists(DATA_PATH_SUBMIT):
    os.mkdir(DATA_PATH_SUBMIT)


DAYS = 25
NUM_STATIONS = 81
NUM_LINES = 3
NUM_TYPES = 4
NUM_TIME_BLOCK = 24 * 6
SIZE_TIME_BLOCK = 10
TIME_POS_START = 11  # yyyy-mm-dd HH:MM:SS
DATE_POS_END = 10  # yyyy-mm-dd HH:MM:SS
DATA_A_DAY = NUM_TIME_BLOCK * NUM_TYPES * 2 * NUM_STATIONS

STATUS_IN = 1
STATUS_OUT = 0

HEADER_ORI = ['time', 'lineID', 'stationID', 'deviceID', 'status', 'userID', 'payType']
HEADER_TIME = 'time'
HEADER_LINE_ID = 'lineID'  # A B C
HEADER_STATION_ID = 'stationID'
HEADER_DEVICE_ID = 'deviceID'
HEADER_STATUS = 'status'  # 0 out, 1 in
HEADER_USER_ID = 'userID'
HEADER_PAY_TYPE = 'payType'
HEADER_TIME_BLOCK = 'time_block'
HEADER_WEEK_DAY = 'weekDay'
HEADER_HOLIDAY = 'holiday'
HEADER_AVER_WEEK = 'averWeek'
HEADER_FLOW = 'flow'

FEAT_LEN = [144, 81, 2, 4, 7, 2, 1, 1, 1, 1]  # 244
# FEAT_LEN = [144, 81, 2, 4, 7, 1]
# 144 + 81 + 2 + 4 + 7 + 2 + 1 + 1 + 1 + 1
# HEADER_FEAT = ['block', 'stationID', 'status', 'payType', 'weekDay', 'flow']
HEADER_FEAT = ['block', 'stationID', 'status', 'payType', 'weekDay',  'holiday', 'flow']
# the flow of yesterday ...., if not have, use average flow of all similar time
HEADER_HIS = ['yesterday', 'lastWeek', 'averWeek', 'averWeekOutIn', 'yesterdayNormal']
# HEADER_HIS = ['yesterday']
HEADER_TEST = ['stationID', 'startTime', 'endTime', 'inNums', 'outNums']

HOLIDAY = ['2019-01-01']

COLORS = ['r', 'g', 'm', 'k', 'c', 'b', 'y',
         'tab:grey', 'tab:gray', 'tab:purple', 'tab:pink', 'tab:orange', ]

TRAIN = 'train'
VALID = 'valid'
GRID = 'grid'
PRED = 'pred'


# features: station_id|range(0, 81), pay_type|range(0, 4), line_id|range(0, 3), in_out|range(0, 2),
#           mi_10|range(0, 6 * 24), week|range(0, 7), holiday|range(0, 2), average_same_week_day(in_out_pay_type),
#           average_pre_num(pre 1, 2, 3 day or a week)
# label: number_in or number_out
# use or not: user_id, devices_id

WEEK_DAY_DICT = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday',
}



# FEAT_LEN, HEADER_FEAT, HEADER_HIS, n_inputs, n_outputs, n_mlp, lr = param_model_1250()
# FEAT_LEN, HEADER_FEAT, HEADER_HIS, n_inputs, n_outputs, n_mlp, lr = param_model_1266()
# FEAT_LEN, HEADER_FEAT, HEADER_HIS, n_inputs, n_outputs, n_mlp, lr = param_model_1294()
# FEAT_LEN, HEADER_FEAT, HEADER_HIS, n_inputs, n_outputs, n_mlp, lr = param_model_1305()
