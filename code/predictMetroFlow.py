# -*- coding: utf-8 -*-

import tensorflow as tf
import datetime
from Autoencoder import Autoencoder
from model_base import *
from layers import MLP
from tensorflow.python.keras.layers import Dense, InputLayer
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras import backend as KTF
from tensorflow.python.ops import math_ops
from analyzeData import analyze
from generateTestFeatures import genHisData
from blend_results import blendResults
from param_models import *

from predToFile import writeToSubmit
import random
import numpy as np
import argparse

from constants import *

parser = argparse.ArgumentParser()
parser.add_argument('--input', '--submit', type=str)
args = parser.parse_args()

# update constants
YESTERDAY_TEST = args.input
SUBMIT_FILE = args.submit
DATE_TEST = SUBMIT_FILE[-14:-4]
s = SUBMIT_FILE.split('/')
DATA_PATH_TEST = '/'.join(s[:-1])

# prepare test data
analyze(DATA_PATH_TEST)
genHisData()

# load test data
test_file = DATA_PATH_TEST + 'features_his.csv'
pred_x = readTestData(test_file)

# parameters needed to set
batch_size = 144 * 4 * 2 * 81
n_inputs = sum(FEAT_LEN)
# n_outputs = 3
n_outputs = 4
# n_mlp = [400, 400, 400]
# n_mlp = [400, 400, 400, 400, 400, 400]
n_mlp = [500, 500]
l2_param = 0.01

paths_save = []

paths_model = [
    # 12.28
    '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 00:16:45.csv',
    # 12.50
    '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 00:31:26.csv',
    # 12.66
    '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 00:34:40.csv',
    # 12.94
    '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 00:47:25.csv',
    # 13.05
    '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 00:49:54.csv',

    # 12.28 -> 12.54 -> 12.53
    '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 09:45:08.csv',
    # 12.50 ->
    '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 18:37:00.csv',
    # 12.66 ->
    '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 18:40:37.csv',
    # 12.94 ->
    '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 21:44:34.csv',
    # 13.05 ->
    '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 21:47:27.csv',
]


def predict(mlp):
    input_x = nextBatch(pred_x, n_outputs)
    y_pred = mlp.predict(input_x)
    y_pred = np.reshape(y_pred, (-1, 1))
    paths_save.append(writeToSubmit((y_pred)))


def loadAndPred(path_model):
    mlp = Sequential()
    mlp.add(InputLayer(input_shape=(n_inputs * n_outputs,)))
    for n in n_mlp:
        mlp.add(Dense(units=n, activation=tf.nn.relu,
                      kernel_regularizer=l2(l2_param), bias_regularizer=l2(l2_param)))
    mlp.add(Dense(units=n_outputs, activation=tf.nn.relu,
                  kernel_regularizer=l2(l2_param), bias_regularizer=l2(l2_param)))

    mlp.load_weights(path_model)
    predict(mlp)


params = [param_model_1228, param_model_1250, param_model_1266, param_model_1294, param_model_1305]
for idx, p_m in enumerate(paths_model):
    params[idx]()
    loadAndPred(p_m)

blendResults(paths_save)
