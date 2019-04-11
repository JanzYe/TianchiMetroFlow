import tensorflow as tf
import datetime
from model_base import *
from tensorflow.python.keras.layers import Flatten, Dense, InputLayer
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras import backend as KTF
from tensorflow.python.ops import math_ops
import os

from predToFile import writeToSubmit
import random
import numpy as np
import argparse

from param_models import *
import constants

mode = constants.PRED

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str)
parser.add_argument('--epoch', type=int)
args = parser.parse_args()
mode = args.mode
training_epochs = args.epoch + constants.EPOCH_EXCEED

file_name = constants.DATA_PATH_MODELS + 'model_1266.h5'

input_file = constants.DATA_PATH_FEAT + 'features_his.csv'
test_file = constants.DATA_PATH_TEST + 'features_his.csv'

# training_epochs = 10000
# training_epochs = 2000
batch_size = 144 * 4 * 2 * 81
# batch_size = 144 * 2 * 4 * 81
display_step = 1

n_hidden = []
l2_param = 0.01
dropout = 0.8
shuffle_batch = True
shuffle_samples = False
weights = [1, 0, 0]

lr = 0.0001
constants.FEAT_LEN = [144, 81, 2, 4, 7, 1]
constants.HEADER_FEAT = ['block', 'stationID', 'status', 'payType', 'weekDay', 'flow']
constants.HEADER_HIS = ['yesterday']

n_inputs = sum(constants.FEAT_LEN)
n_outputs = 4
n_mlp = [500, 500]

# Read database

if mode == constants.TRAIN:
    input_x, input_y = readTrainData(input_file)
    train_x = input_x[2*constants.DATA_A_DAY:-2*constants.DATA_A_DAY, :]
    train_y = input_y[2*constants.DATA_A_DAY:-2*constants.DATA_A_DAY]

    n_samples = len(train_y)

    valid_x = np.reshape(input_x[-2*constants.DATA_A_DAY:, :], (int(2*constants.DATA_A_DAY / n_outputs), sum(constants.FEAT_LEN) * n_outputs))
    valid_y = np.reshape(input_y[-2*constants.DATA_A_DAY:], (int(2*constants.DATA_A_DAY / n_outputs), n_outputs))
else:
    pred_x = readTestData(test_file)


mlp = Sequential()
mlp.add(InputLayer(input_shape=(n_inputs * n_outputs,)))
for n in n_mlp:
    mlp.add(Dense(units=n, activation=tf.nn.relu,
                  kernel_regularizer=l2(l2_param), bias_regularizer=l2(l2_param)))
mlp.add(Dense(units=n_outputs, activation=tf.nn.relu,
                  kernel_regularizer=l2(l2_param), bias_regularizer=l2(l2_param)))



def validate(mlp):
    pred_y = mlp.predict(valid_x)
    mae_ignore_type = np.mean(np.abs(flowIgnoreType(np.reshape(pred_y, (-1,1))) -
                                     flowIgnoreType(np.reshape(valid_y, (-1,1)))))
    print(mae_ignore_type)
    return mae_ignore_type


def predict(mlp):
    input_x = nextBatch(pred_x, n_outputs)
    y_pred = mlp.predict(input_x)
    y_pred = np.reshape(y_pred, (-1, 1))
    writeToSubmit((y_pred))


def my_loss(y_true, y_pred, weights=weights):
    # r, c = y_pred.shape
    r = int(batch_size/n_outputs)
    diff = y_pred - y_true

    mse_samples = 0
    if weights[0] > 0:
       mse_samples = KTF.mean(math_ops.square(diff), axis=-1)

    mse_no_types = 0
    if weights[1] > 0:
        steps_type = int(constants.NUM_TIME_BLOCK * constants.NUM_TYPES / n_outputs)
        steps_one = int(constants.NUM_TIME_BLOCK / n_outputs)
        # print(r)
        # print(c)
        # print(steps_type)
        for i in range(0, r, steps_type):
            diff_no_type = diff[int(i+0*steps_one):int(i + (1)*steps_one), :]
            for j in range(1, constants.NUM_TYPES):
                diff_no_type = diff_no_type + diff[int(i+j*steps_one):int(i + (j+1)*steps_one), :]
            mse_no_types += KTF.sum(math_ops.square(diff_no_type)) / batch_size

    mse_total = 0
    if weights[2] > 0:
        diff_out_total = math_ops.square(math_ops.reduce_sum(diff[:int(r/2), :]))
        diff_in_total = math_ops.square(math_ops.reduce_sum(diff[int(r/2):, :]))
        mse_total = (diff_out_total + diff_in_total) / batch_size

    losses = (weights[0] * mse_samples + weights[1] * mse_no_types + weights[2] * mse_total) / np.sum(weights)

    # the meaning of this val depend on batch size
    # mse_total = math_ops.square(math_ops.reduce_sum(diff)) / batch_size
    #
    # losses = weights[0] * mse_samples + weights[1] * mse_total

    return losses


def my_metric(y_true, y_pred):
    # r, c = y_pred.shape
    r = int(batch_size/n_outputs)
    diff = y_pred - y_true

    mae_no_types = 0
    steps_type = int(constants.NUM_TIME_BLOCK * constants.NUM_TYPES / n_outputs)
    steps_one = int(constants.NUM_TIME_BLOCK / n_outputs)
    # print(r)
    # print(c)
    # print(steps_type)
    for i in range(0, r, steps_type):
        diff_no_type = diff[int(i+0*steps_one):int(i + (1)*steps_one), :]
        for j in range(1, constants.NUM_TYPES):
            diff_no_type = diff_no_type + diff[int(i+j*steps_one):int(i + (j+1)*steps_one), :]
        mae_no_types += KTF.sum(math_ops.abs(diff_no_type)) / (batch_size / constants.NUM_TYPES)

    return mae_no_types


## Initialization

def generator(steps_per_epoch):
    while True:

        pos = np.arange(steps_per_epoch).tolist()
        if shuffle_batch:
            random.shuffle(pos)
        # print(pos)

        for i in pos:
            start = (i) * batch_size
            end = (i + 1) * batch_size
            if end > n_samples:
                end = n_samples
            batch_xs = nextBatch(train_x[start:end, :], n_outputs)
            batch_ys = np.reshape(train_y[start:end], (int(batch_size / n_outputs), n_outputs))

            yield batch_xs, batch_ys

start = datetime.datetime.now()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)

# mlp.compile(optimizer=Adam(learning_rate=lr),
mlp.compile(optimizer=Adam(lr=lr),
              loss=my_loss,
              metrics=[my_metric], )

# file_name_trained = '/home/yezhizi/Documents/TianchiMetro/code/ckpt/lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 10000-n_inputs: 239-n_outputs: 3-n_hidden: []-n_mlp: [400, 400, 400]-n_samples2146176-weights[10000, 100, 1]/mlp-ep7600-loss88.766-val_loss98.917-lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 10000-n_inputs: 239-n_outputs: 3-n_hidden: []-n_mlp: [400, 400, 400]-n_samples2146176-weights[10000, 100, 1].h5'
# mlp.load_weights(file_name_trained)
if mode == constants.TRAIN:

    steps_per_epoch = int(np.ceil(n_samples / batch_size))
    check_point = ModelCheckpoint(file_name, monitor='val_my_metric', verbose=0,
                                  save_best_only=True,
                                  save_weights_only=False, mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='logs/' + file_name[:-3] + '/', histogram_freq=0, write_graph=True,
                               write_images=False)

    result = mlp.fit_generator(generator=generator(steps_per_epoch), steps_per_epoch=steps_per_epoch,
                               epochs=training_epochs, shuffle=shuffle_samples, validation_data=(valid_x, valid_y),
                               verbose=2, callbacks=[check_point, tensor_board])

    print("*************************Finish the softmax output layer training*****************************")
    # saver.save(sess, 'ckpt/sae.ckpt', global_step=epoch)


    # pred = mlp.predict(h
    # print(np.mean(np.abs(pred-train_y[-DATA_A_DAY:])))
    mae = validate(mlp)
    predict(mlp)

    end = datetime.datetime.now()

    rcd = str(end) + '\n'
    rcd += "lr: " + str(lr) + '\n'
    rcd += "batch_size: " + str(batch_size) + '\n'
    rcd += "l2_param: " + str(l2_param) + '\n'
    rcd += "dropout: " + str(dropout) + '\n'
    rcd += "training_epochs: " + str(training_epochs) + '\n'
    rcd += "n_inputs: " + str(n_inputs) + '\n'
    rcd += "n_outputs: " + str(n_outputs) + '\n'
    rcd += "n_mlp: " + str(n_mlp) + '\n'
    rcd += "mae: " + str(mae) + '\n'
    rcd += "time: " + str(end - start) + '\n' + '\n' + '\n'
    print(rcd)
    log_file = open(constants.DATA_PATH_RESULT + "mlp_result", "a")
    log_file.write(rcd)
    log_file.close()

    # mlp.save('ckpt/'+file_name[:-3]+'/'+file_name)


elif mode == constants.PRED:
    mlp.load_weights(file_name)
    predict(mlp)