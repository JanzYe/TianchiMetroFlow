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

from predToFile import writeToSubmit
import random
import numpy as np
from constants import *

from param_models import *

mode = PRED

input_file = DATA_PATH_FEAT + 'features_his.csv'
test_file = DATA_PATH_TEST + 'features_his.csv'
valid_file = DATA_PATH_TEST + 'features_valid.csv'

#training_epochs = 10000
training_epochs = 2000
batch_size = 144 * 4 * 2 * 81
# batch_size = 144 * 2 * 4 * 81
display_step = 1

#
n_inputs = sum(FEAT_LEN)
# n_hidden = [4, 4, 4]
# n_outputs = 3
n_outputs = 4
n_hidden = []
# n_hidden = [400, 400, 400]
# n_mlp = [400, 400, 400]
# n_mlp = [400, 400, 400, 400, 400, 400]
n_mlp = [500, 500]
# n_mlp = [300, 300, 300, 300]
lr = 0.0001
l2_param = 0.01
dropout = 0.8
shuffle_batch = True
shuffle_samples = False
weights = [1, 0, 0]

# FEAT_LEN, HEADER_FEAT, HEADER_HIS, n_inputs, n_outputs, n_mlp, lr = param_model_1250()
# FEAT_LEN, HEADER_FEAT, HEADER_HIS, n_inputs, n_outputs, n_mlp, lr = param_model_1266()
# FEAT_LEN, HEADER_FEAT, HEADER_HIS, n_inputs, n_outputs, n_mlp, lr = param_model_1294()
FEAT_LEN, HEADER_FEAT, HEADER_HIS, n_inputs, n_outputs, n_mlp, lr = param_model_1305()

# Read database
# input_x, input_y = readTrainPartial(input_file)
input_x, input_y = readTrainData(input_file)
# train_x = input_x
# train_y = input_y
# train_x = np.vstack((input_x[:-4*DATA_A_DAY, :], input_x[-3*DATA_A_DAY:, :]))
# train_y = np.hstack((input_y[:-4*DATA_A_DAY], input_y[-3*DATA_A_DAY:]))
# train_x = np.vstack((input_x[2*DATA_A_DAY:-2*DATA_A_DAY, :]))
# train_y = np.hstack((input_y[2*DATA_A_DAY:-2*DATA_A_DAY]))
train_x = np.vstack((input_x[2*DATA_A_DAY:, :]))
train_y = np.hstack((input_y[2*DATA_A_DAY:]))

# y_max = np.max(train_y)
# train_y = train_y / y_max
pred_x = readTestData(test_file)
# valid_x, valid_y = readTrainData(valid_file)
# for i in range(4, 0, -1):
#     max_val = np.max(train_x[:, -i])
#     train_x[:, -i] = train_x[:, -i] / max_val
#     pred_x[:, -i] = pred_x[:, -i] / max_val
#     print(max_val)

n_samples = len(train_y)


if shuffle_samples:
    np.random.seed(2211)
    np.random.shuffle(train_x)
    np.random.seed(2211)
    np.random.shuffle(train_y)



# valid_x = np.reshape(input_x[-DATA_A_DAY:, :], (int(DATA_A_DAY / n_outputs), sum(FEAT_LEN) * n_outputs))
# valid_y = np.reshape(input_y[-DATA_A_DAY:], (int(DATA_A_DAY / n_outputs), n_outputs))

if mode == PRED:
    valid_x = np.reshape(input_x[-6*DATA_A_DAY:-5*DATA_A_DAY, :], (int(DATA_A_DAY / n_outputs), sum(FEAT_LEN) * n_outputs))
    valid_y = np.reshape(input_y[-6*DATA_A_DAY:-5*DATA_A_DAY], (int(DATA_A_DAY / n_outputs), n_outputs))

    # valid_x = np.reshape(input_x[-2 * DATA_A_DAY:, :], (int(2 * DATA_A_DAY / n_outputs), sum(FEAT_LEN) * n_outputs))
    # valid_y = np.reshape(input_y[-2 * DATA_A_DAY:], (int(2 * DATA_A_DAY / n_outputs), n_outputs))

if mode == TRAIN:
    # valid_x = np.reshape(input_x[-2*DATA_A_DAY:, :], (int(2*DATA_A_DAY / n_outputs), sum(FEAT_LEN) * n_outputs))
    # valid_y = np.reshape(input_y[-2*DATA_A_DAY:], (int(2*DATA_A_DAY / n_outputs), n_outputs))

    valid_x = np.reshape(input_x[-6 * DATA_A_DAY:-5 * DATA_A_DAY, :], (int(DATA_A_DAY / n_outputs), sum(FEAT_LEN) * n_outputs))
    valid_y = np.reshape(input_y[-6 * DATA_A_DAY:-5 * DATA_A_DAY], (int(DATA_A_DAY / n_outputs), n_outputs))

# valid_x = np.reshape(valid_x, (int(1*DATA_A_DAY / n_outputs), sum(FEAT_LEN) * n_outputs))
# valid_y = np.reshape(valid_y, (int(1*DATA_A_DAY / n_outputs), n_outputs))


mlp = Sequential()
mlp.add(InputLayer(input_shape=(n_inputs * n_outputs,)))
for n in n_hidden:
    mlp.add(Dense(units=n, activation=tf.nn.tanh,
                  kernel_regularizer=l2(l2_param), bias_regularizer=l2(l2_param)))
for n in n_mlp:
    mlp.add(Dense(units=n, activation=tf.nn.relu,
                  kernel_regularizer=l2(l2_param), bias_regularizer=l2(l2_param)))
mlp.add(Dense(units=n_outputs, activation=tf.nn.relu,
                  kernel_regularizer=l2(l2_param), bias_regularizer=l2(l2_param)))


def validate(mlp):
    # input_x = nextBatch(valid_x[-DATA_A_DAY:, :], n_outputs)
    #
    # batch_ys = np.reshape(train_y[-DATA_A_DAY:], (int(DATA_A_DAY / n_outputs), n_outputs))
    # print(mlp.evaluate(input_x, batch_ys))

    # print(mlp.evaluate(valid_x, valid_y))
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


# mlp = Sequential([
#     Flatten(input_shape=(n_inputs * n_outputs,)),
#     MLP(n_mlp, 'tanh', l2_param, keep_prob=dropout, use_bn=True, seed=1024)
# ])

def my_loss(y_true, y_pred, weights=weights):
    # r, c = y_pred.shape
    r = int(batch_size/n_outputs)
    diff = y_pred - y_true

    mse_samples = 0
    if weights[0] > 0:
       mse_samples = KTF.mean(math_ops.square(diff), axis=-1)

    mse_no_types = 0
    if weights[1] > 0:
        steps_type = int(NUM_TIME_BLOCK * NUM_TYPES / n_outputs)
        steps_one = int(NUM_TIME_BLOCK / n_outputs)
        # print(r)
        # print(c)
        # print(steps_type)
        for i in range(0, r, steps_type):
            diff_no_type = diff[int(i+0*steps_one):int(i + (1)*steps_one), :]
            for j in range(1, NUM_TYPES):
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
    steps_type = int(NUM_TIME_BLOCK * NUM_TYPES / n_outputs)
    steps_one = int(NUM_TIME_BLOCK / n_outputs)
    # print(r)
    # print(c)
    # print(steps_type)
    for i in range(0, r, steps_type):
        diff_no_type = diff[int(i+0*steps_one):int(i + (1)*steps_one), :]
        for j in range(1, NUM_TYPES):
            diff_no_type = diff_no_type + diff[int(i+j*steps_one):int(i + (j+1)*steps_one), :]
        mae_no_types += KTF.sum(math_ops.abs(diff_no_type)) / (batch_size / NUM_TYPES)

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


file_name = 'mlp-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}' + "-lr: " + str(lr) + \
            "-batch_size: " + str(batch_size) + "-l2_param: " \
            + str(l2_param) + "-dropout: " + str(dropout) + "-training_epochs: " + str(training_epochs) + \
            "-n_inputs: " + str(n_inputs) + "-n_outputs: " + str(n_outputs) + "-n_hidden: " + str(n_hidden) \
            + "-n_mlp: " + str(n_mlp) + '-n_samples' + str(n_samples) + '-weights' + str(weights) + '.h5'
start = datetime.datetime.now()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)

steps_per_epoch = int(np.ceil(n_samples/batch_size))
if not os.path.exists('ckpt/'+file_name[56:-3]):
    os.mkdir('ckpt/'+file_name[56:-3])
check_point = ModelCheckpoint('ckpt/'+file_name[56:-3]+'/'+file_name, monitor='val_my_metric', verbose=0, save_best_only=True,
                           save_weights_only=False, mode='auto', period=1)
tensor_board = TensorBoard(log_dir='logs/'+file_name[:-3]+'/', histogram_freq=0, write_graph=True, write_images=False)

# mlp.compile(optimizer=Adam(learning_rate=lr),
mlp.compile(optimizer=Adam(lr=lr),
              loss=my_loss,
              metrics=[my_metric], )

# file_name_trained = '/home/yezhizi/Documents/TianchiMetro/code/ckpt/lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 10000-n_inputs: 239-n_outputs: 3-n_hidden: []-n_mlp: [400, 400, 400]-n_samples2146176-weights[10000, 100, 1]/mlp-ep7600-loss88.766-val_loss98.917-lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 10000-n_inputs: 239-n_outputs: 3-n_hidden: []-n_mlp: [400, 400, 400]-n_samples2146176-weights[10000, 100, 1].h5'
# mlp.load_weights(file_name_trained)

if mode == TRAIN:

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
    log_file = open(DATA_PATH_RESULT + "mlp_result", "a")
    log_file.write(rcd)
    log_file.close()

    mlp.save('ckpt/'+file_name[:-3]+'/'+file_name)


elif mode == PRED:
    # file_name = 'ckpt/mlp-ep250-loss81.952-val_loss96.125-lr: 0.001-batch_size: 93312-l2_param: 0.001-' \
    #             'dropout: 1-training_epochs: 2000-n_inputs: 244-n_outputs: 4-' \
    #             'n_mlp: [400, 400, 400, 400, 400, 400, 4].h5'

    # no averWeekOutIn
    # 13.21
    # file_name = 'ckpt/mlp-ep320-loss81.004-val_loss92.560-lr: 0.001-batch_size: 93312-l2_param: 0.001-dropout: 1-training_epochs: 2000-n_inputs: 243-n_outputs: 4-n_mlp: [400, 400, 400, 400, 400, 400, 4].h5'
    # 12.94
    # validation data -1:
    file_name = 'ckpt/mlp-ep920-loss81.810-val_loss94.843-lr: 0.001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 2000-n_inputs: 243-n_outputs: 4-n_mlp: [400, 400, 400, 400, 400, 400, 4].h5'
    # file_name = 'ckpt/mlp-ep1080-loss77.875-val_loss91.735-lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 2000-n_inputs: 243-n_outputs: 4-n_mlp: [400, 400, 400, 400, 400, 400, 4].h5'
    # 14.85
    # file_name = 'ckpt/mlp-ep1720-loss82.743-val_loss85.495-lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 2000-n_inputs: 243-n_outputs: 2-n_mlp: [200, 200, 200, 200, 200, 200, 2].h5'
    # file_name = 'ckpt/mlp-ep1640-loss78.985-val_loss90.669-lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 8000-n_inputs: 243-n_outputs: 4-n_mlp: [200, 200, 200, 200, 200, 200, 4].h5'

    # validation data -3:
    # 13.05
    file_name = 'ckpt/mlp-ep1420-loss77.416-val_loss99.322-lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 8000-n_inputs: 243-n_outputs: 4-n_mlp: [400, 400, 400, 4].h5'
    # file_name = 'ckpt/mlp-ep1640-loss78.985-val_loss90.669-lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 8000-n_inputs: 243-n_outputs: 4-n_mlp: [200, 200, 200, 200, 200, 200, 4].h5'
    # file_name = 'ckpt/mlp-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 8000-n_inputs: 243-n_outputs: 4-n_mlp: [200, 200, 200, 200, 200, 200, 4].h5'
    # file_name = 'ckpt/mlp-ep7660-loss77.017-val_loss97.647-lr: 0.0001-batch_size: 93312-l2_param: 0.02-dropout: 0.8-training_epochs: 8000-n_inputs: 242-n_outputs: 3-n_mlp: [400, 400, 400, 3].h5'

    # file_name = 'ckpt/mlp-ep7660-loss77.017-val_loss97.647-lr: 0.0001-batch_size: 93312-l2_param: 0.02-dropout: 0.8-training_epochs: 8000-n_inputs: 242-n_outputs: 3-n_mlp: [400, 400, 400, 3].h5'
    # file_name = 'ckpt/mlp-ep2140-loss78.894-val_loss94.851-lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 8000-n_inputs: 242-n_outputs: 3-n_mlp: [400, 400, 400, 3].h5'

    # train data 1: -3  no holiday, before this, train data start from first day

    # train data 2: -2  no holiday, the first Wednesday has no yesterday, averWeek use all same weekdays
    # file_name = 'ckpt/mlp-ep3860-loss50.897-val_loss73.727-lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 8000-n_inputs: 240-n_outputs: 3-n_mlp: [400, 400, 400, 3]n_samples1959552.h5'

    # 12.28, only use yesterday, train data 2: -2, yesterday used see generateHistoricData, before this, yesterday are normal
    file_name = 'ckpt/mlp-ep3720-loss87.851-val_loss98.356-lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 8000-n_inputs: 239-n_outputs: 3-n_mlp: [400, 400, 400, 3]-n_samples1959552.h5'
    # 12.6625
    file_name = 'ckpt/mlp-ep8200-loss90.406-val_loss101.127-lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 10000-n_inputs: 239-n_outputs: 4-n_mlp: [500, 500, 4]-n_samples2146176.h5'
    # file_name = 'ckpt/mlp-ep2780-loss89.827-val_loss97.411-lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 10000-n_inputs: 239-n_outputs: 3-n_mlp: [300, 300, 300, 300, 3]-n_samples2146176.h5'

    # self-define loss
    # 12.50
    file_name = '/home/yezhizi/Documents/TianchiMetro/code/ckpt/lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 10000-n_inputs: 239-n_outputs: 3-n_hidden: []-n_mlp: [400, 400, 400]-n_samples2146176-weights[10000, 100, 1]/mlp-ep5060-loss94.926-val_loss101.145-lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 10000-n_inputs: 239-n_outputs: 3-n_hidden: []-n_mlp: [400, 400, 400]-n_samples2146176-weights[10000, 100, 1].h5'
    # 12.5221
    # file_name = '/home/yezhizi/Documents/TianchiMetro/code/ckpt/lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 10000-n_inputs: 239-n_outputs: 3-n_hidden: []-n_mlp: [400, 400, 400]-n_samples2146176-weights[10000, 100, 1]/mlp-ep7600-loss88.766-val_loss98.917-lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 10000-n_inputs: 239-n_outputs: 3-n_hidden: []-n_mlp: [400, 400, 400]-n_samples2146176-weights[10000, 100, 1].h5'

    # wrong valid set
    # file_name = '/home/yezhizi/Documents/TianchiMetro/code/ckpt/lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 10000-n_inputs: 239-n_outputs: 3-n_hidden: []-n_mlp: [400, 400, 400]-n_samples2146176-weights[1, 1, 0]/mlp-ep7820-loss94.740-val_loss92.153-lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 10000-n_inputs: 239-n_outputs: 3-n_hidden: []-n_mlp: [400, 400, 400]-n_samples2146176-weights[1, 1, 0].h5'

    # 12.28, total training data in training
    # file_name = '/home/yezhizi/Documents/TianchiMetro/code/ckpt/lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 10000-n_inputs: 239-n_outputs: 4-n_hidden: []-n_mlp: [400, 400, 400]-n_samples2146176-weights[1, 0, 0]/mlp-ep3682-loss84.225-val_loss76.188-lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 10000-n_inputs: 239-n_outputs: 4-n_hidden: []-n_mlp: [400, 400, 400]-n_samples2146176-weights[1, 0, 0].h5'
    # 12.50, total training data in training
    # file_name = '/home/yezhizi/Documents/TianchiMetro/code/ckpt/lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 10000-n_inputs: 239-n_outputs: 4-n_hidden: []-n_mlp: [400, 400, 400]-n_samples2146176-weights[10000, 100, 1]/mlp-ep5066-loss89.426-val_loss81.310-lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 10000-n_inputs: 239-n_outputs: 4-n_hidden: []-n_mlp: [400, 400, 400]-n_samples2146176-weights[10000, 100, 1].h5'
    # 12.66, total training data in training
    # file_name = '/home/yezhizi/Documents/TianchiMetro/code/ckpt/lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 10000-n_inputs: 239-n_outputs: 4-n_hidden: []-n_mlp: [500, 500]-n_samples2146176-weights[1, 0, 0]/mlp-ep8229-loss88.305-val_loss81.065-lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 10000-n_inputs: 239-n_outputs: 4-n_hidden: []-n_mlp: [500, 500]-n_samples2146176-weights[1, 0, 0].h5'
    # 12.94
    # validation data -1:
    # file_name = '/home/yezhizi/Documents/TianchiMetro/code/ckpt/lr: 0.001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 2000-n_inputs: 243-n_outputs: 4-n_hidden: []-n_mlp: [400, 400, 400, 400, 400, 400]-n_samples2146176-weights[1, 0, 0]/mlp-ep889-loss85.024-val_loss74.545-lr: 0.001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 2000-n_inputs: 243-n_outputs: 4-n_hidden: []-n_mlp: [400, 400, 400, 400, 400, 400]-n_samples2146176-weights[1, 0, 0].h5'
    # 13.05
    file_name = '/home/yezhizi/Documents/TianchiMetro/code/ckpt/lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 2000-n_inputs: 243-n_outputs: 4-n_hidden: []-n_mlp: [400, 400, 400]-n_samples2146176-weights[1, 0, 0]/mlp-ep1411-loss76.752-val_loss70.924-lr: 0.0001-batch_size: 93312-l2_param: 0.01-dropout: 0.8-training_epochs: 2000-n_inputs: 243-n_outputs: 4-n_hidden: []-n_mlp: [400, 400, 400]-n_samples2146176-weights[1, 0, 0].h5'


    mlp.load_weights(file_name)
    # mlp = load_model(file_name)
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
    rcd += "file_name: " + str(file_name) + '\n'
    rcd += "time: " + str(end - start) + '\n' + '\n' + '\n'
    print(rcd)
    log_file = open(DATA_PATH_RESULT + "mlp_result", "a")
    log_file.write(rcd)
    log_file.close()


print("************************Finish the fine tuning******************************")
