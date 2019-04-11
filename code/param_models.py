# -*- coding: utf-8 -*-

# train data and valid data are divided and selected in different way between model trained in Test A and Test B
# Be aware of this

def param_model_1228():
    lr = 0.0001
    FEAT_LEN = [144, 81, 2, 4, 7, 1]
    HEADER_FEAT = ['block', 'stationID', 'status', 'payType', 'weekDay', 'flow']
    HEADER_HIS = ['yesterday']

    n_inputs = sum(FEAT_LEN)
    n_outputs = 3
    n_mlp = [400, 400, 400]
    return FEAT_LEN, HEADER_FEAT, HEADER_HIS, n_inputs, n_outputs, n_mlp, lr


def param_model_1250():
    lr = 0.0001
    FEAT_LEN = [144, 81, 2, 4, 7, 1]
    HEADER_FEAT = ['block', 'stationID', 'status', 'payType', 'weekDay', 'flow']
    HEADER_HIS = ['yesterday']

    n_inputs = sum(FEAT_LEN)
    n_outputs = 4
    n_mlp = [400, 400, 400]
    return FEAT_LEN, HEADER_FEAT, HEADER_HIS, n_inputs, n_outputs, n_mlp, lr


def param_model_1250_right():
    lr = 0.0001
    FEAT_LEN = [144, 81, 2, 4, 7, 1]
    HEADER_FEAT = ['block', 'stationID', 'status', 'payType', 'weekDay', 'flow']
    HEADER_HIS = ['yesterday']

    n_inputs = sum(FEAT_LEN)
    n_outputs = 3
    n_mlp = [400, 400, 400]
    return FEAT_LEN, HEADER_FEAT, HEADER_HIS, n_inputs, n_outputs, n_mlp, lr


def param_model_1266():
    lr = 0.0001
    FEAT_LEN = [144, 81, 2, 4, 7, 1]
    HEADER_FEAT = ['block', 'stationID', 'status', 'payType', 'weekDay', 'flow']
    HEADER_HIS = ['yesterday']

    n_inputs = sum(FEAT_LEN)
    n_outputs = 4
    n_mlp = [500, 500]
    return FEAT_LEN, HEADER_FEAT, HEADER_HIS, n_inputs, n_outputs, n_mlp, lr


def param_model_1294():
    lr = 0.001
    FEAT_LEN = [144, 81, 2, 4, 7, 2, 1, 1, 1]
    HEADER_FEAT = ['block', 'stationID', 'status', 'payType', 'weekDay',  'holiday', 'flow']
    HEADER_HIS = ['yesterday_normal', 'lastWeek', 'averWeek']

    n_inputs = sum(FEAT_LEN)
    n_outputs = 4
    n_mlp = [400, 400, 400, 400, 400, 400]
    return FEAT_LEN, HEADER_FEAT, HEADER_HIS, n_inputs, n_outputs, n_mlp, lr


def param_model_1305():
    lr = 0.0001
    FEAT_LEN = [144, 81, 2, 4, 7, 2, 1, 1, 1]
    HEADER_FEAT = ['block', 'stationID', 'status', 'payType', 'weekDay',  'holiday', 'flow']
    HEADER_HIS = ['yesterday_normal', 'lastWeek', 'averWeek']

    n_inputs = sum(FEAT_LEN)
    n_outputs = 4
    n_mlp = [400, 400, 400]
    return FEAT_LEN, HEADER_FEAT, HEADER_HIS, n_inputs, n_outputs, n_mlp, lr

