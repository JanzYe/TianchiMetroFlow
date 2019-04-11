# -*- coding:utf-8 -*-

from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import GridSearchCV
import datetime
import joblib
import pandas

from constants import *

title = 'nn_no-cl-im-comb_t15'

input_file = DATA_PATH_FEAT + 'features_his.csv'
days_train = 23


def listToArray(data_list):
    data = data_list[0]
    for d in data_list[1:]:
        if d.ndim < 2:
            d = np.reshape(d, (-1, 1))
        data = np.hstack([data, d])
    return data


def divideTrainValid(days_train):
    data = pandas.read_csv(input_file)
    # ['block', 'stationID', 'status', 'payType', 'weekDay', 'holiday', 'flow']
    # ['yesterday', 'lastWeek', 'averWeek', 'averWeekOutIn']
    num_train = DATA_A_DAY * days_train
    num_valid = DATA_A_DAY * (DAYS - days_train)

    data_y = data[HEADER_FEAT[-1]].values
    max_label = np.max(data_y)
    train_y = data_y[:num_train]
    valid_y = data_y[num_train:]

    train_x = []
    valid_x = []
    encoder = OneHotEncoder()

    idx_feat = 0
    for feat in HEADER_FEAT[:-1]:
        data_feat = encoder.fit_transform(np.reshape(data[feat].values, (-1, 1)))
        data_feat = data_feat.toarray()
        train_x.append(data_feat[:num_train])
        valid_x.append(data_feat[num_train:])
        idx_feat += 1
    for feat in HEADER_HIS:
        data_feat = data[feat].values
        train_x.append(data_feat[:num_train])
        valid_x.append(data_feat[num_train:])
        idx_feat += 1

    train_x = listToArray(train_x)
    valid_x = listToArray(valid_x)

    return train_x, train_y, valid_x, valid_y



def dtr(days_train):
    begin = datetime.datetime.now()
    grid = False
    # train_x = read_coo_mtx(training_X_file)
    # train_y = np.loadtxt(open(training_Y_file), dtype=int)
    # test_x = read_coo_mtx(test_X_file)
    # test_y = np.loadtxt(open(test_Y_file), dtype=int)

    train_x, train_y, valid_x, valid_y = divideTrainValid(days_train)
    # print( train_x.shape, test_x.shape
    print( "Loading data completed.")
    print( "Read time: " + str(datetime.datetime.now() - begin))

    classifier = DecisionTreeRegressor(min_samples_leaf=NUM_TYPES)
    # if grid:
    #     param_grid = {'C': [1, 5, 10]}
    #     grid = GridSearchCV(estimator=classifier, scoring='roc_auc', param_grid=param_grid)
    #     grid.fit(train_x, train_y)
    #     print( "Training completed."
    #     print( grid.cv_results_
    #     print( grid.best_estimator_

    if not grid:
        classifier.fit(train_x, train_y)
        # cross_val_score(classifier, training_x, training_y, cv=10)
        # print( "Cross validation completed."
        # joblib.dump(classifier, "new_basic_lr" + ".pkl", compress=3)      #加个3，是压缩，一般用这个
        # classifier = joblib.load("new_basic_lr.pkl")

        y_pred = classifier.predict(valid_x)

        accuracy = metrics.mean_squared_error(valid_y, y_pred)
        print("mse: " + str(accuracy))

        end = datetime.datetime.now()
        day = datetime.date.today()
        np.savetxt(open(DATA_PATH_RESULT+title+"_lr_pred_"+str(day), "w"), accuracy, fmt='%.5f')

        rcd = str(end) + '\n'
        rcd += "lr: "+title+" 130" + '\n'
        rcd += str(classifier.get_params()) + '\n'
        rcd += "mse: " + str(accuracy) + '\n'
        rcd += "time: " + str(end - begin) + '\n' + '\n' + '\n'
        print( rcd)
        log_file = open(DATA_PATH_RESULT+"dtr_result", "a")
        log_file.write(rcd)
        log_file.close()


if __name__ == '__main__':
    dtr(days_train)

    # test_x, test_y = datasets.load_svmlight_file(test_X_file)
    # np.savetxt(open(constants.project_path + "result/10.7_lr_result", "w"), test_y, fmt='%.5f')
    # print( test_x.shape, test_y.shape
    # test_y = np.loadtxt(open(test_Y_file), dtype=int)
    # pred_y_file = "D:/Dvlp_workspace/SAD/libfm-1.40.windows/output2.libfm"
    # pred_y = np.loadtxt(open(pred_y_file))
    # print( test_y[:10], pred_y[:10]
    # auc_test = metrics.roc_auc_score(test_y, pred_y)
    # print( auc_test