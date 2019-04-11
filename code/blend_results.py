# -*- coding: utf-8 -*-

import pandas as pd
import datetime

from constants import *

# #Test A
# files = [
#     # 12.28
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testA/testA_submit_2019-01-29_2019-03-29 09:15:46.csv',
#     # 12.50
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testA/testA_submit_2019-01-29_2019-03-30 10:34:34.csv',
#     # 12.66
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testA/testA_submit_2019-01-29_2019-03-29 21:38:58.csv',
#     # 12.94
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testA/testA_submit_2019-01-29_2019-03-26 17:06:37.csv',
#     # 13.05
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testA/testA_submit_2019-01-29_2019-03-27 21:53:04.csv',
# ]

#Test B
# files = [
#     # 12.28
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 00:16:45.csv',
#     # 12.50
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 00:31:26.csv',
#     # 12.66
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 00:34:40.csv',
#     # 12.94
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 00:47:25.csv',
#     # 13.05
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 00:49:54.csv',
# ]

# #Test B, total samples
# files = [
#     # 12.28 -> 12.54 -> 12.53
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 09:45:08.csv',
#     # 12.50 ->
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 18:37:00.csv',
#     # 12.66 ->
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 18:40:37.csv',
#     # 12.94 ->
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 21:44:34.csv',
#     # 13.05 ->
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 21:47:27.csv',
# ]

#Test B, ten mix
# files = [
#     # 12.28
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 00:16:45.csv',
#     # 12.50
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 00:31:26.csv',
#     # 12.66
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 00:34:40.csv',
#     # 12.94
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 00:47:25.csv',
#     # 13.05
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 00:49:54.csv',
#
#     # 12.28 -> 12.54 -> 12.53
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 09:45:08.csv',
#     # 12.50 ->
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 18:37:00.csv',
#     # 12.66 ->
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 18:40:37.csv',
#     # 12.94 ->
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 21:44:34.csv',
#     # 13.05 ->
#     '/home/yezhizi/Documents/TianchiMetro/Metro_testB/testB_submit_2019-01-27_2019-03-31 21:47:27.csv',
# ]


def blendResults(files):
    # weights = [2, 1, 1, 1, 1]
    # weights = [1, 1, 1, 1, 1]
    # weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    weights = [1] * len(files)

    in_preds = 0
    out_preds = 0
    for idx, file in enumerate(files):
        # if not file.endswith('.csv'):
        #     continue
        data = pd.read_csv(DATA_PATH_PRED+file)
        in_preds = in_preds + weights[idx] * data[HEADER_IN]
        out_preds = out_preds + weights[idx] * data[HEADER_OUT]

    in_preds = in_preds / sum(weights)
    out_preds = out_preds / sum(weights)

    submit = pd.read_csv(DATA_PATH_TEST + SUBMIT_FILE)

    submit[HEADER_IN] = in_preds
    submit[HEADER_OUT] = out_preds

    file_name = SUBMIT_RESULT[:-4] + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv'

    submit.to_csv(file_name, index=False)


if __name__ == '__main__':
    files = os.listdir(DATA_PATH_PRED)
    blendResults(files)

