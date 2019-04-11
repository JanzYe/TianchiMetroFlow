# -*- coding: utf-8 -*-

import os
import datetime
import pandas
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import traceback
from constants import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str)
args = parser.parse_args()
mode = args.mode


def getFilesList(path):
    dir_list = os.listdir(path)
    files_list = []
    for name in dir_list:
        if name[-4:] == '.csv':
            files_list.append(name)
    return sorted(files_list)


def getWeekDay(date_str):
    date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    return date.weekday()


def timeToMinutes(time):
    # transfer hh:MM:SS to the amount of minutes in a day
    strings = time.split(':')
    minutes = 60 * int(strings[0]) + int(strings[1])
    return minutes


def toTimeBlock(data_time):
    num_flow = np.zeros(NUM_TIME_BLOCK)

    for time in data_time:
        minutes = timeToMinutes(time[TIME_POS_START:])
        block = int(np.floor(minutes / SIZE_TIME_BLOCK))
        num_flow[block] = num_flow[block] + 1

    return num_flow


def toTimeBlockFast(data_time):
    num_flow = np.zeros(NUM_TIME_BLOCK)

    strings = ' '.join(data_time)
    str_arr = np.array(strings.split(' '))
    pos_time = np.arange(1, len(str_arr), 2)
    str_arr = str_arr[pos_time]
    strings = ':'.join(str_arr.tolist())
    str_arr = np.array(strings.split(':'))

    pos_hour = np.arange(0, len(str_arr), 3)
    pos_minute = np.arange(1, len(str_arr), 3, dtype=np.int)
    hours = np.array(str_arr[pos_hour], dtype=np.int)
    minutes = np.array(str_arr[pos_minute], dtype=np.int)

    toMinutes = hours * 60 + minutes
    toBlock = np.array(np.floor(toMinutes / SIZE_TIME_BLOCK), dtype=np.int)

    for t in range(NUM_TIME_BLOCK):
        num_flow[t] = np.sum(toBlock == t)

    return num_flow


def generateXTicks(x):
    ticks = []
    for minu in x:
        hour = str(int(np.floor(minu / 60))).zfill(2)
        minutes = str(int(minu % 60)).zfill(2)
        tick = '%s:%s' % (hour, minutes)
        ticks.append(tick)
    return ticks


def setTicker(sub_fig, fontsize):
    x = np.arange(0, NUM_TIME_BLOCK, 10).tolist()
    sub_fig.set_xticks(x)

    sub_fig.tick_params(axis='both', width=2, length=10, pad=10)

    for tick in sub_fig.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    for tick in sub_fig.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)


def drawStationInOut(num_in, num_out, img_name):
    print('drawing %s ......' % img_name)
    fig = plt.figure(figsize=(20, 18))
    sub_fig = fig.add_subplot(111)

    x = np.arange(0, NUM_TIME_BLOCK*10, 10)
    ticks = generateXTicks(x)
    sub_fig.plot(ticks, num_in, c=COLORS[0], label='flow_in')
    sub_fig.plot(ticks, num_out, c=COLORS[1], label='flow_out')
    plt.legend(loc='upper right', fontsize=20)
    plt.title(img_name[:-4], fontsize=20)
    plt.grid(axis='x')

    setTicker(sub_fig, 20)

    plt.savefig(DATA_PATH_IMG + img_name)
    plt.close('all')


def analyze(path, day=0):
    try:

        if mode == TRAIN:
            print(DATA_PATH_FEAT + 'features_'+DATE_START+'.csv')
            if os.path.exists(DATA_PATH_FEAT + 'features_'+DATE_START+'.csv'):
                print('Training features exist')
                return
            files_list = getFilesList(path)

            # output = open(DATA_PATH_FEAT + 'features.csv', 'w')
            # output.write(','.join(HEADER_FEAT) + '\n')
        else:
            print(DATA_PATH_TEST + 'features_yesterday.csv')
            if os.path.exists(DATA_PATH_TEST + 'features_yesterday.csv'):
                print('Test features exist')
                return
            files_list = [YESTERDAY_TEST]

            # output = open(path + 'features_yesterday.csv', 'w')
            # output.write(','.join(HEADER_FEAT) + '\n')

        feats = {}
        for header in HEADER_FEAT:
            feats[header] = []
        # print(files_list)
        for file_name in [files_list[day]]:
            print('reading %s ......' % file_name)
            data = pandas.read_csv(path+file_name)

            data_station = data[HEADER_STATION_ID].values


            data_status = data[HEADER_STATUS].values
            ind_out_in = []
            ind_out_in.append(data_status == STATUS_OUT)
            ind_out_in.append(data_status == STATUS_IN)

            data_pay_type = data[HEADER_PAY_TYPE].values
            ind_type = []
            for t in range(NUM_TYPES):
                ind_type.append(data_pay_type == t)

            data_time = data[HEADER_TIME].values
            # data_line_id = data[HEADER_LINE_ID].values

            date = data_time[0][:DATE_POS_END]
            week_day = getWeekDay(date)
            holiday = 0
            if date in HOLIDAY:
                holiday = 1

            flow_out_in_one_day = np.zeros((2, NUM_TIME_BLOCK))
            for station_id in range(NUM_STATIONS):

                ind_station = data_station == station_id
                print('station %d, flow: %d' % (station_id, sum(ind_station)))
                flow_out_in = np.zeros((2, NUM_TIME_BLOCK))

                print('generating features station %d......' % station_id)

                for i in range(len(ind_out_in)):
                    ind_s_o = ind_station & ind_out_in[i]
                    for t in range(NUM_TYPES):
                        ind_s_o_t = ind_s_o & ind_type[t]
                        data_time_ind = data_time[ind_s_o_t]
                        # data_line_id_ind = data_line_id[ind_s_o_t]

                        flow_time_block = toTimeBlock(data_time_ind)
                        # flow_time_block = toTimeBlockFast(data_time_ind)
                        flow_out_in[i] = flow_out_in[i] + flow_time_block
                        flow_out_in_one_day[i] = flow_out_in_one_day[i] + flow_time_block

                        # ['block', 'stationID', 'status', 'payType', 'weekDay', 'holiday']
                        # for b in range(len(flow_time_block)):
                        #     output.write(str(b) + ',' + str(station_id) + ',' + str(i) + ',' + str(t) + ',' +
                        #                  str(week_day) + ','+str(holiday) + ',' + str(flow_time_block[b])+'\n')

                        feats[HEADER_FEAT[0]].extend(np.arange(NUM_TIME_BLOCK))
                        feats[HEADER_FEAT[1]].extend(np.ones((NUM_TIME_BLOCK), dtype=np.int) * station_id)
                        feats[HEADER_FEAT[2]].extend(np.ones((NUM_TIME_BLOCK), dtype=np.int) * i)
                        feats[HEADER_FEAT[3]].extend(np.ones((NUM_TIME_BLOCK), dtype=np.int) * t)
                        feats[HEADER_FEAT[4]].extend(np.ones((NUM_TIME_BLOCK), dtype=np.int) * week_day)
                        feats[HEADER_FEAT[5]].extend(np.ones((NUM_TIME_BLOCK), dtype=np.int) * holiday)
                        feats[HEADER_FEAT[6]].extend(flow_time_block.tolist())


                # img_name = '%s_%d_in_out_%s.png' % (file_name[:-4], station_id, WEEK_DAY_DICT[week_day])
                # drawStationInOut(flow_out_in[1], flow_out_in[0], img_name)

                # break

            # img_name = '%s_all_in_out_%s.png' % (file_name[:-4], WEEK_DAY_DICT[week_day])
            # drawStationInOut(flow_out_in_one_day[1], flow_out_in_one_day[0], img_name)
            # break

        feats = pandas.DataFrame(feats)
        if mode == TRAIN:
            feats.to_csv((DATA_PATH_FEAT + 'features_%s.csv') % date, index=False)
        elif mode == PRED:
            feats.to_csv(DATA_PATH_TEST + 'features_yesterday.csv', index=False)
        elif mode == VALID:
            feats.to_csv(DATA_PATH_TEST + 'features_valid.csv', index=False)


    except Exception as err:
        print(traceback.print_exc())


if __name__ == '__main__':
    if mode == TRAIN:
        processes = 25
        pool = multiprocessing.Pool(processes=processes)
        for i in range(DAYS):
            pool.apply_async(analyze, (DATA_PATH_ORI, i))

        pool.close()
        pool.join()

    else:
        analyze(DATA_PATH_TEST)




