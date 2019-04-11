# -*- coding: utf-8 -*-
import pandas
from constants import *


def mergeData():
    feats = {}
    for header in HEADER_FEAT:
        feats[header] = []
    for d in range(DAYS):
        date = DATE_START[:-2] + str(d+1).zfill(2)
        data = pandas.read_csv((DATA_PATH_FEAT + 'features_%s.csv') % date)

        for header in HEADER_FEAT:
            feats[header].extend(data[header].values.tolist())

    feats = pandas.DataFrame(feats)
    feats.to_csv((DATA_PATH_FEAT + 'features.csv'), index=False)


if __name__ == '__main__':
    mergeData()


