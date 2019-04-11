from constants import *
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--which', type=str)
args = parser.parse_args()

if args.which == PRED:
    files = os.listdir(DATA_PATH_PRED)
    for file in files:
        os.remove(DATA_PATH_PRED+file)

