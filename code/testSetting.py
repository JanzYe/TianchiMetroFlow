import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--state', type=str)  # B or C
args = parser.parse_args()
state = args.state
print(state)

f = open('./testSetting.txt', 'w')

f.write(state)
f.close()
