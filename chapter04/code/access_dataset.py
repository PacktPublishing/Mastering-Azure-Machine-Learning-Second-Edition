import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
args = parser.parse_args()

print("Dataset path: {}".format(args.input))
print(os.listdir(args.input))