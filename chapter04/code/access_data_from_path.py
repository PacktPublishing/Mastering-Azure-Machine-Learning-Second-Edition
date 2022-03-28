import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
args = parser.parse_args()

df = pd.read_csv(args.input)
print(df.head())