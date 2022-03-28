import argparse
from azureml.core import Dataset, Run

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
args = parser.parse_args()

run = Run.get_context()
ws = run.experiment.workspace

ds = Dataset.get_by_id(ws, id=args.input)
print(ds.to_pandas_dataframe().head())