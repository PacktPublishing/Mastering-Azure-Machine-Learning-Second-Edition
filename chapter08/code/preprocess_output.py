import argparse
import os

from sklearn.model_selection import train_test_split
import pandas as pd

from azureml.core import Run, Dataset
from sklearn.preprocessing import StandardScaler


# Setup Run
# ---------------------------------------

# Load the current run and ws
run = Run.get_context()
ws = run.experiment.workspace


# Parse parameters
# ---------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
parser.add_argument("--out-train", type=str)
parser.add_argument("--out-test", type=str)
parser.add_argument("--test-size", type=float, default=0.25)
args = parser.parse_args()


# Load data
# ---------------------------------------

# Get a dataset by id
dataset = Dataset.get_by_id(ws, id=args.input)

# Load a TabularDataset into pandas DataFrame
df = dataset.to_pandas_dataframe()


# Configure outptus
# ---------------------------------------

out_train = args.out_train
os.makedirs(os.path.dirname(out_train), exist_ok=True)

out_test = args.out_test
os.makedirs(os.path.dirname(out_test), exist_ok=True)


# Transform data
# ---------------------------------------
df = df.drop_duplicates()

y = df.pop('Survived')
X = df.drop(["PassengerId", "Name", "Ticket"], axis=1)

X = pd.get_dummies(X, columns=["Sex", "Embarked", "Cabin"], drop_first=True)
X = X.fillna(0)

X_t = StandardScaler().fit_transform(X.values)
y_t = y.values


# Store data
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, test_size=args.test_size)

idx_train = [i for i in range(y_train.shape[0])]
df_train = pd.DataFrame(data=X_train, index=idx_train, columns=X.columns)
df_train['target'] = y_train

idx_test = [i for i in range(y_test.shape[0])]
df_test = pd.DataFrame(data=X_test, columns=X.columns)
df_test['target'] = y_test

df_train.to_csv(out_train)
df_test.to_csv(out_test)