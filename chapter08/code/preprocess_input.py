import argparse
from azureml.data.dataset_factory import TabularDatasetFactory

import pandas as pd

from azureml.core import Run, Dataset, Datastore
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
args = parser.parse_args()


# Load data
# ---------------------------------------

# Get a dataset by id
dataset = Dataset.get_by_id(ws, id=args.input)

# Load a TabularDataset into pandas DataFrame
df = dataset.to_pandas_dataframe()


# Transform data
# ---------------------------------------
df = df.drop_duplicates()

y = df.pop('Survived')
X = df.drop(["PassengerId", "Name", "Ticket"], axis=1)

X = pd.get_dummies(X, columns=["Sex", "Embarked", "Cabin"], drop_first=True)
X = X.fillna(0)

X_train = StandardScaler().fit_transform(X.values)
y_train = y.values


# Store data
# ---------------------------------------

df = pd.DataFrame(data=X_train, columns=X.columns, index=X.index)
df['target'] = y_train

datastore = Datastore.get(ws, datastore_name="mldata")

ds_name = "titanic.train"
ds_path = "data/titanic/train"
dataset = TabularDatasetFactory.register_pandas_dataframe(df, target=(datastore, ds_path), name=ds_name)
dataset.register(ws, name=ds_name)
