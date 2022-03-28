import argparse

import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

from azureml.core import Run, Dataset
from keras_azure_ml_cb import AzureMlKerasCallback
from sklearn.preprocessing import StandardScaler

# Setup Run
# ---------------------------------------

# Load the current run and ws
run = Run.get_context()
ws = run.experiment.workspace


# Parse parameters
# ---------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, dest='batch_size', default=50)
parser.add_argument('--epochs', type=int, dest='epochs', default=10)
parser.add_argument('--first-layer-neurons', type=int, dest='n_hidden_1', default=100)
parser.add_argument('--second-layer-neurons', type=int, dest='n_hidden_2', default=100)
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.01)
parser.add_argument('--momentum', type=float, dest='momentum', default=0.9)
args = parser.parse_args()

# Loading data
# ---------------------------------------

# Get a dataset by id
dataset = Dataset.get_by_name(ws, name="titanic")

# Load a TabularDataset into pandas DataFrame
df = dataset.to_pandas_dataframe()
df = df.drop_duplicates()

y = df.pop('Survived')
X = df.drop(["PassengerId", "Name", "Ticket"], axis=1)

X = pd.get_dummies(X, columns=["Sex", "Embarked", "Cabin"], drop_first=True)
X = X.fillna(0)

model = Sequential()

model.add(Dense(args.n_hidden_1, activation='relu', input_dim=X.shape[1], kernel_initializer='uniform'))
model.add(Dropout(0.50))
model.add(Dense(args.n_hidden_2, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

sgd = SGD(lr = args.learning_rate, momentum = args.momentum)
model.compile(optimizer = sgd,  loss = 'binary_crossentropy',  metrics = ['accuracy'])

# Create an Azure Machine Learning monitor callback
azureml_cb = AzureMlKerasCallback(run)

X_train = StandardScaler().fit_transform(X.values)
y_train = y.values

model.fit(X_train, y_train, batch_size = args.batch_size, epochs = args.epochs, validation_split = 0.1, callbacks=[azureml_cb])

scores = model.evaluate(X_train, y_train, batch_size=30)
run.log(model.metrics_names[0], float(scores[0]))
run.log(model.metrics_names[1], float(scores[1]))