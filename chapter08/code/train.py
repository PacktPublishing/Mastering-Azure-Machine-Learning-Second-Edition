import argparse

import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

from azureml.core import Run
from keras_azure_ml_cb import AzureMlKerasCallback


# Setup Run
# ---------------------------------------

# Load the current run and ws
run = Run.get_context()
ws = run.experiment.workspace


# Parse parameters
# ---------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--in-train", type=str)
parser.add_argument("--in-test", type=str)
parser.add_argument('--batch-size', type=int, dest='batch_size', default=50)
parser.add_argument('--epochs', type=int, dest='epochs', default=10)
parser.add_argument('--first-layer-neurons', type=int, dest='n_hidden_1', default=100)
parser.add_argument('--second-layer-neurons', type=int, dest='n_hidden_2', default=100)
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.01)
parser.add_argument('--momentum', type=float, dest='momentum', default=0.9)
args = parser.parse_args()


# Load train/test data
# ---------------------------------------

df_train = pd.read_csv(args.in_train)
df_test = pd.read_csv(args.in_test)

y_train = df_train.pop("target").values
X_train = df_train.values

y_test = df_test.pop("target").values
X_test = df_test.values


# Build model
# ---------------------------------------

model = Sequential()

model.add(Dense(args.n_hidden_1, activation='relu', input_dim=X_train.shape[1], kernel_initializer='uniform'))
model.add(Dropout(0.50))
model.add(Dense(args.n_hidden_2, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

sgd = SGD(lr = args.learning_rate, momentum = args.momentum)
model.compile(optimizer = sgd,  loss = 'binary_crossentropy',  metrics = ['accuracy'])


# Train model
# ---------------------------------------

# Create an Azure Machine Learning monitor callback
azureml_cb = AzureMlKerasCallback(run)

model.fit(X_train, y_train, batch_size = args.batch_size, epochs = args.epochs, validation_split = 0.1, callbacks=[azureml_cb])


# Evaluate model
# ---------------------------------------

scores = model.evaluate(X_test, y_test, batch_size=30)
run.log(model.metrics_names[0], float(scores[0]))
run.log(model.metrics_names[1], float(scores[1]))