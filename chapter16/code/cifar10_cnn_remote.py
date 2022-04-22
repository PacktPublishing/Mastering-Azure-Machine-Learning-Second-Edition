'''
#Train a simple deep CNN on the CIFAR10 small images dataset.

Source: https://raw.githubusercontent.com/keras-team/keras/master/examples/cifar10_cnn.py
'''

from __future__ import print_function
import tensorflow
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from azureml.core import Run
import os
import logging
from keras_azure_ml_cb import AzureMlKerasCallback

# define function to get the best value of a specific metric of all runs in the experiment
def get_metrics_from_exp(experiment, metric, status='Completed'):
    for run in Run.list(experiment, status=status):
        yield run.get_metrics().get(metric)

batch_size = 32
num_classes = 10
epochs = 10
num_predictions = 20

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

# Normalize data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# define model name and file locations
model_name = 'keras_cifar10_trained_model.h5'
model_output_dir = os.path.join(os.getcwd(), 'outputs')

# initiate RMSprop optimizer (https://keras.io/api/optimizers/rmsprop/)
opt = tensorflow.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

# define checkpoint function to only save the model after each epoch
# (decided based on the validation loss function) in the output file path
if not os.path.isdir(model_output_dir):
    os.makedirs(model_output_dir)
model_path = os.path.join(model_output_dir, model_name)
checkpoint_cb = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

# log output of the script (on debug level of the training iterations)
if not os.path.isdir('logs'):
    os.makedirs('logs')
logging.basicConfig(filename='logs/debug.log', filemode='w', level=logging.DEBUG)
logger_cb = CSVLogger('logs/training.log')

# define the loss function, optimizer and tracked metrics of the model training
# (https://keras.io/api/losses/probabilistic_losses/#categoricalcrossentropy-class)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# load the current run context to add metrics later
run = Run.get_context()

# create an Azure Machine Learning monitor callback
azureml_cb = AzureMlKerasCallback(run)

# train the model for a certain number of epochs
model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split = 0.2,
        shuffle=True,
        verbose=0,
        callbacks=[azureml_cb, checkpoint_cb, logger_cb])

# Load the best model
model = load_model(model_path)

# Score trained model
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
run.log('Test loss', scores[0])
print('Test accuracy:', scores[1])
run.log('Test accuracy', scores[1])

# Upload the model binary file(s) of the best model
run.upload_file(model_name, model_path)

# get the best accuracy out of every run before
best_test_acc = max(get_metrics_from_exp(run.experiment,'Test accuracy'), default = 0)

# Register the best model if it is better than in any previous model training
if scores[1] > best_test_acc:
    run.register_model(model_name, model_path=model_name, model_framework='TfKeras')