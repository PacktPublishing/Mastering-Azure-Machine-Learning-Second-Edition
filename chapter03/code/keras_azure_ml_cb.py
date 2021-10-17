from tensorflow.keras.callbacks import Callback
import numpy as np

class AzureMlKerasCallback(Callback):

    def __init__(self, run):
        super(AzureMlKerasCallback, self).__init__()
        self.run = run

    # Keras calls this at the end of an epoch 
    def on_epoch_end(self, epoch, logs=None):
       
        # logs is filled by Keras in the following format at the end of an epoch
        # {loss:'0.1233', accuracy: '0.5945', val_loss:'0.5354', val_accuracy:'0.2344'}
        # the metrics here: loss defined by default, accuracy defined by the model.compile() function
        # loss, accuracy is evaluated with the training data set
        # val_loss, val_accuracy is evaluated with the validation set (defined in the model.fit() function)
        logs = logs or {}

        # add tracked metrics to the run logging
        for metric_name, metric_val in logs.items():
            if isinstance(metric_val, (np.ndarray, np.generic)):
                self.run.log_list(metric_name, metric_val.tolist())
            else:
                self.run.log(metric_name, metric_val)
