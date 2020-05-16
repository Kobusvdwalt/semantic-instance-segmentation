import pickle
from tensorflow.keras.callbacks import Callback
# Predicts on batch while training to see how the network is evolving
class HistoryCheckpoint(Callback):
  def __init__(self, model):
    self.model = model
  def on_epoch_begin(self, batch, logs={}):
    with open('trainHistoryDict', 'wb') as file_pi:
      pickle.dump(self.model.history.history, file_pi)