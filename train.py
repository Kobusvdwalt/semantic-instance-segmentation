import numpy as np 
import os
import cv2
import random
import math

from tensorflow.keras import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback

from data import *
from customMetrics import recall
from customMetrics import precision
from customMetrics import mean_iou
from customMetrics import weightedBCE

from models import FPN

from callbacks import HistoryCheckpoint

model = FPN((None, None, 3))
#model.load_weights('unet.hdf5')
model.compile(optimizer = Adam(lr=1e-4), loss = weightedBCE, metrics = ['binary_accuracy', recall, precision])

train = StudentDataGenerator(2, source='train')
validation = StudentDataGenerator(2, source='val')

history_checkpoint = HistoryCheckpoint(model)
model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(train, steps_per_epoch=100, epochs=20, callbacks=[model_checkpoint, history_checkpoint], max_queue_size=100, workers=6, validation_data=validation, validation_steps=100, shuffle=True)