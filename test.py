import numpy as np 
import os
import cv2
import random
import math
import pickle

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


model = FPN((None, None, 3))
model.load_weights('unet.hdf5')

validation = StudentDataGenerator(2, source='val')

for i in range(0, 100):
  images, labels = validation.__getitem__(i%2)
  result = model.predict_on_batch(images)

  for b in range(0, len(result)):
    cv2.imshow('image', images[b])
    cv2.imshow('result', result[b])
    cv2.waitKey(0)
  