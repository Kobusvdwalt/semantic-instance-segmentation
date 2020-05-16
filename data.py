import numpy as np 
import os
import glob
from tensorflow.keras.utils import Sequence
import math
import random
import cv2
import albumentations

class StudentDataGenerator(Sequence):
    def __init__(self, batch_size, source = 'train'):
        # Augmentation for classroom dataset
        self.batch_size = batch_size
        self.source = source

        self.sizeX = 512
        self.sizeY = 512
        
        self.composeAugmentation()

    def composeAugmentation(self):
        if self.source =='train':
            self.total = 10
            self.augment = albumentations.Compose(
                [
                    albumentations.Rotate(3, always_apply=True),
                    albumentations.RandomSizedCrop((self.sizeY//2, 700), self.sizeY, self.sizeX, 1, always_apply=True),             
                    albumentations.HorizontalFlip(),
                    #albumentations.GridDistortion(always_apply=False),                
                    #albumentations.IAAAffine(rotate=2, shear=5, always_apply=False),
                    
                    #albumentations.OpticalDistortion(),
                    albumentations.ElasticTransform(alpha=64, sigma=24, always_apply=False, alpha_affine=0),
                    albumentations.RandomBrightnessContrast(0.1, 0.1, always_apply=False),
                    #albumentations.Blur(always_apply=False)
                ])
        else:
            self.total = 4
            self.augment = albumentations.Compose(
                [
                    albumentations.RandomSizedCrop((self.sizeY, self.sizeY), self.sizeY, self.sizeX, 1, always_apply=True),
                ])
        

    def setSize(self, x, y):
        self.sizeX = x
        self.sizeY = y
        self.composeAugmentation()

    def __len__(self):
        return math.ceil(self.total / self.batch_size)

    def __getitem__(self, idx):
        imageResult = np.zeros((self.batch_size, self.sizeY, self.sizeX, 3))
        labelResult = np.zeros((self.batch_size, self.sizeY, self.sizeX, 3))
        for b in range(0, self.batch_size):
            sample = random.randint(0, self.total-1)
            image = cv2.imread('data/' + self.source +'/'+ str(sample) + '.jpg')
            label = cv2.imread('data/' + self.source +'/'+ str(sample) + '_label.png')

            et = self.augment(image=image, mask=label)
            image = et['image']
            label = et['mask']

            labelResult[b] = label / 255.0
            imageResult[b] = image / 255.0

        return (imageResult, labelResult)
'''
g = StudentDataGenerator(2, source='train')

for i in range(0, 100):
    images, labels = g.__getitem__(i % 2)
    for b in range(0, len(images)):
        cv2.imshow('image', images[b])
        cv2.imshow('label', labels[b])
        cv2.waitKey(0)

'''