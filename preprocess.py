import cv2
import numpy as np

datasetPath = '../witsss-dataset/'
video = '1'
videoPath = datasetPath + 'data/' + 'video' + video + '/'


# Image preprocess
def imagePreprocess(image, label):
  kernel = np.ones((3, 3),np.uint8)
  label[:, :, 1] = cv2.dilate(label[:, :, 1], kernel, iterations = 1)
  return image, label


# File preprocess
trainCount = -1
valCount = -1
for i in range(100, 2001, 100):
  image = cv2.imread(videoPath + 'video_' + video + '_frame_' + str(i) + '.jpg')
  label = cv2.imread(videoPath + 'video_' + video + '_frame_' + str(i) + '_label.png')
  print(videoPath + 'video_' + video + '_frame_' + str(i) + '.jpg')
  
  if i > 1601:
    target = 'val/'
    valCount += 1
    targetCount = valCount
  else:
    target = 'train/'
    trainCount += 1
    targetCount = trainCount

  image, label = imagePreprocess(image, label)

  cv2.imwrite('data/' + target + str(targetCount) + '.jpg', image)
  cv2.imwrite('data/' + target + str(targetCount) + '_label.png', label)
    