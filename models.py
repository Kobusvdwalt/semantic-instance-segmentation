from tensorflow import keras
from tensorflow.keras.layers import *

def FPN(img_shape):
    lrelu = lambda x: keras.activations.relu(x, alpha=0.1)


    # In
    inputs = keras.Input(img_shape) # 512

    # Level 1
    conv1 = AveragePooling2D(pool_size=(2, 2))(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation = lrelu, padding = 'same')(conv1)
    conv1 = Dropout(0.2)(conv1)
    
    # Level 2
    conv2 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = lrelu, padding = 'same')(conv2)
    conv2 = Conv2D(64, 3, activation = lrelu, padding = 'same')(conv2)
    conv2 = Dropout(0.2)(conv2)

    # Level 3
    conv3 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = lrelu, padding = 'same')(conv3)
    conv3 = Conv2D(128, 3, activation = lrelu, padding = 'same')(conv3)
    conv3 = Dropout(0.2)(conv3)

    # Level 4    
    conv4 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = lrelu, padding = 'same')(conv4)
    conv4 = Conv2D(256, 3, activation = lrelu, padding = 'same')(conv4)
    conv4 = Dropout(0.2)(conv4)

    conv2_o = UpSampling2D(size = (2, 2))(conv2)
    
    conv3_o = UpSampling2D(size = (2, 2))(conv3)
    conv3_o = Conv2D(64, 3, activation = lrelu, padding = 'same')(conv3_o)
    conv3_o = UpSampling2D(size = (2, 2))(conv3_o)

    conv4_o = UpSampling2D(size = (2, 2))(conv4)
    conv4_o = Conv2D(128, 3, activation = lrelu, padding = 'same')(conv4_o)
    conv4_o = UpSampling2D(size = (2, 2))(conv4_o)
    conv4_o = Conv2D(64, 3, activation = lrelu, padding = 'same')(conv4_o)
    conv4_o = UpSampling2D(size = (2, 2))(conv4_o)

    out = concatenate([conv1, conv2_o, conv3_o, conv4_o], axis = 3)
    out = Conv2D(32, 3, activation = lrelu, padding = 'same')(out)
    out = UpSampling2D(size = (2, 2))(out)
    out = Conv2D(16, 3, activation = lrelu, padding = 'same')(out)
    out = Conv2D(3, 1, activation = 'sigmoid', padding = 'same')(out)

    model = keras.Model(inputs = inputs, outputs = out)
    model.summary()

    return model
