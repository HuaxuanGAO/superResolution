
# coding: utf-8
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, LeakyReLU, Conv2DTranspose, Concatenate, Lambda
from keras.optimizers import Adam
from keras import losses
from keras.models import Model
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
import numpy as np

def frac_max_pool(x):
    return tf.nn.fractional_max_pool(x,[1,4/3,4/3,1])[0]

def conv_relu(conv_in, num_filter):
    conv1 = Conv2D(num_filter, (3, 3), strides=(1, 1), padding='same')(conv_in)
    return LeakyReLU()(conv1)

def conv_relu_block(conv_in, num_filter, pool=True):
    conv1 = conv_relu(conv_in, num_filter)
    conv2 = conv_relu(conv1, num_filter)
    if pool:
        pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)
        return pool1
    else:
        return conv2
    
def upsample_and_concat(x1, x2, out_channel):
    conv_trans = Conv2DTranspose(out_channel, kernel_size = (2,2), strides=(2, 2), padding='same', 
                                 activation=None)(x1)
    concat = Concatenate(3)([conv_trans, x2])
    return concat

inputs = Input(shape=(128, 128, 35), dtype='float32', name='input')

conv1 = conv_relu_block(inputs, 32)

conv2 = conv_relu_block(conv1, 64)

conv3 = conv_relu_block(conv2, 128)

conv4 = conv_relu_block(conv3, 256)

conv5 = conv_relu_block(conv4, 512)

up6 = upsample_and_concat(conv5, conv4, 512)
conv6 = conv_relu_block(up6, 256, pool=False)

up7 = upsample_and_concat(conv6, conv3, 256)
conv7 = conv_relu_block(up7, 128, pool=False)

up8 = upsample_and_concat(conv7, conv2, 128)
conv8 = conv_relu_block(up8, 64, pool=False)

up9 = upsample_and_concat(conv8, conv1, 64)
conv9 = conv_relu_block(up9, 32, pool=False)

up10 = upsample_and_concat(conv9, inputs, 32)
conv10 = conv_relu_block(up10, 16, pool=False)

up11 = Conv2DTranspose(8, kernel_size = (2,2), strides=(2, 2), padding='same', 
                                 activation=None)(conv10)

up12 = Conv2DTranspose(3, kernel_size = (2,2), strides=(2, 2), padding='same', 
                                 activation=None)(up11)

up13 = Lambda(frac_max_pool)(up12)

model = Model(inputs=inputs, outputs=up13)
adam = Adam(lr=0.01)
model.compile(loss=losses.mean_squared_error, optimizer=adam)

# model.summary()

train_data_np = np.load('./train_data_35_NIR.npy')
HR_data = np.load('./HR_data_NIR.npy')
print(train_data_np.shape)
print(HR_data.shape)

checkpoint = ModelCheckpoint('unet_NIR', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

model.fit(x = train_data_np, y = HR_data, validation_split=0.1, epochs=50, callbacks=[checkpoint])

