import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Conv2DTranspose, Activation
from tensorflow.keras.optimizers import Adam


def generator_model_create():
    inputs = Input(shape=(256, 256, 3), name='input')
    conv2d_1 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', name='conv2d_1')(inputs)
    relu_1 = Activation('relu', name='relu_1')(conv2d_1)

    conv2d_2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv2d_2')(relu_1)
    batch_norm_1 = BatchNormalization(name='batch_norm_1')(conv2d_2)
    relu_2 = Activation('relu', name='relu_2')(batch_norm_1)

    conv2d_3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv2d_3')(relu_2)
    batch_norm_2 = BatchNormalization(name='batch_norm_2')(conv2d_3)
    relu_3 = Activation('relu', name='relu_3')(batch_norm_2)

    deconv2d_1 = Conv2DTranspose(3, kernel_size=(4, 4), strides=(2, 2), padding='same', name='deconv2d_1')(relu_3)
    tanh_1 = Activation('tanh', name='tanh_1')(deconv2d_1)

    model = Model(inputs=inputs, outputs=tanh_1)

    return model


generator = generator_model_create()

adam_opt = Adam(learning_rate=0.0002, beta_1=0.5)
model_gen.compile(optimizer=adam_opt)

