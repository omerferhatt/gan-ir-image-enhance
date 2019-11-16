import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, Input, Conv2D, BatchNormalization, Conv2DTranspose, Activation, LeakyReLU
from tensorflow.keras.optimizers import Adam

batch_size = 16
steps_per_epoch = 3750
epochs = 10

opt = Adam(learning_rate=0.0002, beta_1=0.5)


def generator_model_create():
    inputs = Input(shape=(256, 256, 3), name='Input')
    conv2d_1 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', name='2DConv1')(inputs)
    relu_1 = Activation('relu', name='relu1')(conv2d_1)

    conv2d_2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='2DConv2')(relu_1)
    batch_norm_1 = BatchNormalization(name='Batch_Norm1')(conv2d_2)
    relu_2 = Activation('relu', name='relu2')(batch_norm_1)

    conv2d_3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='2D_Conv3')(relu_2)
    batch_norm_2 = BatchNormalization(name='Batch_Norm2')(conv2d_3)
    relu_3 = Activation('relu', name='relu3')(batch_norm_2)

    skip = Add(name='Skip_Con')([relu_3, relu_1])

    deconv2d_1 = Conv2DTranspose(3, kernel_size=(4, 4), strides=(2, 2), padding='same', name='2D_Conv_Trans')(skip)
    tanh = Activation('tanh', name='tanh')(deconv2d_1)

    model = Model(inputs=inputs, outputs=tanh, name='Generator')
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def discriminator_model_create():
    inputs = Input(shape=(256, 256, 3), name='Input')

    conv2d_1 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', name='2D_Conv1')(inputs)
    leaky_1 = LeakyReLU(alpha=0.1, name='leaky1')(conv2d_1)

    conv2d_2 = Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same', name='2D_Conv2')(leaky_1)
    batch_norm_1 = BatchNormalization(name='Batch_Norm1')(conv2d_2)
    leaky_2 = LeakyReLU(alpha=0.2, name='leaky2')(batch_norm_1)

    conv2d_3 = Conv2D(256, kernel_size=(4, 4), strides=(2, 2), padding='same', name='2D_Conv3')(leaky_2)
    batch_norm_2 = BatchNormalization(name='Batch_Norm2')(conv2d_3)
    leaky_3 = LeakyReLU(alpha=0.2, name='leaky3')(batch_norm_2)

    conv2d_4 = Conv2D(512, kernel_size=(4, 4), strides=(1, 1), padding='same', name='2D_Conv4')(leaky_3)
    batch_norm_3 = BatchNormalization(name='Batch_Norm3')(conv2d_4)
    leaky_4 = LeakyReLU(alpha=0.2, name='leaky4')(batch_norm_3)

    conv2d_4 = Conv2D(1, kernel_size=(4, 4), strides=(1, 1), padding='same', name='2D_Conv5')(leaky_4)
    sigmoid = Activation('sigmoid', name='sigmoid')(conv2d_4)

    # model = Model(inputs=[inputs_real, inputs_fake], outputs=sigmoid, name='Discriminator')
    model = Model(inputs=inputs, outputs=sigmoid, name='Discriminator')
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model


discriminator = discriminator_model_create()

generator = generator_model_create()
discriminator.trainable = False


def gan_model_create():
    gan_input = Input(shape=(256, 256, 3), name='input')
    fake_image = generator(gan_input)
    gan_output = discriminator(fake_image)

    model = Model(gan_input, gan_output, name='gan_model')
    model.compile(loss='binary_crossentropy', optimizer=opt)

    return model


gan = gan_model_create()
gan.summary()


# for epoch in range(epochs):
#     for batch in range(steps_per_epoch):
#         gen_input, _ = next(fake_data_gen)
#         fake_x = generator.predict(gen_input)
#
#         real_x, _ = next(real_data_gen)
#
#         x = np.concatenate((real_x, fake_x))
#
#         disc_y = np.zeros(2 * batch_size)
#         disc_y[:batch_size] = 0.9
#
#         d_loss = discriminator.train_on_batch(x, disc_y)
#
#         y_gen = np.ones(batch_size)
#         g_loss = gan.train_on_batch(gen_input, y_gen)
#
#     print(f'Epoch: {epoch} \t Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}')

