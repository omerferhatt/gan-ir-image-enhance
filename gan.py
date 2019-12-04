import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Add, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Flatten
from tensorflow.keras.models import Model

from img_io import Dataset


class GAN:
    def __init__(self,
                 input_shape: tuple,
                 optimizer: tf.keras.optimizers.Optimizer,
                 loss: str,
                 metrics: list):
        self.input_shape = input_shape
        self.input_width = input_shape[0]
        self.input_height = input_shape[1]
        self.input_channel = input_shape[2]

        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self.generator = None
        self.discriminator = None
        self.adversarial = None

    def create_generator(self):
        inputs = Input(shape=(self.input_width, self.input_height, self.input_channel), name='Input_Generator')
        x = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', name='2D_Conv_1')(inputs)
        x_add1 = Activation('relu', name='Relu_1')(x)

        x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='2D_Conv_2')(x_add1)
        x = BatchNormalization(name='B_Norm_1')(x)
        x = Activation('relu', name='Relu_2')(x)

        x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='2D_Conv_3')(x)
        x = BatchNormalization(name='B_Norm_2')(x)
        x_add2 = Activation('relu', name='Relu_3')(x)

        x = Add(name='Skip_Con')([x_add1, x_add2])

        x = Conv2DTranspose(3, kernel_size=(4, 4), strides=(2, 2), padding='same', name='2D_Conv_Trans')(x)
        outputs = Activation('tanh', name='TanH')(x)

        model = Model(inputs=inputs, outputs=outputs, name='Generator_Model')
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        self.generator = model

    def create_discriminator(self, is_trainable: bool = False):
        inputs = Input(shape=(self.input_width, self.input_height, self.input_channel * 2), name='Input_Discriminator')

        x = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', name='2D_Conv_1')(inputs)
        x = LeakyReLU(alpha=0.1, name='Leaky_Relu_1')(x)

        x = Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same', name='2D_Conv_2')(x)
        x = BatchNormalization(name='B_Norm_1')(x)
        x = LeakyReLU(alpha=0.2, name='Leaky_Relu_2')(x)

        x = Conv2D(256, kernel_size=(4, 4), strides=(2, 2), padding='same', name='2D_Conv_3')(x)
        x = BatchNormalization(name='B_Norm_2')(x)
        x = LeakyReLU(alpha=0.2, name='Leaky_Relu_3')(x)

        x = Conv2D(512, kernel_size=(4, 4), strides=(1, 1), padding='same', name='2D_Conv_4')(x)
        x = BatchNormalization(name='B_Norm_3')(x)
        x = LeakyReLU(alpha=0.2, name='Leaky_Relu_4')(x)

        x = Conv2D(1, kernel_size=(4, 4), strides=(1, 1), padding='same', name='2D_Conv_5')(x)
        x = Flatten()(x)
        x = Dense(1)(x)
        outputs = Activation('sigmoid', name='Sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs, name='Discriminator_Model')
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        self.discriminator = model
        self.discriminator.trainable = is_trainable

    def create_adversarial(self):
        high_input = Input(shape=self.input_shape, name='High_Input')
        low_input = Input(shape=self.input_shape, name='Low_Input')

        fake_output = self.generator(low_input)

        low_high = tf.concat([low_input, high_input], axis=3, name='Low_High_Concat')
        low_fake = tf.concat([low_input, fake_output], axis=3, name='Low_Fake_Concat')
        disc_input = tf.concat([low_high, low_fake], axis=0, name='Low_High_Low_Fake_Concat')

        prob_output = self.discriminator(disc_input)

        model = Model(inputs=[high_input, low_input], outputs=prob_output, name='Adversarial_Model')
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        self.adversarial = model


class Train:
    def __init__(self, model: GAN, dataset: Dataset, epoch: int, batch_size: int):
        self.model = model
        self.adversarial: tf.keras.models.Model = model.adversarial
        self.generator: tf.keras.models.Model = model.generator
        self.discriminator: tf.keras.models.Model = model.discriminator

        self.total_image: int = dataset.total_image

        self.dataset = dataset
        self.dataset.batch_size = batch_size

        self.epoch: int = epoch
        self.batch_size: int = batch_size
        if self.dataset.total_image % self.dataset.batch_size != 0:
            self.steps_per_epoch = int((self.dataset.total_image / self.dataset.batch_size) + 1)
        else:
            self.steps_per_epoch = int(self.dataset.total_image / self.dataset.batch_size)

    def train_on_batch(self):
        for epoch in range(self.epoch):
            for batch in range(self.steps_per_epoch):
                high_input, low_input = self.dataset.load_batch(batch)
                fake_output = self.generator.predict(low_input)

                low_high = np.concatenate((low_input, high_input), axis=3)
                low_fake = np.concatenate((low_input, fake_output), axis=3)
                x = np.concatenate((low_high, low_fake), axis=0)

                y_disc = np.zeros(2 * self.batch_size)
                y_disc[:self.batch_size] = 0.9

                disc_loss = self.discriminator.train_on_batch(x, y_disc)

                y_gen = np.ones(self.batch_size)
                gen_loss = self.adversarial.train_on_batch([high_input, low_input], y_gen)

        # noinspection PyUnboundLocalVariable
        print(f'Epoch: {epoch} \t Discriminator Loss: {disc_loss} \t\t Generator Loss: {gen_loss}')
