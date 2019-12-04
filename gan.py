import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Add, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Flatten, Concatenate
from tensorflow.keras.models import Model

from img_io import Dataset


class GAN:
    def __init__(self,
                 input_shape: tuple,
                 batch_size: int,
                 optimizer: tf.keras.optimizers.Optimizer,
                 loss: str,
                 metrics: list):
        self.input_shape = input_shape
        self.input_width = input_shape[0]
        self.input_height = input_shape[1]
        self.input_channel = input_shape[2]

        self.batch_size = batch_size

        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self.generator = None
        self.discriminator = None
        self.adversarial = None

        self.__generator_network = None
        self.__discriminator_network = None
        self.__generator_outputs = None
        self.__discriminator_outputs = None

    def create_generator(self):
        inputs = Input(shape=(self.input_width, self.input_height, self.input_channel),
                       batch_size=self.batch_size,
                       name='generator/input')
        outputs = self._generator_network(inputs)

        model = Model(inputs=inputs, outputs=outputs, name='Generator_Model')
        # model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        self.generator = model

    @staticmethod
    def _generator_network(inputs):
        x = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', name='generator/b1/conv2d')(inputs)
        x_add1 = Activation('relu', name='generator/b1/relu')(x)

        x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='generator/b2/conv2d')(x_add1)
        x = BatchNormalization(name='generator/b2/batch_norm')(x)
        x = Activation('relu', name='generator/b2/relu')(x)

        x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='generator/b3/conv2d')(x)
        x = BatchNormalization(name='generator/b3/batch_norm')(x)
        x_add2 = Activation('relu', name='generator/b3/relu')(x)

        x = Add(name='generator/b4/skip_con')([x_add1, x_add2])
        x = Conv2DTranspose(3, kernel_size=(4, 4), strides=(2, 2), padding='same', name='generator/b4/deconv2d')(x)
        outputs = Activation('tanh', name='generator/b4/tanh')(x)
        return outputs

    def create_discriminator(self, is_trainable: bool = False):
        inputs = Input(shape=(self.input_width, self.input_height, self.input_channel),
                       batch_size=self.batch_size,
                       name='discriminator/input')

        outputs = self._discriminator_network(inputs)

        model = Model(inputs=inputs, outputs=outputs, name='Discriminator_Model')
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        self.discriminator = model
        self.discriminator.trainable = is_trainable

    @staticmethod
    def _discriminator_network(inputs):
        x = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', name='discriminator/b1/conv2d')(inputs)
        x = LeakyReLU(alpha=0.1, name='discriminator/b1/leaky')(x)

        x = Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same', name='discriminator/b2/conv2d')(x)
        x = BatchNormalization(name='discriminator/b2/batch_norm')(x)
        x = LeakyReLU(alpha=0.2, name='discriminator/b2/leaky')(x)

        x = Conv2D(256, kernel_size=(4, 4), strides=(2, 2), padding='same', name='discriminator/b3/conv2d')(x)
        x = BatchNormalization(name='discriminator/b3/batch_norm')(x)
        x = LeakyReLU(alpha=0.2, name='discriminator/b3/leaky')(x)

        x = Conv2D(512, kernel_size=(4, 4), strides=(1, 1), padding='same', name='discriminator/b4/conv2d')(x)
        x = BatchNormalization(name='discriminator/b4/batch_norm')(x)
        x = LeakyReLU(alpha=0.2, name='discriminator/b4/leaky')(x)

        x = Conv2D(1, kernel_size=(4, 4), strides=(1, 1), padding='same', name='discriminator/b5/conv2d')(x)
        x = Flatten(name='discriminator/b5/flatten')(x)
        x = Dense(1, name='discriminator/b5/dense')(x)
        outputs = Activation('sigmoid', name='discriminator/b5/sigmoid')(x)

        return outputs

    def create_adversarial(self):
        low_input = Input(shape=self.input_shape, batch_size=self.batch_size, name='adversarial/input_low')

        gen_output = self.generator(low_input)

        # low_high = Concatenate(axis=3, name='adversarial/concat_low_high')([low_input, high_input])
        # low_gen = Concatenate(axis=3, name='adversarial/concat_low_gen')([low_input, gen_output])
        # disc_input = Concatenate(axis=0, name='adversarial/concat_low_high_low_gen')([low_high, low_gen])

        disc_output = self.discriminator(gen_output)

        model = Model(inputs=low_input, outputs=disc_output, name='Adversarial_Model')
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

                y_disc_real = np.ones(self.batch_size)
                y_disc_real[:] = 0.9
                y_disc_fake = np.zeros(self.batch_size)

                disc_loss_real = self.discriminator.train_on_batch(high_input, y_disc_real)
                disc_loss_fake = self.discriminator.train_on_batch(fake_output, y_disc_fake)
                disc_loss = 0.5 * np.add(disc_loss_real, disc_loss_fake)

                y_gen = np.ones(self.batch_size)
                gen_loss = self.adversarial.train_on_batch(low_input, y_gen)
                if batch % 100 == 0:
                    print(f'Batch: {batch} \t Discriminator Loss: {disc_loss[0]:2.3f} Accuracy: {disc_loss[1]:3.2f} \t\t Generator Loss: {gen_loss[0]:3.2f}')
        # noinspection PyUnboundLocalVariable
        print(f'Epoch: {epoch} \t Discriminator Loss: {disc_loss[0]:2.3f} Accuracy: {disc_loss[1]:3.2f} \t\t Generator Loss: {gen_loss[0]:3.2f}')
