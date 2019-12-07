import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Activation, Add, BatchNormalization, LeakyReLU
from keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Flatten, Concatenate
from keras.models import Model
from keras.initializers import RandomNormal
from img_io import Dataset


# GAN model class, it contains all sub models and adversarial model itself.
# Model networks and inputs can be editable from model schemes
class GAN:
    def __init__(self,
                 input_shape: tuple,
                 loss: str,
                 metrics: list,
                 optimizer: keras.optimizers.Optimizer):
        # Model input information
        self.input_shape = input_shape
        self.input_width = input_shape[0]
        self.input_height = input_shape[1]
        self.input_channel = input_shape[2]

        # Model compile informations
        self.batch_size = None
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        # Sub-Models and Adversarial Model
        self.generator = None
        self.discriminator = None
        self.adversarial = None

        # Private network schemes
        self.__generator_network = None
        self.__discriminator_network = None
        self.__generator_outputs = None
        self.__discriminator_outputs = None

    # Creates generator sub-model of adversarial network
    def create_generator(self):
        inputs = Input(shape=(self.input_width, self.input_height, self.input_channel),
                       name='generator/input')
        
        outputs = self._generator_network(inputs)

        model = Model(inputs=inputs, outputs=outputs, name='Generator_Model')
        self.generator = model

    # Generator network scheme
    @staticmethod
    def _generator_network(inputs):
        x = Conv2D(64, kernel_size=(4, 4), strides=(1, 1), padding='same', name='generator/b1/conv2d')(inputs)
        x_add1 = Activation('relu', name='generator/b1/relu')(x)

        x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='generator/b2/conv2d')(x_add1)
        x = BatchNormalization(name='generator/b2/batch_norm')(x)
        x = Activation('relu', name='generator/b2/relu')(x)

        x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='generator/b3/conv2d')(x)
        x = BatchNormalization(name='generator/b3/batch_norm')(x)
        x_add2 = Activation('relu', name='generator/b3/relu')(x)

        x = Concatenate(name='generator/b4/skip_con1')([x_add1, x_add2])

        x = Conv2DTranspose(3, kernel_size=(4, 4), strides=(1, 1), padding='same', name='generator/b4/deconv2d')(x)
        outputs = Activation('tanh', name='generator/b4/tanh')(x)
        return outputs

    # Creates discriminator sub-model of adversarial network
    def create_discriminator(self, is_trainable: bool = False):
        inputs_gen = Input(shape=(self.input_width, self.input_height, self.input_channel),
                           name='discriminator/input_gen')
        inputs_low = Input(batch_shape=(self.batch_size, self.input_width, self.input_height, self.input_channel),
                           name='discriminator/input_low')

        inputs = Concatenate(name='discriminator/inputs')([inputs_gen, inputs_low])

        outputs = self._discriminator_network(inputs)

        model = Model(inputs=[inputs_gen, inputs_low], outputs=outputs, name='Discriminator_Model')
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        self.discriminator = model
        self.discriminator.trainable = is_trainable

    # Discriminator network scheme
    @staticmethod
    def _discriminator_network(inputs):
        init = RandomNormal(stddev=0.02)
        x = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init, name='discriminator/b1/conv2d')(inputs)
        x = LeakyReLU(alpha=0.2, name='discriminator/b1/leaky')(x)

        x = Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init, name='discriminator/b2/conv2d')(x)
        x = BatchNormalization(name='discriminator/b2/batch_norm')(x)
        x = LeakyReLU(alpha=0.2, name='discriminator/b2/leaky')(x)

        x = Conv2D(256, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init, name='discriminator/b3/conv2d')(x)
        x = BatchNormalization(name='discriminator/b3/batch_norm')(x)
        x = LeakyReLU(alpha=0.2, name='discriminator/b3/leaky')(x)

        x = Conv2D(512, kernel_size=(4, 4), strides=(1, 1), padding='same', kernel_initializer=init, name='discriminator/b4/conv2d')(x)
        x = BatchNormalization(name='discriminator/b4/batch_norm')(x)
        x = LeakyReLU(alpha=0.2, name='discriminator/b4/leaky')(x)

        x = Conv2D(1, kernel_size=(4, 4), strides=(1, 1), padding='same', kernel_initializer=init, name='discriminator/b5/conv2d')(x)
        x = Flatten(name='discriminator/b5/flatten')(x)
        x = Dense(1, kernel_initializer=init, name='discriminator/b5/dense')(x)
        outputs = Activation('sigmoid', name='discriminator/b5/sigmoid')(x)

        return outputs

    # Creates combined network from generative and discriminator sub-models to create adversarial model
    def create_adversarial(self):
        low_input = Input(batch_shape=(self.batch_size, self.input_width, self.input_height, self.input_channel),
                          name='adversarial/input_low')
        # high_input = Input(batch_shape=(self.batch_size, self.input_width, self.input_height, self.input_channel),
        #                    name='adversarial/input_high')

        gen_output = self.generator(low_input)
        disc_output = self.discriminator([gen_output, low_input])

        model = Model(inputs=low_input, outputs=disc_output, name='Adversarial_Model')
        model.compile(optimizer=self.optimizer, loss=self.loss, )

        self.adversarial = model


# Train given model with given dataset.
# Epochs and batch sizes can be editable
class Train:
    def __init__(self, model: GAN, dataset: Dataset, epoch: int, batch_size: int):
        # Models itself
        self.model = model
        self.model.batch_size = batch_size

        # Networks inside main GAN model
        self.adversarial: keras.models.Model = model.adversarial
        self.generator: keras.models.Model = model.generator
        self.discriminator: keras.models.Model = model.discriminator

        # Total image number in dataset.
        # Not sum of high and low
        self.total_image: int = dataset.total_image

        # Dataset object and selected load batch size
        # Batch size need to be same with models batch size
        self.dataset = dataset
        self.dataset.batch_size = batch_size

        # Training parameters
        self.epoch: int = epoch
        self.batch_size: int = batch_size
        self.steps_per_epoch: int = int(self.dataset.total_image / self.dataset.batch_size)

    # Trains model with batches
    def train_on_batch(self):
        y_disc_real = np.ones((self.batch_size, 1))
        y_disc_real[:, 0] = 0.9
        y_disc_fake = np.zeros((self.batch_size, 1))
        y_gen = np.ones(self.batch_size)

        for epoch in range(self.epoch):
            for batch in range(self.steps_per_epoch):
                high_input, low_input = self.dataset.load_batch(batch)
                fake_output = self.generator.predict(low_input)

                disc_loss_real = self.discriminator.train_on_batch([high_input, low_input], y_disc_real)
                disc_loss_fake = self.discriminator.train_on_batch([fake_output, low_input], y_disc_fake)
                disc_loss = 0.5 * np.add(disc_loss_real, disc_loss_fake)

                gen_loss = self.adversarial.train_on_batch(low_input, y_gen)

                self.train_logger('batch', epoch, batch, disc_loss, gen_loss)
        # noinspection PyUnboundLocalVariable
        self.train_logger('epoch', epoch, batch, disc_loss, gen_loss)

    def train_logger(self, log_select, epoch, batch, disc_loss, gen_loss):
        if log_select == 'batch' and batch % 300 == 0 and batch != 0:
            print(f'Step: {batch}/{self.steps_per_epoch}\n'
                  f'Discriminator Loss: {disc_loss[0]:2.5f}\n'
                  f'Accuracy: %{disc_loss[1] * 100:3.2f}\n'
                  f'Generator Loss: {float(gen_loss):2.5f}\n'
                  f'Sample images saved in: generated/IE-CGAN_epoch{epoch}_batch{batch}.png\n')
            self.save_sample_images(epoch, batch)
        elif log_select == 'batch' and batch % 100 == 0 and batch != 0:
            print(f'Step: {batch}/{self.steps_per_epoch}\n'
                  f'Discriminator Loss: {disc_loss[0]:2.5f}\n'
                  f'Accuracy: %{disc_loss[1] * 100:3.2f}\n'
                  f'Generator Loss: {float(gen_loss):2.5f}\n')

        if log_select == 'epoch':
            print(f'\tEpoch: {epoch}/{self.epoch}'
                  f'\tAvg. Discriminator Loss: {disc_loss[0]:2.5f}\n'
                  f'\tAvg. Accuracy: %{disc_loss[1] * 100:3.2f}\n'
                  f'\tAvg. Generator Loss: {float(gen_loss):2.5f}\n')

    def save_sample_images(self, epoch, step):
        gen_imgs = self.generator.predict_on_batch(self.dataset.sample_images)

        gen_imgs = 0.5 * gen_imgs + 0.5
        gen_imgs = np.array(gen_imgs * 255).astype(np.uint8)

        r, c = 2, 3
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(f"generated/IE-CGAN_epoch{epoch}_batch{step}.png", dpi=300)
        plt.close()
