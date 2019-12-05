import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Activation, Add, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Flatten, Concatenate
from tensorflow.keras.models import Model


from img_io import Dataset


# GAN model class, it contains all sub models and adversarial model itself.
# Model networks and inputs can be editable from model schemes
class GAN:
    def __init__(self,
                 input_shape: tuple,
                 loss: str,
                 metrics: list,
                 optimizer: str,
                 learning_rate: float,
                 beta1: float,
                 beta2: float = None):
        # Model input information
        self.input_shape = input_shape
        self.input_width = input_shape[0]
        self.input_height = input_shape[1]
        self.input_channel = input_shape[2]

        # Model compile informations
        self.batch_size = None
        self.optimizer = self.create_optimizer(optimizer, learning_rate, beta1, beta2)
        self.loss = loss
        self.metrics = metrics

        # Sub-Models and Adversarial Model
        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator(is_trainable=False)
        self.adversarial = self.create_adversarial()

        # Private network schemes
        self.__generator_network = None
        self.__discriminator_network = None
        self.__generator_outputs = None
        self.__discriminator_outputs = None

    # noinspection PyProtectedMember
    @staticmethod
    def create_optimizer(opt, lr, b1, b2=None):
        _optimizer = tf.keras.optimizers.get(opt)
        _optimizer._set_hyper('learning_rate', lr)
        _optimizer._set_hyper('beta_1', b1)
        if b2 is not None: _optimizer._set_hyper('beta_2', b2)
        return _optimizer

    # Creates generator sub-model of adversarial network
    def create_generator(self):
        inputs = Input(shape=(self.input_width, self.input_height, self.input_channel),
                       batch_size=self.batch_size,
                       name='generator/input')
        outputs = self._generator_network(inputs)

        model = Model(inputs=inputs, outputs=outputs, name='Generator_Model')
        return model

    # Generator network scheme
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

    # Creates discriminator sub-model of adversarial network
    def create_discriminator(self, is_trainable: bool = False):
        inputs_gen = Input(shape=(self.input_width, self.input_height, self.input_channel),
                           batch_size=self.batch_size,
                           name='discriminator/input_gen')
        inputs_low = Input(shape=(self.input_width, self.input_height, self.input_channel),
                           batch_size=self.batch_size,
                           name='discriminator/input_low')

        inputs = Concatenate(name='discriminator/inputs')([inputs_gen, inputs_low])

        outputs = self._discriminator_network(inputs)

        model = Model(inputs=[inputs_gen, inputs_low], outputs=outputs, name='Discriminator_Model')
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        model.trainable = is_trainable
        return model

    # Discriminator network scheme
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

    # Creates combined network from generative and discriminator sub-models to create adversarial model
    def create_adversarial(self):
        low_input = Input(shape=self.input_shape, batch_size=self.batch_size, name='adversarial/input_low')
        high_input = Input(shape=self.input_shape, batch_size=self.batch_size, name='adversarial/input_high')

        gen_output = self.generator(low_input)
        disc_output = self.discriminator([gen_output, high_input])

        model = Model(inputs=low_input, outputs=disc_output, name='Adversarial_Model')
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        return model


# Train given model with given dataset.
# Epochs and batch sizes can be editable
class Train:
    def __init__(self, model: GAN, dataset: Dataset, epoch: int, batch_size: int):
        # Models itself
        self.model = model
        self.model.batch_size = batch_size
        # Networks inside main GAN model
        self.adversarial: tf.keras.models.Model = model.adversarial
        self.generator: tf.keras.models.Model = model.generator
        self.discriminator: tf.keras.models.Model = model.discriminator

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
        for epoch in range(self.epoch):
            for batch in range(self.steps_per_epoch):
                high_input, low_input = self.dataset.load_batch(batch)
                fake_output = self.generator.predict(low_input)

                y_disc_real = np.ones(self.batch_size)
                y_disc_fake = np.zeros(self.batch_size)

                disc_loss_real = self.discriminator.train_on_batch([high_input, low_input], y_disc_real)
                disc_loss_fake = self.discriminator.train_on_batch([fake_output, low_input], y_disc_fake)
                disc_loss = 0.5 * np.add(disc_loss_real, disc_loss_fake)

                y_gen = np.ones(self.batch_size)
                gen_loss = self.adversarial.train_on_batch([low_input, high_input], y_gen)
                self.train_logger('batch', epoch, batch, disc_loss, gen_loss)

        # noinspection PyUnboundLocalVariable
        self.train_logger('epoch', epoch, batch, disc_loss, gen_loss)

    def train_logger(self, log_select, epoch, batch, disc_loss, gen_loss):
        if log_select == 'batch' and batch % 20 == 0:
            print(f'Step in Epoch: {batch}/{self.steps_per_epoch}'
                  f'Discriminator Loss: {disc_loss[0]:2.3f}'
                  f'Accuracy: %{disc_loss[1] * 100:3.2f}'
                  f'Generator Loss: {gen_loss[0]:3.2f}')
        if log_select == 'epoch':
            print(f'Epoch: {epoch}/{self.epoch}'
                  f'Avg. Discriminator Loss: {disc_loss[0]:2.5f}'
                  f'Avg. Accuracy: %{disc_loss[1] * 100:3.2f} \t'
                  f'Avg. Generator Loss: {gen_loss[0]:2.5f}')

    def save_sample_images(self, epoch):
        # TODO run generator with sample images than save them into disk
        # r, c = 2, 3
        #
        # sampled_labels = np.arange(0, 10).reshape(-1, 1)
        #
        # gen_imgs = self.generator.predict([noise, sampled_labels])
        #
        # # Rescale images 0 - 1
        # gen_imgs = 0.5 * gen_imgs + 0.5
        #
        # fig, axs = plt.subplots(r, c)
        # cnt = 0
        # for i in range(r):
        #     for j in range(c):
        #         axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
        #         axs[i,j].set_title("Digit: %d" % sampled_labels[cnt])
        #         axs[i,j].axis('off')
        #         cnt += 1
        # fig.savefig("images/%d.png" % epoch)
        # plt.close()
        pass
