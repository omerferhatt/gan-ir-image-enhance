import os
from img_io import Dataset
from gan import GAN, Train
from tensorflow.keras.optimizers import Adam

os.chdir('C:\\Users\\omerf\\PycharmProjects\\gan-ir-image-enhance')

data = Dataset('data_preprocess\\output_dir')
data.take_file_paths()

opt = Adam(learning_rate=0.0002, beta_1=0.5)

gan = GAN(input_shape=(256, 256, 3),
          batch_size=1,
          optimizer=opt,
          loss='binary_crossentropy',
          metrics=['accuracy'])


gan.create_generator()
gan.create_discriminator(is_trainable=False)
gan.create_adversarial()

train = Train(gan, data, epoch=10, batch_size=1)
train.train_on_batch()
