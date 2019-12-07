import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.optimizers import Adam

from img_io import Dataset
from gan import GAN, Train

# For Windows based OS, otherwise comment below
# os.chdir(r'C:\Users\omerf\PycharmProjects\gan-ir-image-enhance')

# For Unix/Linux based OS, otherwise comment below
os.chdir('/home/ferhat/PycharmProjects/gan-ir-image-enhance')

data = Dataset('data_preprocess/output_dir', shuffle=2)

opt = Adam(lr=0.0002, beta_1=0.5)

model = GAN(input_shape=(256, 256, 3),
            loss='binary_crossentropy',
            metrics=['accuracy'],
            optimizer=opt)

model.create_discriminator(is_trainable=False)
model.create_generator()
model.create_adversarial()

train = Train(model, data, epoch=10, batch_size=1)
train.train_on_batch()
