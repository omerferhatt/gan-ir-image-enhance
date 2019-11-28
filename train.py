from tensorflow.keras.optimizers import Adam
from gan_model import GAN

opt = Adam(learning_rate=0.0002, beta_1=0.5)

gan = GAN(input_shape=(256, 256, 3),
          batch_size=15,
          epoch=10,
          optimizer=opt,
          loss='binary_crossentropy',
          metrics=['accuracy'])

gan.load_data('data_preprocess/output_dir')

gan.create_generator()
gan.create_discriminator(is_trainable=False)
gan.create_adversarial()
