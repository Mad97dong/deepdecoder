import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Conv2DTranspose, Reshape, LeakyReLU, Flatten, Dense, UpSampling2D
from tensorflow.keras import Model, Sequential
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from tqdm.keras import TqdmCallback

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('Tensorflow version', tf.__version__)
from skimage.io import imread, imsave
from skimage.color import rgb2gray, rgba2rgb, gray2rgb
# from skimage.color import rgb2gray, rgba2rgb


def build_decoder():
    decoder = tf.keras.models.Sequential()
    for _ in range(5):
        decoder.add(
            Conv2D(filters=latent_dim, kernel_size=1, strides=1, padding='valid', use_bias=False,
                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.1),
                   input_shape=(16, 16, 128)))
        decoder.add(UpSampling2D(interpolation='bilinear'))
        decoder.add(Activation('relu'))
        decoder.add(BatchNormalization())

    # 2nd last layer
    decoder.add(
        Conv2D(filters=latent_dim, kernel_size=1, strides=1, padding='valid', use_bias=False,
               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.1)))
    decoder.add(Activation('relu'))
    decoder.add(BatchNormalization())

    # last layer
    decoder.add(
        Conv2D(filters=3, kernel_size=1, padding='valid', activation='sigmoid', use_bias=False,
               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.1)))

    return decoder


# optimizer = Adam(0.0002, 0.5)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=200,
    decay_rate=0.65)
optimizer = Adam(learning_rate=lr_schedule)

def train_and_save(model, path, **kwargs):
    if path.exists():
        model.load_weights(str(path))
        # if continue training
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=optimizer)
        _ = model.fit(**kwargs)
        model.save_weights(str(path))
    else:
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=optimizer)
        _ = model.fit(**kwargs)
        model.save_weights(str(path))

