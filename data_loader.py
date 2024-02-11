import pickle
import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from plotting_tools import show_images
from PIL import Image
from sklearn.utils import shuffle
from tensorflow.keras.applications.inception_v3 import preprocess_input
from skimage import exposure


def load_data_old(file, img_height, img_width, n_channels):
    with open(file, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
        train_imgs = d['features'].astype(np.float32)   # 4D numpy.ndarray type, for train = (34799, 32, 32, 3)
        train_labels = d['labels']                      # 1D numpy.ndarray type, for train = (34799,)

        train_imgs, train_labels = shuffle(train_imgs, train_labels)
        train_imgs_resized = tf.image.resize(train_imgs, [img_height, img_width]).numpy()

        if n_channels == 1:
            train_imgs_resized = tf.image.rgb_to_grayscale(train_imgs_resized).numpy()

        train_imgs_equalized = exposure.equalize_hist(train_imgs_resized)
        #show_images(train_imgs_resized)
    return train_imgs_equalized, train_labels


def get_mapping(mapping_csv):
    mappings = {}
    with open(mapping_csv, 'r') as data:
        for line in csv.reader(data):
            mappings[line[0]] = line[1]

    return mappings


def create_data_generators(is_train: bool):
    # Create ImageDataGenerators for training and validation and testing
    test_datagen = ImageDataGenerator(
        rescale=1.0/255.0
    )

    if is_train:
        train_datagen = ImageDataGenerator(
            #rescale=1.0/255.0,
            width_shift_range=0.1, 
            height_shift_range=0.1, 
            rotation_range=0,
            zoom_range=0,
            horizontal_flip=False,
            vertical_flip=False,
        )

        val_datagen = ImageDataGenerator(
            rescale=1.0/255.0,
        )

        return train_datagen, val_datagen, test_datagen
    else:
        return None, None, test_datagen


def create_datagen_flow(test_train_val_data, datagens, seed, batch_size):

    generators = {"test_generator": '', "train_generator": '', "val_generator": ''}
    for i, datagen in enumerate(datagens):
        key = list(generators)[i]
        datagen.fit(test_train_val_data[i][0])
        generators[key] = datagen.flow(test_train_val_data[i][0],
                                       test_train_val_data[i][1],
                                       seed = seed,
                                       batch_size = batch_size, 
                                       shuffle = True,
                                      )
    return generators