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


def load_data_old(file, img_height, img_width):
    # Opening 'pickle' file and getting images
    with open(file, 'rb') as f:
        d = pickle.load(f, encoding='latin1')  # dictionary type, we use 'latin1' for python3
        # At the same time method 'astype()' used for converting ndarray from int to float
        # It is needed to divide float by float when applying Normalization
        train_imgs = d['features'].astype(np.float32)   # 4D numpy.ndarray type, for train = (34799, 32, 32, 3)
        train_labels = d['labels']                        # 1D numpy.ndarray type, for train = (34799,)
        #plt.hist(train_labels, alpha=0.5)
        #plt.show()
        train_imgs, train_labels = shuffle(train_imgs, train_labels)
        train_imgs_resized = tf.image.resize(train_imgs, [img_height, img_width]).numpy()
        #train_imgs_gray = tf.image.rgb_to_grayscale(train_imgs_resized).numpy()
        #plt.imshow(train_imgs_resized[10]/255.0)
        #plt.show()
        train_imgs_equalized = exposure.equalize_hist(train_imgs_resized)
        #plt.imshow(train_imgs_equalized[i])
        #plt.show()
        #train_imgs_resized = preprocess_input(train_imgs_resized).numpy()
        #train_imgs_resized = tf.image.per_image_standardization(train_imgs_resized)
        #train_imgs_gray = tf.image.rgb_to_grayscale(train_imgs_resized).numpy()
        
        #show_images(train_imgs_resized)
        """
        Data is a dictionary with four keys:
            'features' - is a 4D array with raw pixel data of the traffic sign images,
                         (number of examples, width, height, channels).
            'labels'   - is a 1D array containing the label id of the traffic sign image,
                         file label_names.csv contains id -> name mappings.
        """

    # Returning ready data
    return train_imgs_equalized, train_labels


def load_data(file, img_height, img_width):
    # Opening 'pickle' file and getting images
    with open(file, 'rb') as f:
        d = pickle.load(f, encoding='latin1')  # dictionary type, we use 'latin1' for python3
        
        # At the same time method 'astype()' used for converting ndarray from int to float
        # It is needed to divide float by float when applying Normalization
        train_imgs = d['x_train'].transpose(0, 2, 3, 1)   # 4D numpy.ndarray type, for train = (34799, 32, 32, 3)
        val_imgs = d['x_validation'].transpose(0, 2, 3, 1) 
        test_imgs = d['x_test'].transpose(0, 2, 3, 1) 
        train_labels = d['y_train']     
        val_labels = d['y_validation'] 
        test_labels = d['y_test']                    # 1D numpy.ndarray type, for train = (34799,)
        plt.hist(train_labels, alpha=0.5)
        plt.show()
        
        train_imgs_resized = tf.image.resize(train_imgs, [img_height, img_width])#.numpy()
        val_imgs_resized = tf.image.resize(val_imgs, [img_height, img_width])
        test_imgs_resized = tf.image.resize(test_imgs, [img_height, img_width])
        train_imgs_gray = tf.image.rgb_to_grayscale(train_imgs_resized).numpy()
        val_imgs_gray = tf.image.rgb_to_grayscale(val_imgs_resized).numpy()
        test_imgs_gray = tf.image.rgb_to_grayscale(test_imgs_resized).numpy()
        #show_images(train_imgs_gray)
        """
        Data is a dictionary with four keys:
            'features' - is a 4D array with raw pixel data of the traffic sign images,
                         (number of examples, width, height, channels).
            'labels'   - is a 1D array containing the label id of the traffic sign image,
                         file label_names.csv contains id -> name mappings.
        """

    # Returning ready data
    return train_imgs_gray, val_imgs_gray, test_imgs_gray, train_labels, val_labels, test_labels


def get_mapping(mapping_csv):
    mappings = {}
    with open(mapping_csv, 'r') as data:
        for line in csv.reader(data):
            mappings[line[0]] = line[1]

    return mappings


def create_data_generators():
    # Create ImageDataGenerators for training and validation and testing

    train_datagen = ImageDataGenerator(
        #rescale=1.0/255.0,
        width_shift_range=0.1, 
        height_shift_range=0.1, 
        rotation_range=0,
        zoom_range=0,
        horizontal_flip=False,
        vertical_flip=False,
        #brightness_range=[0,0],
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
    )

    test_datagen = ImageDataGenerator(
        rescale=1.0/255.0
    )

    return train_datagen, val_datagen, test_datagen


def create_datagen_flow(train_val_test_data, datagens, seed, batch_size):

    generators = {"train_generator": '', "val_generator": '', "test_generator": ''}
    for i, datagen in enumerate(datagens):
        key = list(generators)[i]
        datagen.fit(train_val_test_data[i][0])
        generators[key] = datagen.flow(train_val_test_data[i][0],
                                       train_val_test_data[i][1],
                                       seed = seed,
                                       batch_size = batch_size, 
                                       shuffle = True,
                                      )
    return generators