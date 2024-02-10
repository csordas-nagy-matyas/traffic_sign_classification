#https://www.tensorflow.org/tutorials/images/transfer_learning
#https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-preprocessed?resource=download&select=datasets_preparing.py
#https://towardsdatascience.com/classification-of-traffic-signs-with-lenet-5-cnn-cb861289bd62
#https://paperswithcode.com/methods/category/activation-functions
#https://www.preprints.org/manuscript/202301.0463/v1

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionV3, MobileNetV2, VGG16

from custom_models import LeNet5, LeNet5Custom, LeNet5CustomV1, LeNet5CustomV2, AlexNet
from plotting_tools import show_random_images, show_predictions, show_history, show_images
from data_loader import load_data, load_data_old, get_mapping, create_data_generators, create_datagen_flow
from balanced_data_generator import BalancedDataGenerator
from model_settings import model_build, model_compile_fit
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from mlxtend.plotting import plot_confusion_matrix


class NetworkRunner:

    def __init__(self, backbone, img_width, img_height):
        self.val_split = 0.2
        self.seed = 10
        self.batch_size = 32
        self.backbone = backbone
        self.img_width = img_width
        self.img_height = img_height
        self.train_pickle = "dataset/train.pickle"
        self.val_pickle = "dataset/valid.pickle"
        self.test_pickle = "dataset/test.pickle"
        self.mapping_csv = "dataset/label_names.csv"

    def load_data(self):
        #self.train_imgs, self.val_imgs, self.test_imgs, self.train_labels, self.val_labels, self.test_labels = load_data(self.train_pickle, self.img_width, self.img_height)
        self.train_imgs, self.train_labels = load_data_old(self.train_pickle, self.img_width, self.img_height)
        self.val_imgs, self.val_labels = load_data_old(self.val_pickle, self.img_width, self.img_height)
        self.test_imgs, self.test_labels = load_data_old(self.test_pickle, self.img_width, self.img_height)
        self.mapping = get_mapping(self.mapping_csv)

    def create_generators(self):
        train_datagen, val_datagen, test_datagen = create_data_generators()
        self.generators = create_datagen_flow([[self.train_imgs, self.train_labels],
                                               [self.val_imgs, self.val_labels],
                                               [self.test_imgs, self.test_labels]],
                                              [train_datagen, val_datagen, test_datagen],
                                              self.seed, self.batch_size)

        self.balanced_train_generator = BalancedDataGenerator(self.train_imgs, self.train_labels, train_datagen, batch_size=self.batch_size)

    def setup_basemodel(self):

        if not hasattr(self.backbone, "custom_model"):
            basemodel = self.backbone(input_shape=(self.img_height, self.img_width,np.shape(self.train_imgs)[3]),
                                      include_top = False,
                                      weights = 'imagenet')

            basemodel.trainable = True
            n_trainable_layers = 190
            for layer in basemodel.layers[-n_trainable_layers:]:
                layer.trainable = True
            for layer in basemodel.layers[:-n_trainable_layers]:
                layer.trainable = False

            print(basemodel.summary())
            # call the build function to build model
            self.model = model_build(basemodel)
        else:
            input_layer = Input(shape=(self.img_height, self.img_width, np.shape(self.train_imgs)[3],))
            x = self.backbone(input_layer)
            self.model = Model(inputs=input_layer, outputs=x)
            print(self.model.summary())
    
    def plot_examples(self):
        show_random_images(self.mapping, self.balanced_train_generator)

    def run(self):

        with tf.device('/CPU:0'):
            self.load_data()
            self.create_generators()
            self.setup_basemodel()
            #self.plot_examples()
            

        with tf.device(tf.config.list_logical_devices('GPU')[0]):
            final_model, history = model_compile_fit(self.model, self.generators, self.balanced_train_generator, self.train_imgs, self.train_labels, self.val_imgs, self.val_labels)

        with tf.device('/CPU:0'):
            result = final_model.evaluate(self.test_imgs, self.test_labels)
            
            print(dict(zip(final_model.metrics_names, result)))
            y_pred = to_categorical(np.argmax(final_model.predict(self.test_imgs), axis=1), 43)
            y_true= to_categorical(self.test_labels, num_classes=43)
            conf_matrix = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))  
            fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Greens)
            plt.xlabel('Predictions', fontsize=18)
            plt.ylabel('Actuals', fontsize=18)
            plt.title('Confusion Matrix', fontsize=18)
            plt.show()

        #show predictions
        with tf.device('/CPU:0'):
            show_history(history)
            show_predictions(final_model, self.mapping, self.test_imgs, self.test_labels, self.img_height, self.img_width)


if __name__ == "__main__":

    networkrunner = NetworkRunner(InceptionV3, 130, 130)
    networkrunner.run()
    
   
    