#https://www.tensorflow.org/tutorials/images/transfer_learning
#https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-preprocessed?resource=download&select=datasets_preparing.py
#https://towardsdatascience.com/classification-of-traffic-signs-with-lenet-5-cnn-cb861289bd62
#https://paperswithcode.com/methods/category/activation-functions
#https://www.preprints.org/manuscript/202301.0463/v1

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.utils import compute_class_weight
from tensorflow.keras.applications import InceptionV3, MobileNetV2, VGG16
from custom_models import LeNet5, LeNet5Custom, LeNet5CustomV1, LeNet5CustomV2, AlexNet
from models.configurations.model_params import MODEL_PARAMS

from plotting_tools import show_random_images, show_predictions, show_history, show_images
from data_loader import load_data_old, get_mapping, create_data_generators, create_datagen_flow
from balanced_data_generator import BalancedDataGenerator
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from mlxtend.plotting import plot_confusion_matrix


class NetworkRunner:
    def __init__(self, backbone, train):
        self.seed = 10
        self.backbone = globals()[backbone]
        self.train = train
        self.val_split = MODEL_PARAMS[self.backbone.__name__]["validation_split"]
        self.batch_size = MODEL_PARAMS[self.backbone.__name__]["batch_size"]
        self.img_width = MODEL_PARAMS[self.backbone.__name__]["input_shape"][0]
        self.img_height = MODEL_PARAMS[self.backbone.__name__]["input_shape"][1]
        self.n_channels = MODEL_PARAMS[self.backbone.__name__]["input_shape"][2]
        self.train_pickle = "dataset/train.pickle"
        self.val_pickle = "dataset/valid.pickle"
        self.test_pickle = "dataset/test.pickle"
        self.mapping_csv = "dataset/label_names.csv"

    def load_data(self):
        if self.train:
            self.train_imgs, self.train_labels = load_data_old(self.train_pickle, self.img_width, self.img_height, self.n_channels)
            self.val_imgs, self.val_labels = load_data_old(self.val_pickle, self.img_width, self.img_height, self.n_channels)
        self.test_imgs, self.test_labels = load_data_old(self.test_pickle, self.img_width, self.img_height, self.n_channels)
        self.mapping = get_mapping(self.mapping_csv)

    def create_generators(self):
        train_datagen, val_datagen, test_datagen = create_data_generators(is_train=self.train)
        if self.train:
            self.generators = create_datagen_flow([[self.test_imgs, self.test_labels],
                                                [self.train_imgs, self.train_labels],
                                                [self.val_imgs, self.val_labels]],
                                                [test_datagen, train_datagen, val_datagen],
                                                self.seed, self.batch_size)

            self.balanced_train_generator = BalancedDataGenerator(self.train_imgs, self.train_labels, train_datagen, batch_size=self.batch_size)
        else:
            self.generators = create_datagen_flow([[self.test_imgs, self.test_labels]],
                                                   [test_datagen],
                                                   self.seed, self.batch_size)

    def setup_basemodel(self):

        if hasattr(self.backbone(), "custom_model"):
            input_layer = Input(shape=(self.img_height, self.img_width, np.shape(self.train_imgs)[3],))
            x = self.backbone()(input_layer)
            self.final_model = Model(inputs=input_layer, outputs=x)
            print(self.model.summary())
        else:
            self.basemodel = self.backbone(input_shape=(self.img_height, self.img_width,np.shape(self.train_imgs)[3]),
                                      include_top = False,
                                      weights = 'imagenet')

            self.basemodel.trainable = True
            n_trainable_layers = MODEL_PARAMS[self.backbone.__name__]["n_trainable_layers"]
            for layer in self.basemodel.layers[-n_trainable_layers:]:
                layer.trainable = True
            for layer in self.basemodel.layers[:-n_trainable_layers]:
                layer.trainable = False

            print(self.basemodel.summary())
            # call the build function to build model
            self.model_build()

    def model_build(self):
        # flatten the output of the base model
        x = Flatten()(self.basemodel.output)
        for layer in MODEL_PARAMS[self.backbone.__name__]["fcn_layers"]: 
            x = globals()[layer](**MODEL_PARAMS[self.backbone.__name__]["fcn_layers"][layer])(x)

        # add final layer for classification
        x = Dense(43, activation='softmax')(x)

        self.final_model = Model(self.basemodel.input, x)
    
    def model_compile_fit(self):
        self.final_model.compile(optimizer = optimizers.RMSprop(learning_rate=MODEL_PARAMS[self.backbone.__name__]["learning_rate"]),
                                                loss='sparse_categorical_crossentropy',
                                                metrics=['accuracy'])
        callbacks = [EarlyStopping(monitor = 'loss', 
                        patience = 5, 
                        mode = 'min', 
                        min_delta=0.01)]
        
        #class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_labels), y=train_labels)
        #class_weights = dict(enumerate(class_weights))

        self.history = self.final_model.fit(self.balanced_train_generator, #self.balanced_train_generator, #self.generators["train_generator"]
                            validation_data = (self.val_imgs, self.val_labels),#(self.val_imgs, self.val_labels), self.generators["val_generator"],
                            steps_per_epoch=MODEL_PARAMS[self.backbone.__name__]["steps_per_epoch"], # num of batches in one epoch
                            epochs=MODEL_PARAMS[self.backbone.__name__]["epochs"],
                            callbacks=callbacks)

    def plot_examples(self):
        show_random_images(self.mapping, self.balanced_train_generator)

    def run(self):

        with tf.device('/CPU:0'):
            self.load_data()
            self.create_generators()
            #self.plot_examples()
            
        if self.train:
            self.setup_basemodel()
            with tf.device(tf.config.list_logical_devices('GPU')[0]):
                self.model_compile_fit()
                self.final_model.save(f"models/{self.backbone.__name__}_model.h5")
                show_history(self.history)
        
        with tf.device('/CPU:0'):
            trained_model = load_model(f"models/{self.backbone.__name__}_model.h5")
            result = trained_model.evaluate(self.test_imgs, self.test_labels)
            
            print(dict(zip(trained_model.metrics_names, result)))
            y_pred = to_categorical(np.argmax(trained_model.predict(self.test_imgs), axis=1), 43)
            y_true= to_categorical(self.test_labels, num_classes=43)
            conf_matrix = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))  
            fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Greens)
            plt.xlabel('Predictions', fontsize=18)
            plt.ylabel('Actuals', fontsize=18)
            plt.title('Confusion Matrix', fontsize=18)
            plt.show()

        #show predictions
        with tf.device('/CPU:0'):
            show_predictions(trained_model, self.mapping, self.test_imgs, self.test_labels, self.img_height, self.img_width, self.n_channels)
    