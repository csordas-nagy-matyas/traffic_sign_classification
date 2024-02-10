import numpy as np

from sklearn.utils import compute_class_weight
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input


def model_build(basemodel):

    # flatten the output of the base model
    x = Flatten()(basemodel.output)
    # add a fully connected layer 
    x = Dense(128, activation='relu')(x)
    # add dropout layer for regularization
    #x = Dropout(0.2)(x)
    #x = Dense(256, activation='relu')(x)

    # add final layer for classification
    x = Dense(43, activation='softmax')(x)

    model = Model(basemodel.input, x)
    
    return model


def model_compile_fit(model, generators, balanced_generator, train_imgs, train_labels, val_imgs, val_labels):
    model.compile(optimizer = optimizers.RMSprop(learning_rate=0.001),
                                              loss='sparse_categorical_crossentropy',
                                              metrics=['accuracy'])
    callbacks = [EarlyStopping(monitor = 'loss', 
                    patience = 5, 
                    mode = 'min', 
                    min_delta=0.01)]
    
    #class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_labels), y=train_labels)
    #class_weights = dict(enumerate(class_weights))

    history = model.fit(balanced_generator, #balanced_generator, #generators["train_generator"]
                        validation_data = (val_imgs, val_labels),#generators["val_generator"],
                        steps_per_epoch=300, # num of batches in one epoch
                        epochs=60,
                        callbacks=callbacks)
    
    return model, history