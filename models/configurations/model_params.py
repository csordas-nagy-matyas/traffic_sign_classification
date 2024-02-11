MODEL_PARAMS = {
    "InceptionV3": {
        "input_shape": (130, 130, 3),
        "n_trainable_layers" : 190,
        "learning_rate" : 0.0001,
        "batch_size": 32,
        "steps_per_epoch": 300,
        "epochs": 60,
        "validation_split": 0.2,
        "fcn_layers": {"Dense": {"units": 128, "activation": "relu"}}
    },
    "VGG16": {
        "input_shape": (120, 120, 3),
        "n_trainable_layers" : 190,
        "learning_rate" : 0.0001,
        "batch_size": 32,
        "steps_per_epoch": 500,
        "epochs": 60,
        "validation_split": 0.2,
        "fcn_layers": {"Dense": {"units": 128, "activation": "relu"}}
    },
    "LeNet5Custom": {
        "input_shape": (60, 60, 1),
        "learning_rate" : 0.0001,
        "batch_size": 32,
        "steps_per_epoch": 500,
        "epochs": 60,
        "validation_split": 0.2,
    },
    "AlexNet": {
        "input_shape": (130, 130, 1),
        "learning_rate" : 0.0001,
        "batch_size": 32,
        "steps_per_epoch": 300,
        "epochs": 60,
        "validation_split": 0.2,
    }
}