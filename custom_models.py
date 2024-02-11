import tensorflow as tf
from keras import layers, activations, backend
from keras.layers import (
    Conv2D,
    MaxPool2D,
    Flatten,
    Dense,
    Dropout,
    AveragePooling2D,
    BatchNormalization,
    Activation,
)


class LeNet5(tf.keras.Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.custom_model = "LeNet5"
        # creating layers in initializer
        self.conv1 = Conv2D(
            filters=6, kernel_size=(5, 5), padding="same", activation="relu"
        )
        self.max_pool2x2 = MaxPool2D(pool_size=(2, 2))
        self.conv2 = Conv2D(
            filters=16, kernel_size=(5, 5), padding="same", activation="relu"
        )
        self.conv3 = Conv2D(
            filters=120, kernel_size=(5, 5), padding="same", activation="relu"
        )
        self.flatten = Flatten()
        self.fc2 = Dense(units=120, activation="relu")
        self.fc3 = Dense(units=84, activation="relu")
        self.fc4 = Dense(units=43, activation="softmax")

    def __call__(self, input_tensor):
        conv1 = self.conv1(input_tensor)
        maxpool1 = self.max_pool2x2(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.max_pool2x2(conv2)
        conv3 = self.conv3(maxpool2)
        flatten = self.flatten(conv3)
        fc2 = self.fc2(flatten)
        fc3 = self.fc3(fc2)
        fc4 = self.fc4(fc3)

        return fc4


class LeNet5Custom(tf.keras.Model):
    def __init__(self):
        super(LeNet5Custom, self).__init__()
        self.custom_model = "LeNet5"
        # creating layers in initializer
        self.conv1 = Conv2D(
            filters=6, kernel_size=(5, 5), padding="same", activation="relu"
        )
        self.max_pool2x2 = MaxPool2D(pool_size=(2, 2))
        self.conv2 = Conv2D(
            filters=16, kernel_size=(5, 5), padding="same", activation="relu"
        )
        self.conv3 = Conv2D(
            filters=120, kernel_size=(5, 5), padding="same", activation="relu"
        )
        self.conv4 = Conv2D(
            filters=200, kernel_size=(5, 5), padding="same", activation="relu"
        )
        self.flatten = Flatten()
        self.fc2 = Dense(units=120, activation="relu")
        self.fc3 = Dense(units=84, activation="relu")
        self.fc4 = Dense(units=43, activation="softmax")

    def __call__(self, input_tensor):
        conv1 = self.conv1(input_tensor)
        maxpool1 = self.max_pool2x2(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.max_pool2x2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.max_pool2x2(conv3)
        conv4 = self.conv4(maxpool3)
        flatten = self.flatten(conv4)
        fc2 = self.fc2(flatten)
        fc3 = self.fc3(fc2)
        fc4 = self.fc4(fc3)

        return fc4


# val_acc: 85% (64, 64)
class LeNet5CustomV1(tf.keras.Model):
    def __init__(self):
        super(LeNet5CustomV1, self).__init__()
        self.custom_model = "LeNet5"
        # creating layers in initializer
        self.conv1 = Conv2D(
            filters=6, kernel_size=(5, 5), padding="same", activation="relu"
        )
        self.max_pool2x2 = MaxPool2D(pool_size=(2, 2))
        self.conv2 = Conv2D(
            filters=16, kernel_size=(5, 5), padding="same", activation="relu"
        )
        self.conv3 = Conv2D(
            filters=120, kernel_size=(5, 5), padding="same", activation="relu"
        )
        self.conv4 = Conv2D(
            filters=200, kernel_size=(5, 5), padding="same", activation="relu"
        )
        self.flatten = Flatten()
        # self.fc2 = Dense(units=120, activation="relu")
        self.fc3 = Dense(units=84, activation="relu")
        self.dropout = Dropout(0.2)
        self.fc4 = Dense(units=43, activation="softmax")

    def __call__(self, input_tensor):
        conv1 = self.conv1(input_tensor)
        maxpool1 = self.max_pool2x2(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.max_pool2x2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.max_pool2x2(conv3)
        conv4 = self.conv4(maxpool3)
        flatten = self.flatten(conv4)
        # fc2 = self.fc2(flatten)
        fc3 = self.fc3(flatten)
        dropout = self.dropout(fc3)
        fc4 = self.fc4(dropout)

        return fc4


# val_acc: 84.76% train: 91%
class LeNet5CustomV2(tf.keras.Model):

    def __init__(self):
        super(LeNet5CustomV2, self).__init__()
        self.custom_model = "LeNet5"
        # creating layers in initializer
        self.conv1 = Conv2D(
            filters=6, kernel_size=(5, 5), padding="same", activation="relu"
        )
        self.max_pool2x2 = MaxPool2D(pool_size=(2, 2))
        self.conv2 = Conv2D(
            filters=16, kernel_size=(5, 5), padding="same", activation="relu"
        )
        self.conv3 = Conv2D(
            filters=120, kernel_size=(5, 5), padding="same", activation="relu"
        )
        self.conv4 = Conv2D(
            filters=200, kernel_size=(5, 5), padding="same", activation="relu"
        )
        self.flatten = Flatten()
        self.fc2 = Dense(units=120, activation="relu")
        self.fc3 = Dense(units=84, activation="relu")
        self.dropout = Dropout(0.1)
        self.fc4 = Dense(units=43, activation="softmax")

    def __call__(self, input_tensor):
        conv1 = self.conv1(input_tensor)
        maxpool1 = self.max_pool2x2(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.max_pool2x2(conv2)
        conv3 = self.conv3(maxpool2)
        flatten = self.flatten(conv3)
        fc3 = self.fc3(flatten)
        dropout2 = self.dropout(fc3)
        fc4 = self.fc4(dropout2)

        return fc4

    def coslu(self, x):
        alpha = 3
        beta = 1
        return (x + (alpha * backend.cos(beta * x))) * backend.sigmoid(x)

    def relun(self, x):
        n = 1.8
        return backend.minimum(backend.maximum(0.0, x), n)

    def shilu(self, x):
        alpha = 1.5
        beta = 0.0
        return alpha * backend.relu(x) + beta


class AlexNet(tf.keras.Model):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.custom_model = "AlexNet"
        # creating layers in initializer
        self.conv1 = Conv2D(
            filters=24, kernel_size=(5, 5), strides=(2, 2), activation="relu"
        )
        self.max_pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2))
        self.batchnorm1 = BatchNormalization()
        self.conv2 = Conv2D(
            filters=48, kernel_size=(5, 5), padding="same", activation="relu"
        )
        self.max_pool2 = MaxPool2D(pool_size=(3, 3), strides=(1, 1))
        self.batchnorm2 = BatchNormalization()
        self.conv3 = Conv2D(
            filters=96, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.batchnorm3 = BatchNormalization()
        self.conv4 = Conv2D(
            filters=96, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.batchnorm4 = BatchNormalization()
        self.conv5 = Conv2D(
            filters=48, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.batchnorm5 = BatchNormalization()
        self.max_pool5 = MaxPool2D(pool_size=(3, 3), strides=(1, 1))

        self.flatten = Flatten()
        self.fc1 = Dense(units=1024, activation="relu")
        self.fc2 = Dense(units=512, activation="relu")
        self.dropout = Dropout(0.1)
        self.fc3 = Dense(units=43, activation="softmax")

    def __call__(self, input_tensor):
        # 1st Convolutional Block
        conv1 = self.conv1(input_tensor)
        maxpool1 = self.max_pool1(conv1)
        batchnorm1 = self.batchnorm1(maxpool1)

        # 2nd Convolutional Block
        conv2 = self.conv2(batchnorm1)
        maxpool2 = self.max_pool2(conv2)
        batchnorm2 = self.batchnorm2(maxpool2)

        # 3rd Convolutional Block
        conv3 = self.conv3(batchnorm2)
        batchnorm3 = self.batchnorm3(conv3)

        # 4th Convolutional Block
        conv4 = self.conv4(batchnorm3)
        batchnorm4 = self.batchnorm4(conv4)

        # 5th Convolutional Block
        conv5 = self.conv5(batchnorm4)
        batchnorm5 = self.batchnorm5(conv5)
        maxpool5 = self.max_pool5(batchnorm5)

        # Fully Connected Layer
        flatten = self.flatten(maxpool5)
        fc1 = self.fc1(flatten)
        # dropout1 = self.dropout(fc1)
        fc2 = self.fc2(fc1)
        # dropout2 = self.dropout(fc2)
        fc3 = self.fc3(fc2)

        return fc3
