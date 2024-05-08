import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2

from keras import regularizers
from keras import activations
from keras import Layer
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D
       

class TemperatureScaledSoftmax(Layer):
    def __init__(self, temperature=10.0, **kwargs):
        super(TemperatureScaledSoftmax, self).__init__(**kwargs)
        self.temperature = temperature  # Adjustable temperature parameter

    def call(self, inputs):
        scaled_logits = inputs / self.temperature
        return tf.nn.softmax(scaled_logits)

    def get_config(self):
        config = super(TemperatureScaledSoftmax, self).get_config()
        config.update({"temperature": self.temperature})
        return config

class CustomModel(tf.keras.Model):
       def __init__(self):
              super(CustomModel, self).__init__()
              self.num_classes = 30
              self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
              
              self.body = [
                     Conv2D(64, 5, 1, padding='same', activation=activations.leaky_relu, name='conv1'),
                     Conv2D(64, 5, 1, padding='same', activation=activations.leaky_relu, name='conv2'),
                     Conv2D(64, 5, 1, padding='same', activation=activations.leaky_relu, name='conv3'),
                     MaxPool2D(3, name="pool1"),
                     Conv2D(128, 3, 1, padding='same', activation=activations.leaky_relu, name='conv4'),
                     Conv2D(128, 3, 1, padding='same', activation=activations.leaky_relu, name='conv5')
              ]
              
              self.head = [
                     Flatten(),
                     Dense(256, activation=tf.nn.leaky_relu, kernel_regularizer=regularizers.L2(l2=0.001)),
                     Dropout(0.3),
                     Dense(128, activation=tf.nn.leaky_relu, kernel_regularizer=regularizers.L2(l2=0.001)),
                     Dense(self.num_classes, activation=None) # 30 classes
              ]
              self.body = tf.keras.Sequential(self.body, name="custom_base")
              self.head = tf.keras.Sequential(self.head, name="custom_head")
              self.temperature_softmax = TemperatureScaledSoftmax(temperature=10)
       def call(self, x):
              """ Passes the image through the network. """

              x = self.body(x)
              x = self.head(x)
              x = self.temperature_softmax(x)
              return x

       @staticmethod
       def loss_fn(labels, predictions):
              """ Loss function for model. """

              loss_obj = tf.keras.losses.SparseCategoricalCrossentropy() # 30 classes

              return loss_obj(labels, predictions)
       
       
       
class VGGModel(tf.keras.Model):
    def __init__(self):
        super(VGGModel, self).__init__()

        self.num_classes = 30
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


        self.vgg16 = [
            # Block 1
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv1"),
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv2"),
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv1"),
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv2"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv1"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv2"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv3"),
            MaxPool2D(2, name="block3_pool"),
            # Block 4
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv3"),
            MaxPool2D(2, name="block4_pool"),
            # Block 5
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv3"),
            MaxPool2D(2, name="block5_pool")
        ]

        for layer in self.vgg16:
            layer.trainable = False


        self.head = [
            Flatten(),
            Dense(128, activation=tf.nn.leaky_relu),
            Dropout(0.1),
            Dense(64, activation=tf.nn.leaky_relu),
            Dense(self.num_classes, activation='softmax') # 30 classes

        ]

        self.vgg16 = tf.keras.Sequential(self.vgg16, name="vgg_base")
        self.head = tf.keras.Sequential(self.head, name="vgg_head")

    def call(self, x):
        """ Passes the image through the network. """

        x = self.vgg16(x)
        x = self.head(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """


        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy() # 30 classes

        return loss_obj(labels, predictions)
    
