from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

# Temporary bug workaround
# https://github.com/fchollet/keras/issues/3857
# https://github.com/tensorflow/tensorflow/issues/4616
import tensorflow as tf
tf.python.control_flow_ops = tf

# TODO change dimension ordering
# Recent version of keras expects the number of convolutional filters to come *last*, not first.

def vgg16(input_shape, nb_classes):
  model = Sequential()

  # border_mode 'same' ensures that the output will have the same dimensions
  # as the input (except perhaps for the number of channels).
  model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=input_shape))
  model.add(Activation('relu'))
  # Note: Keras does automatic shape inference.
  model.add(Convolution2D(64, 3, 3, border_mode='same'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Convolution2D(128, 3, 3, border_mode='same'))
  model.add(Activation('relu'))
  model.add(Convolution2D(128, 3, 3, border_mode='same'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Convolution2D(256, 3, 3, border_mode='same'))
  model.add(Activation('relu'))
  model.add(Convolution2D(256, 3, 3, border_mode='same'))
  model.add(Activation('relu'))
  model.add(Convolution2D(256, 3, 3, border_mode='same'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Convolution2D(512, 3, 3, border_mode='same'))
  model.add(Activation('relu'))
  model.add(Convolution2D(512, 3, 3, border_mode='same'))
  model.add(Activation('relu'))
  model.add(Convolution2D(512, 3, 3, border_mode='same'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Convolution2D(512, 3, 3, border_mode='same'))
  model.add(Activation('relu'))
  model.add(Convolution2D(512, 3, 3, border_mode='same'))
  model.add(Activation('relu'))
  model.add(Convolution2D(512, 3, 3, border_mode='same'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Flatten())
  model.add(Dense(4096))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(4096))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(nb_classes))
  model.add(Activation('softmax'))

  # lr decay is accomplished using a custom callback
  sgd = SGD(lr=0.003, momentum=0.3, nesterov=True)
  # TODO: Should I be using the log-likelihood loss? (refer to VGG paper)
  model.compile(loss='categorical_crossentropy', optimizer=sgd)
  return model

def simple_cnn(input_shape, nb_classes):
  model = Sequential()

  model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=input_shape))
  model.add(Activation('relu'))
  model.add(Convolution2D(64, 3, 3, border_mode='same'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Flatten())
  model.add(Dropout(0.5))
  model.add(Dense(nb_classes))
  model.add(Activation('softmax'))

  sgd = SGD(lr=0.003, momentum=0.9, nesterov=True)
  model.compile(loss='categorical_crossentropy', optimizer=sgd)
  return model

