#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import tensorflow.keras as keras

import os
import numpy as np
import argparse

from tensorflow.python.client import device_lib
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

bucket = 'surface-detection-dataset'

n_filters = 4
n_nodes = 512
filter_size = 3
n_epochs = 100
n_strides = 1
n_classes = 1

HEIGHT = 15000
WIDTH = 10288
DEPTH = 1
    
INPUT_TENSOR_NAME = "inputs_input"

BATCH_SIZE = 4

def keras_model_fn(hyperparameters):
    n_filters_ = n_filters
    
    model = Sequential()

    model.add(Conv2D(n_filters_, (5, 5), strides=(2, 2), padding='same', input_shape=(HEIGHT, WIDTH, DEPTH), activation="relu", name="inputs"))
    model.add(Conv2D(n_filters_, (5, 5), strides=(1, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(n_filters_, (5, 5), strides=(2, 2), padding='same', activation="relu"))
    model.add(Conv2D(n_filters_, (5, 5), strides=(1, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(n_filters_, (5, 5), strides=(2, 2), padding='same', activation="relu"))
    model.add(Conv2D(n_filters_, (5, 5), strides=(1, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(n_filters_, (5, 5), strides=(2, 2), padding='same', activation="relu"))
    model.add(Conv2D(n_filters_, (5, 5), strides=(1, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    n_filters_ *= 2
    
    model.add(Conv2D(n_filters_, (3, 3), strides=(1, 1), padding='same', activation="relu"))
    model.add(Conv2D(n_filters_, (3, 3), strides=(1, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(n_filters_, (3, 3), strides=(1, 1), padding='same', activation="relu"))
    model.add(Conv2D(n_filters_, (3, 3), strides=(1, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(n_filters_, (3, 3), strides=(1, 1), padding='same', activation="relu"))
    model.add(Conv2D(n_filters_, (3, 3), strides=(1, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(n_nodes, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation="softmax"))
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model


def serving_input_fn(hyperparameters):   
    tensor = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, DEPTH])
    inputs = {INPUT_TENSOR_NAME: tensor}
    
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def train_input_fn(training_dir, hyperparameters):
    return  _input(tf.estimator.ModeKeys.TRAIN, batch_size=hyperparameters['batch_size'], data_dir=training_dir)


def eval_input_fn(testing_dir, hyperparameters):
    return _input(tf.estimator.ModeKeys.EVAL, batch_size=hyperparameters['batch_size'], data_dir=testing_dir)


def _input(mode, batch_size, data_dir):    
    # assert os.path.exists(data_dir), ("Unable to find images resources for input, are you sure you downloaded them ?")
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True
        )
    else:
        datagen = ImageDataGenerator(rescale=1./255)
        
    generator = datagen.flow_from_directory(data_dir, target_size=(HEIGHT, WIDTH), batch_size=batch_size, class_mode='binary')
    # images, labels = generator.next()
    # return {INPUT_TENSOR_NAME: images}, labels
    return generator

def parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=n_epochs, help='Number of epoch to train on dataset')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for each iteration')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate of the optimizer')
    parser.add_argument('--decay', type=float, default=0.0001, help='Decay of the learning rate for the optimizer')

    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'), help='Directory for the training dataset')
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'), help='Directory for the test dataset')

    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'), help='Directory of the model output')

    return parser.parse_known_args()


if __name__ =='__main__':
    args, _ = parse_args()
    
    hyperparameters = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'decay': args.decay
    }
    
    print('Training Directory: ' + args.train)
    print('Test Directory: ' + args.test)
    
    train_gen = train_input_fn(args.train, hyperparameters)
    test_gen = eval_input_fn(args.test, hyperparameters)
    
    model = keras_model_fn(hyperparameters)
    print(model.summary())
    
    checkpointer = ModelCheckpoint(filepath='Baseline_Model-{epoch:03d}-{loss:.3f}.hdf5',
                                   verbose=1,
                                   save_best_only=True)

    tb = TensorBoard(log_dir=args.model_dir)

    callbacks = [tb, checkpointer]
    
    print(device_lib.list_local_devices())
    
    history = model.fit_generator(generator=train_gen,
                                  epochs=args.epochs,
                                  verbose=1,
                                  callbacks=callbacks,
                                  validation_data=test_gen,
                                  use_multiprocessing=True,
                                  workers=4)
    
    tf.contrib.saved_model.save_keras_model(model, args.model_dir)
    