'''
This is the neural network training script that I have been using.
'''

from keras.preprocessing import image
import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from pathlib import Path
import random

n_filters = 8
n_nodes = 450
filter_size = 5

n_epochs = 1000000
n_batches = 1

n_strides = 3

n_classes = 2

x_pixels = 10288
y_pixels = 15000

#Should probably name it as n_inputs instead
n_inputs = 3

training_dir = 'training set/'
example_names = os.listdir(training_dir)

'''
This function loads an arbitrary image set
The image files are assumed to be of the form "label_imageidentifier.png"
'''
def load_data_set(image_names):
    x = []
    y = []
    #y = np.zeros((n_batches, 2))
    
    for image_name in image_names:
        this_image = image.img_to_array(image.load_img(training_dir + image_name))
        x.append(this_image)
        image_name_parts = image_name.split('_')
        #y.append(image_name_parts[0])
        y_vector = np.zeros((1, 2))
        y_vector[0, int(image_name_parts[0])] = 1
        y.append(y_vector.flatten())
    
    #Get a numpy array with shape (n_samples, n_y_pixels, n_x_pixels, n_colors)
    x = np.array(x)
    x = x.astype('float32')
    x /= 255
    return x, np.array(y)

# Create a model and add layers
model = Sequential()

model.add(Conv2D(n_filters, filter_size, strides = n_strides, padding='same', input_shape=(y_pixels, x_pixels, n_inputs), activation="relu"))
model.add(Conv2D(n_filters, filter_size, strides = n_strides, activation="relu"))

model.add(Dropout(0.25))

model.add(Conv2D(n_filters, filter_size, strides = n_strides, padding='same', activation="relu"))
model.add(Conv2D(n_filters, filter_size, strides = n_strides,  activation="relu"))

model.add(Dropout(0.25))

model.add(Conv2D(n_filters, filter_size, strides = n_strides, padding='same', activation="relu"))
model.add(Conv2D(n_filters, filter_size, strides = n_strides,  activation="relu"))

model.add(Dropout(0.25))

model.add(Conv2D(n_filters, filter_size, strides = n_strides, padding='same', activation="relu"))
model.add(Conv2D(n_filters, filter_size, strides = n_strides,  activation="relu"))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(n_nodes, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation="softmax"))

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model

# Save the neural network structure
model_structure = model.to_json()
f = Path('model_structure.json')
f.write_text(model_structure)

current_accuracy = 0
while True:
    random.shuffle(example_names)
    for n_name, example_name in enumerate(example_names):
        '''
        Load the training data set
        '''
        x_training, y_training = load_data_set([example_name])
        
        model_history = model.fit(
            x_training,
            y_training,
            batch_size = n_batches,
            epochs = 1,
            shuffle = True
        )
        #if model_history.history['acc'][0] > current_accuracy:
        # Save neural network's trained weights
        model.save_weights('model_weights.h5')
        print('saved higher accuracy model')