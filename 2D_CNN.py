from tensorflow.keras.preprocessing import image
import os
import numpy as np
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from pathlib import Path
#import cupy as cp

n_filters = 16
n_nodes = 450
filter_size = 3


#n_epochs =  27
#n_epochs =  25
n_epochs = 1000000
#n_batches = 10
n_batches = 20

n_strides = 1

#empirically determined
#filter_size = 140 according to math
#try a filter of size 5

'''
This function loads an arbitrary image set
The image files are assumed to be of the form "label_imageidentifier.png"
'''
def load_data_set(files_dir):
    image_names = os.listdir(files_dir)
    x = []
    y = []
    #y = np.zeros((n_batches, 2))
    
    for image_name in image_names:
        this_image = image.img_to_array(image.load_img(files_dir + image_name))
        x.append(this_image)
        image_name_parts = image_name.split('_')
        #y.append(image_name_parts[0])
        y_vector = np.zeros((1, 2))
        y_vector[0, int(image_name_parts[0])] = 1
        y.append(y_vector.flatten())
    
    #Get a numpy array with shape (n_samples, n_y_pixels, n_x_pixels, n_colors)
    x = np.array(x)
    
    # Normalize data set to 0-to-1 range
    #This would not be necessary if we didn't save the time series as
    #an image file and prescaled it before saving
    x = x.astype('float32')
    x /= 255
    
    # Convert class vectors to binary class matrices
    #n_classes = len(np.unique(y))
    #y = keras.utils.to_categorical(y, n_classes)
    return x, np.array(y)

'''
Load the training data set
'''
#training_dataset = '../../../data/images/ann6 images/training set/'
training_dataset = 'training set/example/'
x_training, y_training = load_data_set(training_dataset)

n_classes = y_training.shape[1]

x_pixels = x_training.shape[2]
y_pixels = x_training.shape[1]

#Should probably name it as n_inputs instead
n_inputs = x_training.shape[3]

print('done loading')

# Create a model and add layers
model = Sequential()

model.add(Conv2D(n_filters, filter_size, strides = n_strides, padding='same', input_shape=(y_pixels, x_pixels, n_inputs), activation="relu"))
model.add(Conv2D(n_filters, filter_size, strides = n_strides, activation="relu"))

model.add(Dropout(0.25))

model.add(Conv2D(n_filters, filter_size, strides = n_strides, padding='same', activation="relu"))
model.add(Conv2D(n_filters, filter_size, strides = n_strides,  activation="relu"))

model.add(Dropout(0.25))
'''
model.add(Conv2D(n_filters, filter_size, strides = n_strides, padding='same', activation="relu"))
model.add(Conv2D(n_filters, filter_size, strides = n_strides,  activation="relu"))

model.add(Dropout(0.25))

model.add(Conv2D(n_filters, filter_size, strides = n_strides, padding='same', activation="relu"))
model.add(Conv2D(n_filters, filter_size, strides = n_strides,  activation="relu"))

model.add(Dropout(0.25))
'''
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
print('done compiling model')

# Train the model

mc = ModelCheckpoint('model_weights.h5', monitor = 'acc', mode = 'max', verbose = 1, save_best_only = True)

# Save the neural network structure
model_structure = model.to_json()
f = Path('model_structure.json')
f.write_text(model_structure)
'''
model_history = model.fit(
    x_training,
    y_training,
    batch_size = n_batches,
    epochs = 10,
    shuffle = True
)
'''

model.fit(
    x_training,
    y_training,
    batch_size = n_batches,
    epochs = n_epochs,
    shuffle = True,
    callbacks = [mc]
)


# Save neural network's trained weights
#model.save_weights("ann" + n_experiment + "/model_weights.h5")
