'''
This is how I have been testing the 2D CNNs
'''
import time
import keras
import os
from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np

testing_set_dir = 'enhanced set/testing set/'

# Load the json file that contains the model's structure
f = Path('model_structure.json')
model_structure = f.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights('at 95 acc/model_weights.h5')

'''
q = model.get_weights()
model.set_weights(q[0 : 1])
'''

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
        this_image = image.img_to_array(image.load_img(files_dir + image_name, grayscale = True))
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
Load the testing set
'''
inputs, expected_outputs = load_data_set(testing_set_dir)
n_classes = expected_outputs.shape[1]
#n_classes = 8

# Make a prediction using the model
time_before = time.time()

results = model.predict(inputs)
time_after = time.time()
print(str(time_after - time_before))

true_count = np.zeros([n_classes, 1])
#positive rates refers to true and false positives
positive_rates = np.zeros([n_classes, n_classes])

'''
for n0 in range(0, len(inputs)):
    true_index = int(np.argmax(expected_outputs[n0]))
    combined_true_index = n_classes if true_index in [2] else n_classes + 1
    true_count[true_index] += 1
    true_count[combined_true_index] += 1
'''

#Summary of the results provided by the neural net

for n0 in range(0, len(results)):
    
    true_index = int(np.argmax(expected_outputs[n0]))
    predicted_index = int(np.argmax(results[n0]))
    
    true_count[true_index] += 1
    
    positive_rates[true_index, predicted_index] += 1

true_count = np.repeat(true_count, n_classes, axis = 1)

positive_rates = positive_rates / true_count * 100