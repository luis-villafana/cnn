import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time
import os
import shutil
import time
import random

images_dir1 = 'go 1/U625 - GO/'
images_dir2 = 'go 2/U625 - GO(2)/'
images_dir3 = 'go 3/U625 - GO(3)/'
training_dir = 'training set go/'
testing_dir = 'testing set go/'

images_dirs = [images_dir1, images_dir2, images_dir3]
instance_names = []

for image_dir in images_dirs:
    recording_dirs = os.listdir(image_dir)
    for recording_dir in recording_dirs:
        
        instance_names.append(image_dir + recording_dir + '/')

for n_shuffle in range(0, 10):
    random.shuffle(instance_names)
    
for n_move, instance_name in enumerate(instance_names):
    new_name = instance_name.split('/')
    new_name = new_name[len(new_name) - 2] + '/'
    if n_move % 20 == 0:
        shutil.move(instance_name, testing_dir + 'go ' + new_name)
    else:
        shutil.move(instance_name, training_dir + 'go ' + new_name)
'''
        image_steps = os.listdir(image_dir + recording_dir)
        for step in image_steps:
            images_names.append(image_dir + recording_dir + '/' + step)
            
for n_shuffle in range(0, 10):
    random.shuffle(images_names)
    
for n_move, image_name in enumerate(images_names):
    new_name = image_name.split('/')
    new_name = new_name[len(new_name) - 1]
    if n_move % 20 == 0:
        shutil.move(image_name, testing_dir + 'go ' + new_name)
    else:
        shutil.move(image_name, training_dir + 'go ' + new_name)
'''