import os
import shutil
import numpy as np

#images_dir1 = 'go 1/U625 - GO/'
images_dir2 = 'go 2/U625 - GO(2)/'
images_dir3 = 'go 3/U625 - GO(3)/'
all_data_dir = 'all data/1/'
training_dir = 'training set go/'
testing_dir = 'testing set go/'

#images_dirs = [images_dir2, images_dir3]
images_dirs = ['no go/U625 - NO GO/']
n_instance = 0

old_instance_names = []
new_instance_names = []

for image_dir in images_dirs:
    recording_dirs = os.listdir(image_dir)

    for n_recording, recording_dir in enumerate(recording_dirs):
        
        old_instance_names.append(image_dir + recording_dir + '/')
        new_instance_names.append(all_data_dir + '0_' + str(n_instance) + '/')
        #shutil.move(old_instance_names[n_recording], new_instance_names[n_recording])
        #new_instance_name
        n_instance += 1

for n in range(0, len(new_instance_names)):
    shutil.move(old_instance_names[n], new_instance_names[n])