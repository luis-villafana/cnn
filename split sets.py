import os
import shutil

dir_unsplit = 'whole images/1/'
#dir_unsplit = 'all data/whole images/1/'
dir_training = '../neural net/training set/1/'
dir_testing = '../neural net/testing set/1/'

#dir_numbers = [int(images_dir) for images_dir in processed_images_dirs]
files_0 = os.listdir(dir_unsplit)
files_1 = [dir_unsplit + file_name for file_name in files_0]
files_training = [dir_training + file_name for file_name in files_0]
files_testing = [dir_testing + file_name for file_name in files_0]

for n, file in enumerate(files_1):
    if n % 20 == 0:
        shutil.copy(file, files_testing[n])
    else:
        shutil.copy(file, files_training[n])