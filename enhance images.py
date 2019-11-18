import os
import shutil
import imageio
import numpy as np
from PIL import Image, ImageEnhance

#D:\neural net\training set\example\enhanced

brightness_factor = 3

#D:\neural net\original set\training set

whole_images_dir = 'original set/training set/'

#D:\neural net\enhanced set\training set

downsampled_images_dir = 'enhanced set/training set/'

whole_image_names = os.listdir(whole_images_dir)
for image_name in whole_image_names:
    raw_image = Image.open(whole_images_dir + image_name)
    enhancer = ImageEnhance.Brightness(raw_image)
    
    enhanced_image = enhancer.enhance(brightness_factor)
    
    enhanced_image = enhanced_image.resize((1286, 1875))
    
    enhanced_image.save(downsampled_images_dir + image_name)
    #downsampled_image = whole_image.resize((2 * 1286, 2 * 1875))
    #downsampled_image.save(downsampled_images_dir + image_name)
    print(image_name)
    #whole_image = imageio.volread(whole_images_dir + image_name)
    #downsampled_image = 

'''
for n_bundle, bundle in enumerate(split_images_bundles):
    image_names = os.listdir(split_images_dir + bundle)
    #bundles_list.append([len(image_names), bundle, image_names])
    split_images = []
    for image_name in image_names:
        split_images.append(imageio.volread(split_images_dir + bundle + '/' + image_name))
    whole_images = split_images.pop(0)
    
    for image in split_images:
        whole_images = np.concatenate((whole_images, image), axis = 0)
        imageio.imsave(combined_images_dir + bundle + '.jpg', whole_images)
'''

'''
name1s = []
name2s = []
for folder in split_images_bundles:
    name1 = split_images_dir + folder + '/'
    name2 = split_images_dir + '1_' + folder.split('_')[1] + '/'
    
    os.rename(name1, name2)
'''