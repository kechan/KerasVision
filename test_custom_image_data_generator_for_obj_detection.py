import argparse
import os, shutil, glob, h5py
from subprocess import check_call
import sys

from utils import Params

from keras.utils import to_categorical
from data.load_data import from_splitted_hdf5
from data.augmentation.CustomImageDataGenerator import * 
from data.data_util import preview_data_aug

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

PYTHON = sys.executable
parser = argparse.ArgumentParser()

def test_custom_image_data_generator(train_set_x, train_set_y, dev_set_x, dev_set_y):

    '''
    data_gen = CustomImageDataGenerator(rescale=1./255, 
                                        gaussian_blur_range=1.0, 
					height_shift_range=0.2, 
					width_shift_range=0.2, 
					shear_range=0.1,
					zoom_range=0.4,
					color_shift=[15, 15, 15],
					rot90=True, 
					cut_out=(20, 7),
					contrast_stretching=True
					)
    '''

    #data_gen = CustomImageDataGenerator(rescale=1./255, height_shift_range=0.4)
    data_gen = CustomImageDataGenerator(rescale=1./255, cut_out=(20, 7))

    index = np.random.randint(len(train_set_x))

    image = train_set_x[index:index+1]

    #image_label = to_categorical(train_set_y[index,:], num_classes=7)
    image_label = train_set_y[index:index+1,:]

    preview_num = 3
    i = 0

    for img, label in data_gen.flow(image, image_label, batch_size=32, save_to_dir=None):

        plt.figure()
	imgplot = plt.imshow(img[0])

	print("label: {}".format(label))

        i += 1
	if i % preview_num == 0:
	    break
  
    plt.show() 

def test_custom_image_data_generator_for_dir(directory_path):

    data_gen = CustomImageDataGenerator(rescale=1./255, 
                                        gaussian_blur_range=1.0, 
					height_shift_range=0.2, 
					width_shift_range=0.2, 
					shear_range=0.1,
					zoom_range=0.4,
					color_shift=[15, 15, 15],
					rot90=True, 
					cut_out=(20, 7),
					contrast_stretching=True
					)

    preview_num = 2
    i = 0
    for batch_imgs, batch_labels in data_gen.flow_from_directory_with_1x1_conv_target(os.path.join(directory_path, 'train'), 
                                                                      target_size = (224, 224), 
								      class_mode = 'categorical', 
								      batch_size = 32):
        #plt.figure()
	#imgplot = plt.imshow(batch_imgs[0])
	#print(batch_imgs.shape)
        print(batch_labels[0])

	i += 1
        if i % preview_num == 0:
	    break
       
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args()

    home_dir = os.getenv("HOME")
    mount_dir = "/Volumes/My Mac Backup"

    project_dir = os.path.join(home_dir, "Documents", "CoinSee")
    #project_dir = os.path.join(mount_dir, "CoinSee")
    
    top_data_dir = os.path.join(project_dir, "data")

    
    '''
    data_dir = os.path.join(top_data_dir, "224x224_cropped_merged_heads_hdf5")

    train_set_x, train_set_y, dev_set_x, dev_set_y, _, _, classes = \
               from_splitted_hdf5(data_dir)

    test_custom_image_data_generator(train_set_x, train_set_y, dev_set_x, dev_set_y)
    '''
    

    '''
    data_dir = os.path.join(top_data_dir, 'cropped_merged_heads_resized_224')

    test_custom_image_data_generator_for_dir(data_dir)
    '''

    #'''
    data_dir = os.path.join(top_data_dir, '224x224_original_and_cropped_merged_heads_bbox')

    train_file = os.path.join(data_dir, "train_224_224.hdf5")
    dataset = h5py.File(train_file, 'r')
    classes = dataset['list_classes'][:]

    # shuffle and select

    p = np.random.permutation(len(dataset['train_set_x']))   #[:320]

    train_set_x = dataset['train_set_x'][:][p]
    train_set_y = dataset['train_set_y'][:][p]
    train_idx_filenames = dataset['train_idx_filenames'][:][p]

    indice_classes = {}
    for i, c in enumerate(classes):
        indice_classes[str(i)] = c

    dev_file = os.path.join(data_dir, "validation_224_224.hdf5")
    dataset = h5py.File(dev_file, 'r')

    dev_set_x = dataset['dev_set_x'][:]   #128
    dev_set_y = dataset['dev_set_y'][:]   #128
    dev_idx_filenames = dataset['dev_idx_filenames'][:]   #128

    test_file = os.path.join(data_dir, "test_224_224.hdf5")
    dataset = h5py.File(test_file, 'r')

    test_set_x = dataset['test_set_x'][:]
    test_set_y = dataset['test_set_y'][:]
    test_idx_filenames = dataset['test_idx_filenames'][:]

    test_custom_image_data_generator(train_set_x, train_set_y, dev_set_x, dev_set_y)
    #'''


