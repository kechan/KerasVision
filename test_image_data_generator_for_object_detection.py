import argparse
import os, shutil, glob, h5py
from subprocess import check_call
import sys

from utils import Params

from keras import backend as K

from keras.utils import to_categorical
from data.load_data import from_splitted_hdf5
from data.augmentation.CustomImageDataGenerator import * 
from data.augmentation.ImageDataGeneratorObjectDetection import *
from data.data_util import preview_data_aug

from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator
from keras.preprocessing.image import transform_matrix_offset_center, apply_transform, array_to_img

import matplotlib.pyplot as plt

PYTHON = sys.executable
parser = argparse.ArgumentParser()

def get_data(data_dir):
    
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

    return train_set_x, train_set_y, dev_set_x, dev_set_y, test_set_x, test_set_y

 
def test_data_gen_for_obj_detection(train_set_x, train_set_y, dev_set_x, dev_set_y, save_to_dir=None):

    data_gen = ImageDataGeneratorObjectDetection(rescale=1./255, height_shift_range=0.3, width_shift_range=0.3)

    index = np.random.randint(len(train_set_x))

    image = train_set_x[index:index+3, :]
    image_label = train_set_y[index:index+3,:]

    preview_num = 3
    i = 0

    for img, label in data_gen.flow(image, image_label, batch_size=32, save_to_dir=save_to_dir):

	print("label: {}".format(label))

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
    data_dir = os.path.join(top_data_dir, '224x224_original_and_cropped_merged_heads_bbox')
    tmp_dir = os.path.join(top_data_dir, 'tmp')

    train_set_x, train_set_y, dev_set_x, dev_set_y, test_set_x, test_set_y = get_data(data_dir)

    test_data_gen_for_obj_detection(train_set_x, train_set_y, dev_set_x, dev_set_y, save_to_dir=tmp_dir) 


