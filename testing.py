import argparse
import os, shutil, glob
from subprocess import check_call
import sys

from utils import Params

from data.load_data import from_splitted_hdf5
from data.augmentation.CustomImageDataGenerator import * 
from data.data_util import preview_data_aug

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

PYTHON = sys.executable
parser = argparse.ArgumentParser()

def test_custom_image_data_generator(train_set_x, train_set_y, dev_set_x, dev_set_y):

    data_gen = CustomImageDataGenerator(rescale=1./255, 
                                        gaussian_blur_range=0.7, 
					height_shift_range=0.2, 
					color_shift=True,
					width_shift_range=0.2, 
					rot90=True, 
					cut_out=(10, 20),
					contrast_stretching=True
					)

    index = np.random.randint(len(train_set_x))
    original_image = train_set_x[index:index+1]

    preview_num = 10
    i = 0
    for batch, _ in data_gen.flow(original_image, train_set_y[index,:], batch_size=32):
        plt.figure()
	imgplot = plt.imshow(batch[0])

        i += 1
	if i % preview_num == 0:
	    break
  
    plt.show() 

if __name__ == "__main__":
    args = parser.parse_args()

    home_dir = os.getenv("HOME")
    project_dir = os.path.join(home_dir, "Documents", "CoinSee")
    top_data_dir = os.path.join(project_dir, "data")

    data_dir = os.path.join(top_data_dir, "224x224_cropped_hdf5")

    train_set_x, train_set_y, dev_set_x, dev_set_y, _, _, classes = \
               from_splitted_hdf5(os.path.join(top_data_dir, "224x224_cropped_hdf5"))

    test_custom_image_data_generator(train_set_x, train_set_y, dev_set_x, dev_set_y)
