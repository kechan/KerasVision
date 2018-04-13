import argparse
import random
import os
from tqdm import tqdm
from load_data import from_splitted_hdf5
from augmentation.CustomImageDataGenerator import * 

#from data_util import *
import h5py

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import Xception, VGG16, VGG19, InceptionV3, ResNet50, MobileNet, DenseNet121 

import numpy as np


BATCH_SIZE = 32

description = """Description:\n
Extract feature for transfer learning from hdf5 dataset
"""

'''
From:
       data_dir > train_set.hdf5
                  dev_set.hdf5
		  test_set.hdf5

to:
       output_dir > feature_train_set.hdf5
       		    feature_dev_set.hdf5
		    feature_test_set.hdf5
'''

parser = argparse.ArgumentParser(description=description)
parser.add_argument('--pretrained_model', default='VGG16', help='Name of pre-trained model')
parser.add_argument('--data_dir', default='original', help="Directory with the dataset")
parser.add_argument('--output_dir', default='original_hdf5', help="Where to write the new data")
parser.add_argument('--data_aug_n', default='0', help="Number of data augmentation round, the dataset will increase this number of times.")
#parser.add_argument('--classes', help="comma delimited string list of classes")
#parser.add_argument('--resize', default='-1', help='resize image to this dimension, a square')

def extract_features(pretrained_model, set_x, set_y, sample_count):
    ''' Go through the X-Y set and return extracted features from pre-trained model for transfer learning. '''
    datagen = ImageDataGenerator(rescale=1./255)
    batch_size = BATCH_SIZE

    height, width, channel = pretrained_model.output_shape[1:]

    features = np.zeros(shape=(sample_count, height, width, channel))
    labels = np.zeros(shape=(sample_count, 1))

    generator = datagen.flow(set_x, set_y, batch_size=BATCH_SIZE)

    i = 0
    for inputs_batch, labels_batch in tqdm(generator):
        
        features_batch = pretrained_model.predict(inputs_batch)
        
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break

    return features, labels

def extract_features_with_data_aug(pretrained_model, data_gen, set_x, set_y, sample_count):
    ''' Go through X-Y set and return data-augmented extracted features from pre-trained model for transfer learning.''' 
    batch_size = BATCH_SIZE

    height, width, channel = pretrained_model.output_shape[1:]

    features = np.zeros(shape=(sample_count, height, width, channel))
    labels = np.zeros(shape=(sample_count, 1))

    generator = data_gen.flow(set_x, set_y, batch_size=BATCH_SIZE)

    i=0
    for inputs_batch, labels_batch in tqdm(generator):

        features_batch = pretrained_model.predict(inputs_batch)
        
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break

    return features, labels


def dump_h5(feature_set_x, feature_set_y, classes, outfile_path=None):

    # figure out if this is train, dev, or test
    prefix = None
    if "train" in outfile_path:
        prefix = "train"
    elif "dev" in outfile_path or "validation" in outfile_path:
        prefix = "dev"
    elif "test" in outfile_path:
        prefix = "test"

        
    # open a hdf5 file and create earrays
    if not os.path.exists(outfile_path):
        hdf5_file = h5py.File(outfile_path, mode='w')

	shape = feature_set_x.shape
        hdf5_file.create_dataset(prefix + "_set_x", feature_set_x.shape, np.float32, maxshape=(None, shape[1], shape[2], shape[3] ))
        hdf5_file[prefix + "_set_x"][...] = feature_set_x

	shape = feature_set_y.shape
        hdf5_file.create_dataset(prefix + "_set_y", feature_set_y.shape, np.uint8, maxshape=(None, shape[1]))
        hdf5_file[prefix + "_set_y"][...] = feature_set_y
    
        hdf5_file.create_dataset("list_classes", classes.shape, 'S10')
        hdf5_file["list_classes"][...] = classes 

    else:
        hdf5_file = h5py.File(outfile_path, mode='r+')         #append data

	set_x = hdf5_file[prefix + "_set_x"]
	orig_len_x = len(set_x)
	set_x.resize(orig_len_x + len(feature_set_x), axis=0)
	set_x[orig_len_x:, :] = feature_set_x

	set_y = hdf5_file[prefix + "_set_y"]
	orig_len_y = len(set_y)
	set_y.resize(orig_len_y + len(feature_set_y), axis=0)
	set_y[orig_len_y:, :] = feature_set_y

    hdf5_file.close()

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    data_dir = args.data_dir
    output_dir = args.output_dir
    if args.data_aug_n is not None:
        data_aug_n = int(args.data_aug_n)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    train_set_x, train_set_y, dev_set_x, dev_set_y, test_set_x, test_set_y, classes = from_splitted_hdf5(data_dir)

    # for quick testing
    '''
    train_set_x = train_set_x[:5, :, :, :]
    train_set_y = train_set_y[:5, :]
    dev_set_x = dev_set_x[:5, :, :, :]
    dev_set_y = dev_set_y[:5, :]
    test_set_x = test_set_x[:5, :, :, :]
    test_set_y = test_set_y[:5, :]
    '''

    set_x = [train_set_x, dev_set_x, test_set_x]
    set_y = [train_set_y, dev_set_y, test_set_y]

    if args.pretrained_model == 'VGG16':
        pretrained_model = VGG16(weights='imagenet', include_top=False, input_shape=train_set_x.shape[1:])
    elif args.pretrained_model == 'VGG19':
        pretrained_model = VGG19(weights='imagenet', include_top=False, input_shape=train_set_x.shape[1:])
    elif args.pretrained_model == 'Xception':
        pretrained_model = Xception(weights='imagenet', include_top=False, input_shape=train_set_x.shape[1:])

    data_gen = CustomImageDataGenerator(rescale=1./255, 
                                        gaussian_blur_range=0.7, 
					height_shift_range=0.2, 
					width_shift_range=0.2, 
					color_shift=True,
					rot90=True, 
					contrast_stretching=True
					)

    ## train set
    hdf5_filename = args.pretrained_model + "_feature_train.hdf5"
    outfile_path = os.path.join(output_dir, hdf5_filename)

    if not os.path.exists(outfile_path):    # if train set exists, don't extract for original images again
        feature_set_x, feature_set_y = extract_features(pretrained_model, train_set_x, train_set_y, len(train_set_x))
        print(outfile_path)
        dump_h5(feature_set_x, feature_set_y, classes, outfile_path)

    # data aug for train set
    for k in range(data_aug_n):
        feature_set_x, feature_set_y = extract_features_with_data_aug(pretrained_model, data_gen, train_set_x, train_set_y, len(train_set_x))
        dump_h5(feature_set_x, feature_set_y, classes, outfile_path)
    
    for i, c in enumerate(['dev', 'test']):

        hdf5_filename =  args.pretrained_model + "_feature_" + c + ".hdf5"
        outfile_path = os.path.join(output_dir, hdf5_filename)

	if not os.path.exists(outfile_path):   # if dev or test set exists, don't do anything, no data aug necessary
            feature_set_x, feature_set_y = extract_features(pretrained_model, set_x[i], set_y[i], len(set_x[i]))
	    print(outfile_path)
            dump_h5(feature_set_x, feature_set_y, classes, outfile_path)

