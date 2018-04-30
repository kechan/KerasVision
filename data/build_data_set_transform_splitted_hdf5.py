import argparse
import random
import os
from tqdm import tqdm
from load_data import from_splitted_hdf5

import h5py
import numpy as np
import cv2


BATCH_SIZE = 32

description = """Description:\n
Perform image transform for a hdf5 dataset
"""

'''
From:
       data_dir > train_set.hdf5
                  dev_set.hdf5
		  test_set.hdf5

to:
       output_dir > train_set.hdf5
       		    dev_set.hdf5
		    test_set.hdf5
'''

parser = argparse.ArgumentParser(description=description)
parser.add_argument('--data_dir', default='original', help="Directory with the dataset")
parser.add_argument('--output_dir', default='original_hdf5', help="Where to write the new data")


def dump_h5(set_x, set_y, classes, outfile_path=None):

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

	shape = set_x.shape
        hdf5_file.create_dataset(prefix + "_set_x", set_x.shape, np.uint8, maxshape=(None, shape[1], shape[2], shape[3] ))
        hdf5_file[prefix + "_set_x"][...] = set_x

	shape = set_y.shape
        hdf5_file.create_dataset(prefix + "_set_y", set_y.shape, np.uint8, maxshape=(None, shape[1]))
        hdf5_file[prefix + "_set_y"][...] = set_y
    
        hdf5_file.create_dataset("list_classes", classes.shape, 'S10')
        hdf5_file["list_classes"][...] = classes 

    else:
        hdf5_file = h5py.File(outfile_path, mode='r+')         #append data

	set_x = hdf5_file[prefix + "_set_x"]
	orig_len_x = len(set_x)
	set_x.resize(orig_len_x + len(set_x), axis=0)
	set_x[orig_len_x:, :] = set_x

	set_y = hdf5_file[prefix + "_set_y"]
	orig_len_y = len(set_y)
	set_y.resize(orig_len_y + len(set_y), axis=0)
	set_y[orig_len_y:, :] = set_y

    hdf5_file.close()

def apply_transform(set_x):

    shape = set_x.shape

    new_set_x = np.zeros((len(set_x), shape[1], shape[2], shape[3])).astype('uint8')
    
    for i in range(len(set_x)):
        new_set_x[i] = np.expand_dims(cv2.cvtColor(set_x[i], cv2.COLOR_RGB2GRAY), axis=-1).astype('uint8')

    return new_set_x

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    data_dir = args.data_dir
    output_dir = args.output_dir

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    train_set_x, train_set_y, dev_set_x, dev_set_y, test_set_x, test_set_y, classes = from_splitted_hdf5(data_dir)

    # for quick testing
    
    #train_set_x = train_set_x[:5, :, :, :]
    #train_set_y = train_set_y[:5, :]
    #dev_set_x = dev_set_x[:5, :, :, :]
    #dev_set_y = dev_set_y[:5, :]
    #test_set_x = test_set_x[:5, :, :, :]
    #test_set_y = test_set_y[:5, :]

    ## train set
    hdf5_filename = "train.hdf5"
    outfile_path = os.path.join(output_dir, hdf5_filename)

    set_x = [train_set_x, dev_set_x, test_set_x]
    set_y = [train_set_y, dev_set_y, test_set_y]

    for i, c in enumerate(['train', 'dev', 'test']):

        hdf5_filename =  c + ".hdf5"
        outfile_path = os.path.join(output_dir, hdf5_filename)

	if not os.path.exists(outfile_path):   # if set exists, don't do anything, no data aug necessary
            new_set_x = apply_transform(set_x[i])
	    print(outfile_path)
            dump_h5(new_set_x, set_y[i], classes, outfile_path)

