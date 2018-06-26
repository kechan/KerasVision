import argparse
import random
import os

from PIL import Image
from tqdm import tqdm
from shutil import copyfile

from data_util import *


description = """Description:\n
Split and resize dataset to output hdf5\
"""

'''
From: 
	data_dir > train > Label1 Folder
		         > Label2 Folder
		         > Label3 Folder
			 etc.

                 > validation > Label1 Folder
		  	      > Label2 Folder 
			      > Label3 Folder
			 etc.
	         
                 > test > Label1 Folder
		        > Label2 Folder 
		        > Label3 Folder
			etc.

to:

	output_dir > train_set.hdf5
	             dev_set.hdf5
		     test_set.hdf5
'''

parser = argparse.ArgumentParser(description=description)
parser.add_argument('--data_dir', default='original', help="Directory with the dataset")
parser.add_argument('--output_dir', default='original_hdf5', help="Where to write the new data")
parser.add_argument('--classes', help="comma delimited string list of classes")
parser.add_argument('--resize', default='-1', help='resize image to this dimension, a square')

def process_classes(comma_sep_class_string):
    return comma_sep_class_string.split(",")

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    data_dir = args.data_dir
    output_dir = args.output_dir
    classes = process_classes(args.classes)
    size = int(args.resize)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    indice_classes = {}
    for i, c in enumerate(classes):
        indice_classes[str(i)] = c 

    for i, split in enumerate(['train', 'validation', 'test']):
    #for i, split in enumerate(['validation', 'test']):
        data_path = os.path.join(data_dir, split)
	if os.path.exists(data_path):
	    outfile_path = os.path.join(output_dir, split + "_" + str(size) + "_" + str(size) + ".hdf5")

	    train_dev_test_ratio = [0.0, 0.0, 0.0]
            train_dev_test_ratio[i] = 1.0
            #train_dev_test_ratio[i+1] = 1.0

            generate_h5(data_path, labels_to_classes_dictionary=indice_classes, 
                outfile_path=outfile_path, resize_height=size, resize_width=size, 
                train_dev_test_ratio=train_dev_test_ratio)
        
