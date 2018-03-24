import argparse
import os

from PIL import Image
from tqdm import tqdm
from shutil import copyfile

from data_util import *


description = """Description:\n
Resize/split dataset and output a single hdf5\
"""

'''
From: 
	data_dir > Label1 
		 > Label2 
		 > Label3 
		 etc.

to:

	output_dir > dataset.hdf5
'''

parser = argparse.ArgumentParser(description=description)
parser.add_argument('--data_dir', default='original', help="Directory with the dataset")
parser.add_argument('--output_dir', default='original_hdf5', help="Where to write the new data")
parser.add_argument('--classes', help="comma delimited string list of classes")
parser.add_argument('--resize', default='-1', help='resize image to this dimension, a square')
parser.add_argument('--split_ratio', default='0.6,0.2,0.2', help='split ratio for train, dev, and test set')

def process_classes(comma_sep_class_string):
    return comma_sep_class_string.split(",")

def process_split_ratio(comma_sep_n):
    return [float(s) for s in comma_sep_n.split(",")]

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    data_dir = args.data_dir
    output_dir = args.output_dir
    classes = process_classes(args.classes)
    size = int(args.resize)
    split_ratio = process_split_ratio(args.split_ratio)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    indice_classes = {}
    for i, c in enumerate(classes):
        indice_classes[str(i)] = c

    outfile_path = os.path.join(output_dir, str(size) + "_" + str(size) + ".hdf5")

    generate_h5(data_dir, labels_to_classes_dictionary=indice_classes, 
        outfile_path=outfile_path, resize_height=size, resize_width=size,
	train_dev_test_ratio=split_ratio)
