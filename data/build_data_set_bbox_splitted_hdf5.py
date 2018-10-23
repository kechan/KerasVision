import argparse
import random
import os

from PIL import Image
from tqdm import tqdm
from shutil import copyfile

from data_util import *
import pandas as pd

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


X = {an image}
Y = (isObject, c_x, c_y, r, is_5c, is_10c, is_25c, is_5_10_25c_heads, is_one, is_two) 
eg.
    (1, 0.5, 0.5, 0.333, 0, 0, 1, 0, 0, 0) == "a 25c centered at (0.5, 0.5) with apparent bounding radius 0.333"
'''

parser = argparse.ArgumentParser(description=description)
parser.add_argument('--data_dir', default='original', help="Directory with the dataset")
parser.add_argument('--output_dir', default='original_hdf5', help="Where to write the new data")
parser.add_argument('--classes', help="comma delimited string list of classes")
parser.add_argument('--resize', default='-1', help='resize image to this dimension, a square')
parser.add_argument('--bbox_csv_path', default='', help='csv file containing bbox info')

def process_classes(comma_sep_class_string):
    return comma_sep_class_string.split(",")

def get_center_and_bounding_radius(image_row):

    if len(image_row) == 0:
        return 0., 0., 0.
    
    c_x = image_row.center_x.values[0].astype('int')
    c_y = image_row.center_y.values[0].astype('int')

    x_1, y_1 = image_row.x_1.values[0].astype('int'), image_row.y_1.values[0].astype('int')
    x_2, y_2 = image_row.x_2.values[0].astype('int'), image_row.y_2.values[0].astype('int')
    x_3, y_3 = image_row.x_3.values[0].astype('int'), image_row.y_3.values[0].astype('int')
    x_4, y_4 = image_row.x_4.values[0].astype('int'), image_row.y_4.values[0].astype('int')
            
    # compute rectangular bbox from raw coords

    d_1 = (c_x - x_1)**2 + (c_y - y_1)**2
    d_2 = (c_x - x_2)**2 + (c_y - y_2)**2
    d_3 = (c_x - x_3)**2 + (c_y - y_3)**2
    d_4 = (c_x - x_4)**2 + (c_y - y_4)**2

    #r = np.max([d_1, d_2, d_3, d_4])
    r = np.median([d_1, d_2, d_3, d_4])
    r = np.ceil(np.sqrt(r) * 1.05).astype('int')    # multiply by a factor 1.05 to ensure it is bounding 

    return c_x, c_y, r 

def generate_h5_dataset(data_path, dataset_prefix, labels_to_classes_dictionary, outfile_path=None, shuffle_data=True, resize_height=224, resize_width=224, data_order='tf', bounding_box_info=None):

    if dataset_prefix is None:
        dataset_prefix = ''
    
    if outfile_path is None:
        outfile_path = "dataset_" + str(resize_height) + "_" + str(resize_width) + ".h5"

    # read all addresses and all 'label' folders
    all_filenames = []
    all_labels = []

    for label_int, label_str in labels_to_classes_dictionary.iteritems():
        label_dir = os.path.join(data_path, label_str)
        if os.path.isdir(label_dir) and os.path.exists(label_dir):
        
            filenames = glob.glob(os.path.join(label_dir, "*.*"))
            labels = [int(label_int) for filename in filenames]

            all_filenames.extend(filenames)
            all_labels.extend(labels)

    if shuffle_data:
        c = list(zip(all_filenames, all_labels))
        shuffle(c)
        all_filenames, all_labels = zip(*c)

    if data_order == 'th': 
        data_shape = (len(all_filenames), 3, resize_height, resize_width)
    elif data_order == 'tf':
        data_shape = (len(all_filenames), resize_height, resize_width, 3)

    # open a hdf5 file and create arrays
    hdf5_file = h5py.File(outfile_path, mode='w')

    if data_shape[0] > 0:
        hdf5_file.create_dataset(dataset_prefix + "_set_x", data_shape, np.uint8, 
                                 maxshape=(None, data_shape[1], data_shape[2], data_shape[3]),
			         chunks=data_shape)

	hdf5_file.create_dataset(dataset_prefix + "_set_y", (len(all_filenames), 10), np.float32, 
	                         maxshape=(None, 10), 
	                         chunks=(len(all_filenames), 10))

        #hdf5_file[dataset_prefix + "_set_y"][...] = all_labels

    # list of classes
    list_classes = [""] * len(labels_to_classes_dictionary)
    for label_int, label_str in labels_to_classes_dictionary.iteritems():
        list_classes[int(label_int)] = label_str 

    list_classes = [n.encode("ascii", "ignore") for n in list_classes]

    hdf5_file.create_dataset("list_classes", (len(list_classes),), 'S10')
    hdf5_file["list_classes"][...] = np.array(list_classes)

    # idx to filename
    idx_filenames = []
    hdf5_file.create_dataset(dataset_prefix + "_idx_filenames", (data_shape[0],), 'S100')

    for i in range(len(all_filenames)):
	if i % 100 == 0 and i > 1:
            print dataset_prefix + ' data: {}/{}'.format(i, len(all_filenames))
	filename = all_filenames[i]

	#print(filename)

	image_row = bounding_box_info[bounding_box_info.image_name == os.path.basename(filename)]

	c_x, c_y, r = get_center_and_bounding_radius(image_row)

	img = imageio.imread(filename)

	# normalize c_x, c_y, and r
	c_x = float(c_x) / float(img.shape[1])
	c_y = float(c_y) / float(img.shape[0])
	r = float(r) / float(img.shape[1])

	img = resize(img, output_shape=(resize_height, resize_width), preserve_range=True, anti_aliasing=True)
	#img = ndimage.imread(filename, flatten=False)
	#img = scipy.misc.imresize(img, size=(resize_height, resize_width))

        # if the data order is Theano, axis orders should change
        if data_order == 'th':
            img = np.rollaxis(img, 2)

        hdf5_file[dataset_prefix + "_set_x"][i, ...] = img[:,:,:3]

	y = np.zeros((1, 10))
	if all_labels[i] != 0:
	    y[0, 0] = 1                    # not a background
	    y[0, all_labels[i] + 3] = 1    # one-hot encoding of class label
        y[0, 1] = c_x
	y[0, 2] = c_y
	y[0, 3] = r

	hdf5_file[dataset_prefix + "_set_y"][i, ...] = y

	idx_filenames.append(filename)

    hdf5_file[dataset_prefix + "_idx_filenames"][...] = np.array(idx_filenames)

    hdf5_file.close()

 


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    data_dir = args.data_dir
    output_dir = args.output_dir
    classes = process_classes(args.classes)
    size = int(args.resize)
    bbox_csv_path = args.bbox_csv_path

    bounding_box_info = pd.read_csv(bbox_csv_path)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    indice_classes = {}
    for i, c in enumerate(classes):
        indice_classes[str(i)] = c 

    #for i, split in enumerate(['train', 'validation', 'test']):
    for i, split in enumerate(['train']):
	
        data_path = os.path.join(data_dir, split)
	if os.path.exists(data_path):
	    outfile_path = os.path.join(output_dir, split + "_" + str(size) + "_" + str(size) + ".hdf5")

	    if split == 'validation':
	        prefix = 'dev'
	    else:
	        prefix = split

            generate_h5_dataset(data_path, dataset_prefix=prefix, labels_to_classes_dictionary=indice_classes, outfile_path=outfile_path, resize_height=size, resize_width=size, bounding_box_info=bounding_box_info) 
