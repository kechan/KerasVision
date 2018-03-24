import argparse
import random
import os

from PIL import Image
from tqdm import tqdm
from shutil import copyfile

SIZE = 64

description = """Description:\n
Split and resize dataset for Keras data generator\
"""

'''
From: 
	data_dir > Label1 Folder
		 > Label2 Folder
		 > Label3 Folder

			etc.

To:
	output_dir > train > Label1 Folder
			     Label2 Folder
			     Label3 Folder
			     etc.

		   > validation > Label1 Folder
		   		  Label2 Folder
				  Label3 Folder
				  etc.

		   > test > Label1 Folder
		   	    Label2 Folder
			    Label3 Folder
			    etc.

'''

parser = argparse.ArgumentParser(description=description)
parser.add_argument('--data_dir', default='original', help="Directory with the dataset")
parser.add_argument('--output_dir', default='NxN4datagen', help="Where to write the new data")
parser.add_argument('--train_dev_test_ratio', default='0.6,0.2,0.2', help="Split ratio")
parser.add_argument('--resize', default='-1', help='resize image to this dimension, a square')

def resize_and_save(filename, output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    if size > 0:
        image = Image.open(filename)
        # Use bilinear interpolation instead of the default "nearest neighbor" method
        image = image.resize((size, size), Image.BILINEAR)
        image.save(os.path.join(output_dir, filename.split('/')[-1]))
    else:
        copyfile(filename, os.path.join(output_dir, os.path.basename(filename)))        

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # parse train/dev/test ratio 
    ratios = args.train_dev_test_ratio.split(",")    
    assert len(ratios) == 3
    train_ratio = float(ratios[0])
    dev_ratio = float(ratios[1])
    test_ratio = float(ratios[2])

    size = int(args.resize)

    data_dir = args.data_dir

    # get all label-folders
    labelfolders = os.listdir(data_dir)
    labelfolders = [os.path.join(data_dir, labelfolder) for labelfolder in labelfolders if not labelfolder.startswith('.')]

    for labelfolder in labelfolders:
        filenames = os.listdir(labelfolder)
	filenames = [os.path.join(data_dir, f) for f in filenames if f.endswith('.jpg') or f.endswith('.png')]
	random.shuffle(filenames)
	
	train_split = int(train_ratio * len(filenames))
	dev_split = int((train_ratio + dev_ratio) * len(filenames))

	train_filenames = filenames[:train_split]
	dev_filenames = filenames[train_split:dev_split]
	test_filenames = filenames[dev_split:]

	filenames = {'train': train_filenames,
                     'validation': dev_filenames,
                     'test': test_filenames}

        if not os.path.exists(args.output_dir):
	    os.mkdir(args.output_dir)
	else:
	    print("Warning: output dir {} already exists".format(args.output_dir))

	# Preprocess train, dev, test 
	for split in ['train', 'validation', 'test']:
	    output_dir_split = os.path.join(args.output_dir, split)
	    if not os.path.exists(output_dir_split):
	        os.mkdir(output_dir_split)
            output_dir_split_label = os.path.join(output_dir_split, labelfolder)
	    if not os.path.exists(output_dir_split_label):
	        os.mkdir(output_dir_split_label)

            for filename in tqdm(filenames[split]):
	        resize_and_save(filename, output_dir_split_label, size=size)
