""" Responsible for:

1) loading data from either hdf5 files or directories
2) perform (further) train, dev, test splits if necessary
3) 

"""

import os
import glob
import random
from data_util import *

#target_label = {"0": "UNK", "1": "5c", "2": "10c", "3": "25c", "4": "$1", "5": "$2"}
#label_target = reverse_dict(target_label)


def from_splitted_hdf5(data_dir):

    hdf5_files = []
    if os.path.isdir(data_dir):
        hdf5_files = glob.glob(os.path.join(data_dir, "*.hdf5")) 

    assert len(hdf5_files) == 3, "Expecting 3 files with prefix train_, validation_, and test_"
    assert len([f for f in hdf5_files if 'train_' in f]) == 1, "Expecting a file with train_*"
    assert len([f for f in hdf5_files if 'validation_' in f]) == 1, "Expecting a file with validation_*"
    assert len([f for f in hdf5_files if 'test_' in f]) == 1, "Expecting a file with test_*"

    # train set 
    train_file = [f for f in hdf5_files if 'train_' in f][0]
    train_set_x, train_set_y, _, _, _, _, classes = load_all_data(train_file)

    validation_file = [f for f in hdf5_files if 'validation_' in f][0]
    _, _, dev_set_x, dev_set_y, _, _, classes = load_all_data(validation_file)

    test_file = [f for f in hdf5_files if 'test_' in f][0]
    _, _, _, _, test_set_x, test_set_y, classes = load_all_data(test_file)

    return train_set_x, train_set_y, dev_set_x, dev_set_y, test_set_x, test_set_y, classes



def from_single_hdf5(data_dir):

    hdf5_file = glob.glob(os.path.join(data_dir, "*.hdf5")[0]

    train_set_x, train_set_y, dev_set_x, dev_set_y, test_set_x, test_set_y, classes = load_all_data(hdf5_file)
