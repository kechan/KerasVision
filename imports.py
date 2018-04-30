import os
import numpy as np
import pandas as pd
import h5py

from data.data_util import *
from data.load_data import from_splitted_hdf5
from data.build_extracted_feature_set_splitted_hdf5 import dump_h5

#%load_ext autoreload
#%autoreload 2

project = "CoinSee"

TOP_DATA_DIR = os.path.join("/Volumes/My Mac Backup/", project, "data")
DATA_DIR = os.path.join(TOP_DATA_DIR, "224x224_cropped_aug_hdf5")

#train_hdf5 = os.path.join(DATA_DIR, "VGG16_feature_train.hdf5")
#f = h5py.File(train_hdf5, mode='r+')
