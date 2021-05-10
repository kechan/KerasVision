import os, gc, sys, time, pickle, pytz, multiprocessing, h5py, glob, re, PIL, base64, shutil, random, urllib, hashlib
import tempfile
import string
from pathlib import *
from functools import partial
from datetime import date, datetime, timedelta
from IPython.display import HTML
from io import BytesIO

def onColab(): return os.path.exists('/content')
bOnColab = onColab()

if bOnColab:
  from google.colab import auth
  auth.authenticate_user()
  print('Authenticated')
  
if bOnColab and not os.path.exists('/content/drive'):       # presence of /content indicates you are on google colab
  from google.colab import drive
  drive.mount('/content/drive')
  print('gdrive mounted')

import pandas as pd
import numpy as np

from google.cloud import storage

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.utils import Sequence

from tensorflow.keras.layers import Dense, Input, Embedding, LSTM, Reshape, Dropout, Activation, Dot
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import backend as K

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomTranslation, RandomRotation
from tensorflow.keras.layers.experimental.preprocessing import RandomCrop, RandomContrast, RandomZoom

from tensorflow.keras.applications import ResNet50

AUTO = tf.data.experimental.AUTOTUNE

from ipywidgets import interact, Checkbox, Button, Output, HBox, VBox, AppLayout, Label, Layout, Text, Textarea
from ipywidgets import HTML as IPyWidgetHTML     # conflict with "from IPython.display import HTML"

from IPython.display import Image, Audio

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.dates as mdates
import seaborn as sns
# sns.set(style='darkgrid', context='talk', palette='Dark2')
sns.set(rc={'figure.figsize': (11, 4)})

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['font.size'] = 16

project_name = 'CoinSee'

if bOnColab:
  home = Path('/content/drive/MyDrive')
  local = Path('/content')
  local_photos = local/'photos'

  data = home/project_name/'data'
  train_home = home/project_name/'training'
  tmp = home/project_name/'tmp'
  labels_dir = home/project_name/'labels'
  utils_path = home/project_name/'utils'
  models_dir = home/project_name/'model'

try:
  sys.path.insert(0, str(utils_path))

  from common_util import load_from_pickle, save_to_pickle, say_done
  from common_util import plot_training_loss, plot_loss_accuracy, plot_loss_and_metrics, combine_history
  from small_fastai_utils import join_df 

  from common_util import isNone_or_NaN
  from common_util import image_d_hash, tf_image_d_hash
  from common_util import count_photos
  from common_util import join_filter_drop_df, tf_dataset_peek

  from common_util import get_image_id_from_image_name
except Exception as e:
  print(e)
  print("Not installing rlp_dataloader, common_util and small_fastai_utils")

print("\u2022 Using TensorFlow Version:", tf.__version__)

pd.set_option('display.max_colwidth', None)

