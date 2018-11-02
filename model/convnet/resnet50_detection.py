import os, PIL
from PIL import ExifTags

import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Conv2D, Dropout, Concatenate, Reshape
from keras.layers import BatchNormalization, Activation
from keras.layers import Input, Lambda

from keras.applications import ResNet50
import keras.backend as K

from keras.utils.generic_utils import get_custom_objects

from .resnet50_localization import regression_model_with_input_shape

def resnet50_detection_regression(input_shape=None, dropout=0.0, weights=None):

  ''' Build a ResNet backboned detection model 
  
  Parameters
  ----------
  input_shape: Input shape of image, e.g. (453, 453, 3)
  weights: path to a .h5 model file

  Returns
  -------
  the Keras model
  '''


  model = regression_model_with_input_shape(input_shape, dropout=dropout)
  if weights is not None:
    model.load_weights(weights)

  model = model.with_avg_pool_stride_one()

  return model

def preprocess_true_boxes(set_y, max_boxes=1, conv_height=9, conv_width=9):  
  ''' Output a numpy array with shape (num_sample, max_boxes, box_params) from set_y which is 1-dim array of objects of varying size 
  (depending on the # of boxes in that sample of image. 

  Parameters:
  -----------
  set_y: numpy array with shape (num_sample,)
  max_boxes: the maximum number of boxes allow, we will zero-pad if the image has less than this number of box 
  conv_height: height of conv features (number of rows)
  conv_width: width of conv features (number of cols)

  Returns:
  --------
  '''

  # boxes: numpy array (sample, max_boxes, box_params), coordinates and dimensions are normalized r.p.t. original image
  boxes = np.zeros((len(set_y), max_boxes, 9), dtype=np.float32)

  for i, y in enumerate(set_y):
    y = y.reshape((-1, 9))
    zero_padding = np.zeros( (max_boxes - y.shape[0], 9), dtype=np.float32)
    boxes[i] = np.vstack((y, zero_padding))
    
  detectors_mask = [0 for i in range(len(boxes))]          # placeholders for an eventual tensor construction
  matching_true_boxes = [0 for i in range(len(boxes))]      

  for k, boxz in enumerate(boxes):
    num_box_params = boxz.shape[1]
    _detectors_mask = np.zeros((conv_height, conv_width, 1, 1), dtype=np.float32)                    # 9 x 9 x num_anchors x 1 (where num_anchors == 1)
    _matching_true_boxes = np.zeros((conv_height, conv_width, 1, num_box_params), dtype=np.float32)  # 9 x 9 x num_anchors x 9 
  
    for box in boxz:
      if np.sum(box[3:]) > 0:       # skip if this is a zero pad (aka not a box)
      
        box_class = box[3:]
        box = box[0:3] * np.array([conv_width, conv_height, conv_width])                # scale coordinate and size r.p.t. conv feature space
      
        i = np.floor(box[1]).astype('int')    # y coordinate (row for matrix)
        j = np.floor(box[0]).astype('int')
    
        _detectors_mask[i, j, 0] = 1
      
        _x = box[0] - j
        _y = box[1] - i
        _r = box[2]
      
        tmp = [np.log(_x / (1. - _x)),   # sigmoid^{-1}
               np.log(_y / (1. - _y)),   
               np.log(_r)]
        tmp.extend(list(box_class))
      
        adjusted_box = np.array(tmp, dtype=np.float32)
      
        _matching_true_boxes[i, j, 0] = adjusted_box

    detectors_mask[k] = _detectors_mask
    matching_true_boxes[k] = _matching_true_boxes

  #detectors_mask: numpy array (sample, conv_height, conv_width, 1, 1) of 0 and 1, with 1 indicated presence of a true box
  #matching_true_boxes: np array (sample, conv_height, conv_width, 1, box_params) providing coordinates, size, and class info 
  #                     of the box at that sample & location

  detectors_mask = np.array(detectors_mask)
  matching_true_boxes = np.array(matching_true_boxes)

  # combine detectors_mask and matching_true_boxes into a single final numpy array, and this will be the ultimate train_set_y going forward
  set_y_final = np.concatenate([detectors_mask, matching_true_boxes], axis=-1)

  # Since the shape has to match with the model output which is (N, conv_height, conv_width, 10) for 1 box per cell, we will need to 
  # reshape this, and doing so generally for >1 boxes per cell. 
  set_y_final = set_y_final.reshape((set_y_final.shape[0], 9, 9, -1))

  return set_y_final

def generate_conv_index_conv_dim(conv_height, conv_width):

  ''' generate a grid mesh array such that element with index [0, i, j, 0, 2] is [i, j]

   Parameters
   ----------
   conv_height: height of conv features (number of rows)
   conv_width: width of conv features (number of cols)

   Returns
   -------
   A tensor that is the grid mesh array described above.

  '''

  conv_dims = K.constant([conv_height, conv_width])
  conv_height_index = K.arange(0, stop=conv_dims[0])
  conv_width_index = K.arange(0, stop=conv_dims[1])
  conv_height_index = tf.tile(conv_height_index, [conv_dims[1]])

  conv_width_index = tf.tile(K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
  conv_width_index = K.flatten(K.transpose(conv_width_index))

  conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
  conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
  conv_index = K.cast(conv_index, K.floatx())

  conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.floatx())
  
  return conv_index, conv_dims
