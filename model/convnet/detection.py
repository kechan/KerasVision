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

from keras.applications import ResNet50, MobileNet
import keras.backend as K

from keras.utils.generic_utils import get_custom_objects

from .localization import regression_model_with_input_shape

def detection_regression(application='ResNet50', input_shape=None, dropout=0.0, avg_pool_stride=(1, 1), weights=None):

  ''' Build a ResNet backboned detection model 
  
  Parameters
  ----------
  input_shape: Input shape of image, e.g. (453, 453, 3)
  weights: path to a .h5 model file

  Returns
  -------
  the Keras model
  '''


  model = regression_model_with_input_shape(application=application, input_shape=input_shape, dropout=dropout, avg_pool_stride=avg_pool_stride)
  if weights is not None:
    model.load_weights(weights)

  #model = model.with_avg_pool_stride_one()

  return model

def preprocess_true_boxes(set_y, max_boxes=1, conv_height=9, conv_width=9):  
  ''' Output a sparse representation of transformed set_y which is original stored as a list of boxes (of varying length) per image. 
  (x,y) for each grid cell is t_x, t_y that is the pre-activation output of the model.
  r for each grid is t_r (pre-activation output of the model)
 

  Parameters:
  -----------
  set_y: numpy array with shape (num_sample,) 
         or (num_sample, box_params) in case if this is the train set y from localization 
  max_boxes: the maximum number of boxes allow, we will zero-pad if the image has less than this number of box 
  conv_height: height of conv features (number of rows)
  conv_width: width of conv features (number of cols)

  Returns:
  --------
  numpy array (sparse) with shape (sample, conv_height, conv_width, box_params) 

  0th param is 0 or 1 (aka detectors_mask)
  1-2 is (x,y) of the box in conv cell local coordinate system (need conversion "(sigmoid(x) + conv_index) / conv_dims" to image coord  
  3 is side length of the box in conv coordinate system (need conversion "exp(r) / conv_dims" to image coord
  4-end is one-hot encoding of class/label.

  '''

  if set_y.ndim == 1:   # detection format
    detection_format = True
  else:
    detection_format = False

  # boxes: numpy array (sample, max_boxes, box_params), coordinates and dimensions are normalized r.p.t. original image
  boxes = np.zeros((len(set_y), max_boxes, 9), dtype=np.float32)

  for i, y in enumerate(set_y):

    if detection_format:
      y = y.reshape((-1, 9))

      if max_boxes >= y.shape[0]:
        zero_padding = np.zeros( (max_boxes - y.shape[0], 9), dtype=np.float32)
        boxes[i] = np.vstack((y, zero_padding))
      else:   # > max_boxes, truncate
        boxes[i] = y[:max_boxes]
    else:
      # for localization
      if y[0] == 1.0:
        y = np.reshape(y[1:], (1, 9))
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
      
        #print("box: {}".format(box))
        i = np.floor(box[1]).astype('int')    # y coordinate (row for matrix)
        j = np.floor(box[0]).astype('int')
    
        _detectors_mask[i, j, 0] = 1
      
        _x = box[0] - j
        _y = box[1] - i
        _r = box[2]

	#print("_x: {}".format(_x))
	#print("_y: {}".format(_y))
      
        tmp = [np.log((_x / (1. - _x + K.epsilon())) + K.epsilon()),   # sigmoid^{-1}
               np.log((_y / (1. - _y + K.epsilon())) + K.epsilon()),   
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

  # since the shape has to match with the model output which is (n, conv_height, conv_width, 10) for 1 box per cell, we will need to 
  # reshape this, and doing so generally for >1 boxes per cell. 
  set_y_final = set_y_final.reshape((set_y_final.shape[0], conv_height, conv_width, -1))

  return set_y_final

def generate_conv_index_conv_dim(conv_height, conv_width):

  ''' generate a grid mesh array such that element with index [0, i, j, 0, 2] is (x = j, y = i) in cg coordinate

   parameters
   ----------
   conv_height: height of conv features (number of rows)
   conv_width: width of conv features (number of cols)

   returns
   -------
   A tensor that is the grid mesh array described above.

   e.g. 
   
   conv_index, conv_dims = generate_conv_index_conv_dim(9, 9)

   conv_index[:, 8, 2, :, :]    # returns Tensor([[[2., 8.]]])

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


def generate_numpy_conv_index_conv_dim(conv_height, conv_width):
  ''' generate a grid mesh array such that element with index [0, i, j, 0, 2] is (x = j, y = i) in cg coordinate 

   parameters
   ----------
   conv_height: height of conv features (number of rows)
   conv_width: width of conv features (number of cols)

   returns
   -------
   A numpy that is the grid mesh array described above.

   e.g. 
   
   conv_index, conv_dims = generate_numpy_conv_index_conv_dim(9, 9)

   conv_index[:, 8, 2, :, :]    # returns array([[[2., 8.]]], dtype=float32)
  
  '''

  conv_dims = conv_height, conv_width
  conv_height_index = np.arange(0, stop=conv_dims[0])
  conv_width_index = np.arange(0, stop=conv_dims[1])
  conv_height_index = np.tile(conv_height_index, [conv_dims[1]])

  conv_width_index = np.tile(np.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
  conv_width_index = np.reshape(np.transpose(conv_width_index), (-1,))
  conv_index = np.transpose(np.stack([conv_height_index, conv_width_index]))
  conv_index = np.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
  #conv_index = K.cast(conv_index, K.floatx())
  conv_index = conv_index.astype(np.float32)

  #conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 2]), K.floatx())
  conv_dims = np.reshape(conv_dims, (1, 1, 1, 1, 2)).astype(np.float32)
  
  return conv_index, conv_dims


def sparse_representation(set_y, max_boxes=1, conv_height=9, conv_width=9):
  ''' return sparse representation of set_y which is original stored as a list of boxes (of varying length) per image.

  Parameters:
  -----------
  set_y: numpy array with shape (num_sample,)
  max_boxes: the maximum number of boxes allow, we will zero-pad if the image has less than this number of box 
  conv_height: height of conv features (number of rows)
  conv_width: width of conv features (number of cols)

  
  Return:
  -------
  numpy array (sparse) with shape (sample, conv_height, conv_width, box_params) 

  0th param is 0 or 1 (aka detectors_mask)
  1-2 is (x,y) of the box normalized to orginal image size  
  3 is side length of the box normalized to orginal image size 
  4-end is one-hot encoding of class/label.

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
      
        #box_class = box[3:]
        scaled_box = box[:2] * np.array([conv_width, conv_height])                # scale coordinate and size r.p.t. conv feature space
      
        i = np.floor(scaled_box[1]).astype('int')    # y coordinate (row for matrix)
        j = np.floor(scaled_box[0]).astype('int')
    
        _detectors_mask[i, j, 0] = 1
      
        #_x = box[0] - j
        #_y = box[1] - i
        #_r = box[2]
      
        #tmp = [np.log(_x / (1. - _x)),   # sigmoid^{-1}
        #       np.log(_y / (1. - _y)),   
        #       np.log(_r)]
        #tmp.extend(list(box_class))
      
        #adjusted_box = np.array(tmp, dtype=np.float32)
      
        _matching_true_boxes[i, j, 0] = box

    detectors_mask[k] = _detectors_mask
    matching_true_boxes[k] = _matching_true_boxes

  #matching_true_boxes: np array (sample, conv_height, conv_width, 1, box_params) providing coordinates, size, and class info 
  detectors_mask = np.array(detectors_mask)
  matching_true_boxes = np.array(matching_true_boxes)

  # combine detectors_mask and matching_true_boxes into a single final numpy array, and this will be the ultimate train_set_y going forward
  set_y_sparse = np.concatenate([detectors_mask, matching_true_boxes], axis=-1)

  # since the shape has to match with the model output which is (n, conv_height, conv_width, 10) for 1 box per cell, we will need to 
  # reshape this, and doing so generally for >1 boxes per cell. 
  set_y_sparse = set_y_sparse.reshape((set_y_sparse.shape[0], conv_height, conv_width, -1))

  return set_y_sparse


def eval(model_output, image_shape, max_boxes=1, score_threshold=0.6, iou_threshold=0.5):
  ''' Full evaluation on the model outputs by 
    
      1) confidence score: filter out all boxes whose confidence is below the score_threshold
      2) Non-max suppression: Suppress all boxes with IOU > iou_threshold with the highest confidence box 

  Parameters:
  -----------
  model_output: prediction output from model with last activation (Evaluate) layer, whose output is ready interpretable (grid independent)  
  image_shape: original image shape 
  max_boxes: max. number of boxes allowed
  score_threshold: Used to filter boxes
  iou_threshold: a parameter used in the non-max suppression algorithm
 
  Returns:
  -------
  out_boxes: list of boxes (coordinates, x, y, r) 
  out_scores: list of confidence scores
  out_classes: list of predicted classes (not one-hot encoded)

  '''

  boxes = yolo_boxes_to_corners(model_output[..., 1:3], model_output[..., 3:4])
  out_boxes, out_scores, out_classes = yolo_filter_boxes(boxes, model_output[..., 0:1], model_output[..., 4:], threshold=score_threshold)

  # scale boxes back to original image shape 
  height = image_shape[0]
  width = image_shape[1]

  image_dims = np.stack([height, width, height, width])
  image_dims = np.reshape(image_dims, (1, 4))

  out_boxes = out_boxes * image_dims

  nms_index = tf.image.non_max_suppression(out_boxes, out_scores, max_boxes, iou_threshold=iou_threshold)
  nms_index = K.eval(nms_index)

  out_boxes = out_boxes[nms_index]
  out_scores = out_scores[nms_index]
  out_classes = out_classes[nms_index]
  
  return out_boxes, out_scores, out_classes

def yolo_boxes_to_corners(box_xy, box_r):
  """Convert box predictions to bounding box corners."""
  box_mins = box_xy - box_r
  box_maxes = box_xy + box_r

  return np.concatenate([
        box_mins[..., 1:2],  # y_min (aka top)
        box_mins[..., 0:1],  # x_min (aka left)
        box_maxes[..., 1:2],  # y_max (aka bottom)
        box_maxes[..., 0:1]  # x_max (aka right)
    ], axis=-1)

def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=0.6):
  """Filter YOLO boxes based on object and class confidence."""

  box_scores = box_confidence * box_class_probs
  box_classes = np.argmax(box_scores, axis=-1)
  box_class_scores = np.max(box_scores, axis=-1)
  
  prediction_mask = box_class_scores >= threshold
  
  boxes = boxes[prediction_mask]               # boxes = tf.boolean_mask(boxes, prediction_mask)
  scores = box_class_scores[prediction_mask]   # scores = tf.boolean_mask(box_class_scores, prediction_mask)
  classes = box_classes[prediction_mask]       # classes = tf.boolean_mask(box_classes, prediction_mask)
  
  return boxes, scores, classes


