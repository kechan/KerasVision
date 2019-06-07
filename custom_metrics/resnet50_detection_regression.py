import tensorflow as tf
import numpy as np

from keras import backend as K
from keras.metrics import categorical_accuracy
from keras.utils.generic_utils import get_custom_objects

from model.convnet.detection import generate_conv_index_conv_dim, generate_numpy_conv_index_conv_dim
from model.convnet.detection import yolo_boxes_to_corners as numpy_yolo_boxes_to_corners

from model.convnet.detection import eval as numpy_eval

from model.convnet.localization import EvaluateOutputs


# global hack, how to arrange to pass this into the metric method (which only takes y_true and y_pred as argument 


# ResNet50
image_shape = (453, 453)    
conv_height, conv_width = 9, 9

# mobilenet
# image_shape = (447, 447)
# conv_height, conv_width = 7, 7

conv_index, conv_dims = generate_conv_index_conv_dim(conv_height, conv_width)
np_conv_index, np_conv_dims = generate_numpy_conv_index_conv_dim(conv_height, conv_width)


def iou(box1, box2):

  ''' Calculate iou between box1 and box2 (which can be arrays of boxes)
  
  Parameters:
  -----------
  box1: shape of (N, 4) where N is number of 1st array of boxes, and 2nd dim is (top, left, bottom, right)
  box2: shape of (M, 4) where M is number of 2nd array of boxes, and 2nd dim is (top, left, bottom, right)
  
  Return:
  
  IOU among box1 and box2 with of shape (N, M)
  
  '''
  
  mins1 = box1[..., :2]
  maxes1 = box1[..., 2:]
  
  mins2 = box2[..., :2]
  maxes2 = box2[..., 2:] 
  
  intersect_mins = K.maximum(mins1, mins2)
  intersect_maxes = K.minimum(maxes1, maxes2)
  intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
  intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
  
  wh1 = maxes1 - mins1
  wh2 = maxes2 - mins2
      
  areas1 = wh1[..., 0] * wh1[..., 1]
  areas2 = wh2[..., 0] * wh2[..., 1]
    
  union_areas = areas1 + areas2 - intersect_areas
    
  iou_scores = intersect_areas / union_areas
    
  return iou_scores


def numpy_iou(box1, box2):
  ''' Calculate iou between box1 and box2 (which can be arrays of boxes)
  
  Parameters:
  -----------
  box1: shape of (N, 4) where N is number of 1st array of boxes, and 2nd dim is (top, left, bottom, right)
  box2: shape of (M, 4) where M is number of 2nd array of boxes, and 2nd dim is (top, left, bottom, right)
  
  Return:
  
  IOU among box1 and box2 with of shape (N, M)
  
  '''
  
  mins1 = box1[..., :2]
  maxes1 = box1[..., 2:]
  
  mins2 = box2[..., :2]
  maxes2 = box2[..., 2:] 
  
  intersect_mins = np.maximum(mins1, mins2)
  intersect_maxes = np.minimum(maxes1, maxes2)
  intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
  intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
  
  wh1 = maxes1 - mins1
  wh2 = maxes2 - mins2
      
  areas1 = wh1[..., 0] * wh1[..., 1]
  areas2 = wh2[..., 0] * wh2[..., 1]
    
  union_areas = areas1 + areas2 - intersect_areas
    
  iou_scores = intersect_areas / union_areas
    
  return iou_scores


def true_boxes_true_masks_true_classes(set_y): 
  ''' 
  return as a list of true_boxes and true_classes from set_y (that was already been transformed to be input-ready for training)
  '''

  if K.ndim(set_y) == 3:
    conv_index_local = conv_index[0]
    conv_dims_local = conv_dims[0]
  else:
    conv_index_local = conv_index
    conv_dims_local = conv_dims

  true_masks = set_y[..., 0]
  true_boxes = set_y[..., 1:4]
  true_classes = set_y[..., 4:]

  true_boxes = true_boxes[..., tf.newaxis, :]

  true_xy = (K.sigmoid(true_boxes[..., :2]) + conv_index_local) / conv_dims_local   # x, y from t->image
  true_r = K.exp(true_boxes[..., 2:3]) / conv_dims_local[..., 0]                    # r from t->image 

  true_boxes = yolo_boxes_to_corners(true_xy, true_r)

  true_classes = K.argmax(true_classes, axis=-1)
  
  return true_boxes, true_masks, true_classes


def sigmoid(x):
  return 1. / (1. + np.exp(-x))
  

def numpy_true_boxes_true_masks_true_classes(set_y):
  ''' 
  return as a list of true_boxes and true_classes from set_y (that was already been transformed to be input-ready for training)
  '''

  if set_y.ndim == 3:
    conv_index_local = np_conv_index[0]
    conv_dims_local = np_conv_dims[0]
  else:
    conv_index_local = np_conv_index
    conv_dims_local = np_conv_dims
  
  true_masks = set_y[..., 0]
  true_boxes = set_y[..., 1:4]
  true_classes = set_y[..., 4:]

  #true_boxes = true_boxes[:, :, :, np.newaxis, :]
  true_boxes = true_boxes[..., np.newaxis, :]

  true_xy = (sigmoid(true_boxes[..., :2]) + conv_index_local) / conv_dims_local   # x, y from t->image
  true_r = np.exp(true_boxes[..., 2:3]) / conv_dims_local[..., 0]                 # r from t->image 

  true_boxes = numpy_yolo_boxes_to_corners(true_xy, true_r)

  true_classes = np.argmax(true_classes, axis=-1)
  
  return true_boxes, true_masks, true_classes


def eval(model_output, image_shape, max_boxes=1, score_threshold=0.6, iou_threshold=0.5):
  ''' Full evaluation on the model outputs by 
    
      1) confidence score: filter out all boxes whose confidence is below the score_threshold
      2) Non-max suppression: Suppress all boxes with IOU > iou_threshold with the highest confidence box 

  Parameters:
  -----------
  model_output: prediction output from model with last activation (Evaluate) layer, whose output is ready interpretable (grid independent)  
  image_shape: original image shape 
  #max_boxes: max. number of boxes allowed
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

  image_dims = K.stack([height, width, height, width])
  image_dims = K.cast(K.reshape(image_dims, (1, 4)), K.floatx())

  out_boxes = out_boxes * image_dims

  #max_boxes_tensor = K.variable(max_boxes, dtype='int32')
  #K.get_session().run(tf.variables_initializer([max_boxes_tensor]))

  max_boxes_tensor = K.constant(max_boxes, dtype='int32')

  nms_index = tf.image.non_max_suppression(out_boxes, out_scores, max_boxes_tensor, iou_threshold=iou_threshold)

  out_boxes = K.gather(out_boxes, nms_index)
  out_scores = K.gather(out_scores, nms_index)
  out_classes = K.gather(out_classes, nms_index)

  return out_boxes, out_scores, out_classes

def yolo_boxes_to_corners(box_xy, box_r):
  """Convert box predictions to bounding box corners."""
  box_mins = box_xy - box_r
  box_maxes = box_xy + box_r
  
  return K.concatenate([
        box_mins[..., 1:2],  # y_min (aka top)
        box_mins[..., 0:1],  # x_min (aka left)
        box_maxes[..., 1:2],  # y_max (aka bottom)
        box_maxes[..., 0:1]  # x_max (aka right)
    ], axis=-1)

def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=0.6):
  """Filter YOLO boxes based on object and class confidence."""

  box_scores = box_confidence * box_class_probs
  box_classes = K.argmax(box_scores, axis=-1)
  box_class_scores = K.max(box_scores, axis=-1)
  
  prediction_mask = box_class_scores >= threshold
  
  boxes = tf.boolean_mask(boxes, prediction_mask)               # boxes = tf.boolean_mask(boxes, prediction_mask)
  scores = tf.boolean_mask(box_class_scores, prediction_mask)   # scores = tf.boolean_mask(box_class_scores, prediction_mask)
  classes = tf.boolean_mask(box_classes, prediction_mask)       # classes = tf.boolean_mask(box_classes, prediction_mask)
  
  return boxes, scores, classes

def numpy_d_acc(y_true, y_pred, max_boxes=1, image_shape=(453, 453), score_threshold=0.6, iou_threshold=0.5):

  ''' Calculate detection metrics for a single sample

  Parameters:
  ----------
  y_true:   post-eval model output, a numpy of shape (conv_height, conv_width, box_params)

  ''' 

  pred_boxes, pred_scores, pred_classes = numpy_eval(y_pred, image_shape, max_boxes=max_boxes, score_threshold=score_threshold, iou_threshold=iou_threshold)
   
  true_box, true_mask, true_class = numpy_true_boxes_true_masks_true_classes(y_true)
  true_mask = true_mask.astype('bool')

  true_box = true_box[true_mask]

  true_box = np.squeeze(true_box)

  true_class = true_class[true_mask]

  height, width = image_shape

  image_dims = np.stack([height, width, height, width])
  image_dims = np.reshape(image_dims, (1, 4))

  true_box = true_box * image_dims

  iou_matrix = numpy_iou(pred_boxes[:, np.newaxis, :], true_box[np.newaxis, :, :])

  if len(pred_boxes) > 0:
    # case of >2 predictions targeting 1 true box, we keep the one with highest IOU
    iou_matrix = iou_matrix * (iou_matrix - np.max(iou_matrix, axis=0, keepdims=True) >= 0).astype(np.float32)

    # case of >2 true boxes with 1 prediction 
    iou_matrix = iou_matrix * (iou_matrix - np.max(iou_matrix, axis=1, keepdims=True) >= 0).astype(np.float32)
  
  matched_prediction_idx, matched_truth_idx = np.nonzero(np.maximum(iou_matrix - 0.5, 0))

  # precision = # true positives / # prediction made (What proportion of positive identifications was actually correct?)

  tot_num_predictions = pred_boxes.shape[0]
  tot_num_ground_truths = true_box.shape[0]

  num_true_positives = np.sum(np.equal(pred_classes[matched_prediction_idx], true_class[matched_truth_idx]).astype('int'))

  # do these for numerical stability, # of true positives or # of predictions can be 0.
  num_true_positives = num_true_positives + K.epsilon()
  tot_num_predictions = tot_num_predictions + K.epsilon()
  tot_num_ground_truths = tot_num_ground_truths + K.epsilon()

  precision = num_true_positives / float(tot_num_predictions)

  # recall = # correct prediction / # of positive ground truth observations (What proportion of actual positives was identified correctly?)

  recall = num_true_positives / float(tot_num_ground_truths)

  f1 = 2 * (precision * recall) / (precision + recall)
  
  return f1 
  

def d_acc(x):
   ''' Calculate detection metrics for a single sample

   Parameters:
   x: a tuple for (y_true, y_pred) where y_pred is post-eval output from model

   '''

   max_boxes = 20    # TODO: this should be some sort of global constant

   y_true = x[0]
   y_pred = x[1]

   # convert y_true to list of boxes and classes

   pred_boxes, pred_scores, pred_classes = eval(y_pred, image_shape, max_boxes=max_boxes)

   true_box, true_mask, true_class = true_boxes_true_masks_true_classes(y_true)
   true_mask = K.cast(true_mask, dtype='bool')

   true_box = K.squeeze(true_box, axis=2)    # Note: for batch processing, axis=3
   true_box = tf.boolean_mask(true_box, true_mask)
   true_class = tf.boolean_mask(true_class, true_mask)

   height, width = image_shape

   image_dims = K.stack([height, width, height, width])
   image_dims = K.cast(K.reshape(image_dims, (1, 4)), K.floatx())

   true_box = true_box * image_dims

   # need to compare the list of box and class predictions between ground truth and prediction:
   # pred_boxes, pred_classes, true_box, true_class

   iou_matrix = iou(pred_boxes[:, tf.newaxis, :], true_box[tf.newaxis, :, :])

   # case of >2 predictions targeting 1 true box, we keep the one with highest IOU
   iou_matrix = iou_matrix * K.cast(iou_matrix - K.max(iou_matrix, axis=0, keepdims=True) >= 0.0, dtype=K.floatx())

   # case of >2 true boxes with 1 prediction 
   iou_matrix = iou_matrix * K.cast(iou_matrix - K.max(iou_matrix, axis=1, keepdims=True) >= 0.0, dtype=K.floatx())
   

   #matched_prediction_idx, matched_truth_idx = K.squeeze(np.nonzero(K.maximum(iou_matrix - 0.5, 0)))

   iou_matrix = K.maximum(iou_matrix - 0.5, 0)
   zero = K.constant(0, dtype=K.floatx())        # tf way of doing np.nonzero(...)
   where = K.not_equal(iou_matrix, zero)
   where = tf.where(where)

   matched_prediction_idx = where[..., 0]
   matched_truth_idx = where[..., 1]

   # calculate precision, recall and f1

   # precision = # true positives / # prediction made (What proportion of positive identifications was actually correct?)

   tot_num_predictions = K.cast(K.shape(pred_boxes)[0], K.floatx())
   tot_num_ground_truths = K.cast(K.shape(true_box)[0], K.floatx())


   num_true_positives = K.sum(K.cast(K.equal(
                                         K.gather(pred_classes, matched_prediction_idx), 
				         K.gather(true_class, matched_truth_idx)
				     ), 
                              K.floatx()))
   
   # do these for numerical stability, # of true positives or # of predictions can be 0.
   num_true_positives = num_true_positives + K.epsilon()
   tot_num_predictions = tot_num_predictions + K.epsilon()
   tot_num_ground_truths = tot_num_ground_truths + K.epsilon()

   precision = num_true_positives / tot_num_predictions

   # recall = # correct prediction / # of positive ground truth observations (What proportion of actual positives was identified correctly?)

   recall = num_true_positives / tot_num_ground_truths

   f1 = 2.0 * (precision * recall) / (precision + recall)
  
   return f1 

def D_acc(y_true, y_pred):
   # Detection Accuracy:

   # This is the strictest accuracy
   # result is counted as correct only if ALL the following are satisifed
   
   # 1) number of predicted bounding box must match
   # 2) all true boxes must be predicted for 
   # 3) a box match is only registered if the class is matched and with IOU > 0.6 
   # 4) there must not be any false positives (extraneous boxes) 


   # Doing this sample by sample for now (should try to further vectorize this.

   #true_boxes, true_masks, true_classes = true_boxes_true_masks_true_classes(dev_set_y_for_training)

   # convert t_pred into y_pred (output after Evaluate) 
   _y_pred = EvaluateOutputs()(y_pred)

   return tf.map_fn(d_acc, (y_true, _y_pred), dtype=K.floatx())


def numpy_D_acc(y_true, y_pred, max_boxes=1, image_shape=(453, 453), score_threshold=0.6, iou_threshold=0.5):

   accs = np.zeros((len(y_true), ), dtype=np.float32)
   
   for k in range(len(y_true)):
       acc = numpy_d_acc(y_true[k], y_pred[k], max_boxes=max_boxes, image_shape=image_shape, score_threshold=score_threshold, iou_threshold=iou_threshold)
       accs[k] = acc 

   return accs
