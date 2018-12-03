import tensorflow as tf
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

from model.convnet.resnet50_detection import generate_conv_index_conv_dim

conv_height, conv_width = 9, 9
conv_index, conv_dims = generate_conv_index_conv_dim(conv_height, conv_width)


def inv_preprocess_true_boxes(detectors_mask, matching_true_boxes, conv_index, conv_dims, max_boxes=1):
  
  max_boxes_tensor = K.variable(max_boxes, dtype='int32')
  K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
  
  mask = tf.squeeze(detectors_mask, axis=-1)    #.astype(np.bool)
  
  box_xy = (K.sigmoid(matching_true_boxes[..., :2]) + conv_index) / conv_dims
  box_r = K.exp(matching_true_boxes[..., 2:3]) / conv_dims[..., 0]

  boxes_y = tf.concat([box_xy, box_r, matching_true_boxes[..., 3:]], axis=-1)
  boxes_y = boxes_y * detectors_mask
  
  N = K.shape(matching_true_boxes)[0]
  
  #recovered_boxes = K.zeros((N, max_boxes, 9), dtype=K.floatx())              # sample, n_boxes, n_params
  #recovered_boxes_ist = []

  def padzero(x):
    matching_true_box = x[0]
    mask_ = x[1]

    bb = tf.boolean_mask(matching_true_box, mask_)
    zero_padding = K.zeros( (max_boxes_tensor - K.shape(bb)[0], 9), dtype=K.floatx())

    bb = K.concatenate([bb, zero_padding], axis=0)

    return bb

  recovered_boxes = tf.map_fn(padzero, (boxes_y, mask), dtype=K.floatx())

  '''
  for matching_true_box, mask_  in zip(boxes_y, mask):
    #matching_true_box = boxes_y[k]
    #mask_ = mask[k]  
    #bb = matching_true_box[mask_]
    bb = tf.boolean_mask(matching_true_box, mask_)
      
    zero_padding = K.zeros( (max_boxes_tensor - K.shape(bb)[0], 9), dtype=K.floatx())
  
    bb = K.concatenate([bb, zero_padding], axis=0)

    #recovered_boxes[k] = K.stack([bb, zero_padding], axis=0)
    recovered_boxes_list.append(bb)
    
  recovered_boxes = K.stack(recovered_boxes_list)  
  ''' 

  return recovered_boxes


def transform_predicted_from_t_to_actual(t_pred, conv_index, conv_dims):
  pred_xy = (K.sigmoid(t_pred[..., 1:3]) + conv_index) / conv_dims
  pred_r = K.exp(t_pred[..., 3:4]) / conv_dims[..., 0]
  
  return pred_xy, pred_r


def obj_detection_loss(y_true, y_pred):

  max_boxes = 20
  object_scale = 5.
  no_object_scale = 1.
  class_scale = 1.
  coordinates_scale = 1.

  N = K.shape(y_true)[0]       # number of samples in batch
  
  # retrieve the detectors_mask and matching_true_boxes from y_true
  masks_and_true_boxes = K.reshape(y_true, [N, 9, 9, 1, -1])
  detectors_mask = masks_and_true_boxes[..., 0:1]
  matching_true_boxes = masks_and_true_boxes[..., 1:]
  
  # reshape y_pred as well, we call these t parameters as they are before final activation values
  t_pred = K.reshape(y_pred, [N, 9, 9, 1, -1])
  
  # loss related to classification
  matching_classes = matching_true_boxes[..., 3:]
  y_pred_class = K.softmax(t_pred[..., 4:])

  classification_loss = K.sum(class_scale * detectors_mask * K.square(matching_classes - y_pred_class), axis=(-4, -3, -2, -1))
  
  # loss related to coordinates
  matching_box_coord = matching_true_boxes[..., :3]
  y_pred_coord = t_pred[..., 1:4]

  coordinates_loss = K.sum(coordinates_scale * detectors_mask * K.square(matching_box_coord - y_pred_coord), axis=(-4, -3, -2, -1))
  
  # get a box tensor whose 2nd dimension list the individual boxes  
  
  boxes = inv_preprocess_true_boxes(detectors_mask, matching_true_boxes, conv_index, conv_dims, max_boxes=max_boxes)
  
  boxes_shape = K.shape(boxes)
  boxes = K.reshape(boxes, [boxes_shape[0], 1, 1, 1, boxes_shape[1], boxes_shape[2]])
  
  pred_xy, pred_r = transform_predicted_from_t_to_actual(t_pred, conv_index, conv_dims)
  pred_xy = K.expand_dims(pred_xy, 4)
  pred_r = K.expand_dims(pred_r, 4)

  true_xy, true_r = boxes[..., 0:2], boxes[..., 2:3]

  # Find IOU of each predicted box with each ground truth box.
  pred_mins = pred_xy - pred_r
  pred_maxes = pred_xy + pred_r

  true_mins = true_xy - true_r
  true_maxes = true_xy + true_r

  intersect_mins = K.maximum(pred_mins, true_mins)
  intersect_maxes = K.minimum(pred_maxes, true_maxes)
  intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
  intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

  pred_areas = 4. * pred_r[..., 0] * pred_r[..., 0]   # a square
  true_areas = 4. * true_r[..., 0] * true_r[..., 0]

  union_areas = pred_areas + true_areas - intersect_areas
  iou_scores = intersect_areas / union_areas

  # Best IOUs for each location.
  best_ious = K.max(iou_scores, axis=4)  # Best IOU scores.
  best_ious = K.expand_dims(best_ious)

  # A detector has found an object if IOU > thresh for some true box.
  object_detections = K.cast(best_ious > 0.6, K.dtype(best_ious))
  
  no_object_weights = no_object_scale * (1 - object_detections) * (1 - detectors_mask)
  no_objects_loss = no_object_weights * K.square(-K.sigmoid(t_pred[..., 0:1]))

  objects_loss = object_scale * detectors_mask * K.square(1 - K.sigmoid(t_pred[..., 0:1]))

  confidence_loss = K.sum(objects_loss + no_objects_loss, axis=(-4, -3, -2, -1))
  
  
  total_loss = 0.5 * (confidence_loss + classification_loss + coordinates_loss)
  
  #total_loss = 0.5 * (classification_loss + coordinates_loss)
  
  return total_loss
