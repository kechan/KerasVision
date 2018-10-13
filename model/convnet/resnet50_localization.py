import os, PIL
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

# Custom Activation for exp(X)

class Exp(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Exp, self).__init__(activation, **kwargs)
        self.__name__ = 'exp_activation'

def exp_activation(X):
  return K.exp(X)


def resnet50_localization(input_shape=None, conv_base_source=None, params=None, ModelType=None):
    ''' Build a ResNet50 based convnet that output classification and bounding box info 

    Parameters
    ----------
    input_shape : Input shape of image e.g. (224, 224, 3) 
    conv_base_source : source model file (h5) path from which the 'resnet50' layer will be used
    params : others

    Returns
    -------
    Keras model

    '''
    
    get_custom_objects().update({'exp_activation': Exp(exp_activation)})

    if conv_base_source is None:
        conv_base = ResNet50(weights='imagenet', include_top=False) 
    else:
        src_model = load_model(conv_base_source)
        conv_base = src_model.get_layer('resnet50')

    if input_shape is None:
        X_input = Input((None, None, 3))
    else:
        X_input = Input(input_shape)

    X = conv_base(X_input)
     
    if params is not None and params.dropout is not None:
        dropout = params.dropout
    else:
        dropout = 0.0
 
    p_o = Dropout(dropout)(X)
    p_o = Conv2D(1, (1, 1), activation='sigmoid', name='object_prediction')(p_o)   #(X)

    b_x = Dropout(dropout)(X)
    b_x = Conv2D(1, (1, 1), activation='sigmoid', name='b_x')(b_x)                 #(X)

    b_y = Dropout(dropout)(X)
    b_y = Conv2D(1, (1, 1), activation='sigmoid', name='b_y')(b_y)                 #(X)

    b_r = Dropout(dropout)(X)
    b_r = Conv2D(1, (1, 1))(b_r)                                                   #(X)
    b_r = Activation(exp_activation, name='b_r')(b_r)

    p_c = Dropout(dropout)(X)
    p_c = Conv2D(6, (1, 1), activation='softmax', name='p_c')(p_c)                 #(X)

    out = Concatenate(axis=-1)([p_o, b_x, b_y, b_r, p_c])
    
    out = Reshape((-1,))(out)

    if ModelType is None:
        return Model(inputs=X_input, outputs=out)
    else:
        return ModelType(inputs=X_input, outputs=out)

def resnet50_localization_regression(input_shape=None, conv_base_source=None, params=None, ModelType=None):
    ''' Build a ResNet50 based convnet that output classification and bounding box related info 
        The custom layer EvaluateOutputs is needed as final layer to get the observed prediction.

    Parameters
    ----------
    input_shape : Input shape of image e.g. (224, 224, 3) 
    conv_base_source : source model file (h5) path from which the 'resnet50' layer will be used
    params : others

    Returns
    -------
    Keras model

    '''
    if conv_base_source is None:
        conv_base = ResNet50(weights='imagenet', include_top=False) 
    else:
        src_model = load_model(conv_base_source)
        conv_base = src_model.get_layer('resnet50')

    if input_shape is None:
        X_input = Input((None, None, 3))
    else:
        X_input = Input(input_shape)

    X = conv_base(X_input)

    if params is not None and params.dropout is not None:
        dropout = params.dropout
    else:
        dropout = 0.0

    # model with linear activation output
    X = Dropout(dropout)(X)
    out = Conv2D(10, (1, 1), name='t_output')(X)
    out = Reshape((-1,))(out)

    if ModelType is None:
        return Model(inputs=X_input, outputs=out)
    else:
        return ModelType(inputs=X_input, outputs=out)


class EvaluateOutputs(keras.layers.Layer):
    ''' This is a custom layer designed to be used as prediction outputs for resnet50_localization_regression()
        See resnet50_localization_regression_eval
    '''

    def __init__(self, **kwargs):
        super(EvaluateOutputs, self).__init__(**kwargs)
        
    def call(self, inputs):
        p_o = K.sigmoid(inputs[:, 0:1])
        p_c = K.softmax(inputs[:, 4:])
        
        b_xy = K.sigmoid(inputs[:, 1:3])
        b_r = K.exp(inputs[:, 3:4])
        
        return K.concatenate([p_o, b_xy, b_r, p_c], axis=-1)
    
    def compute_output_shape(self, input_shape):
        #print(input_shape)
        shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape(shape)
        
    @classmethod
    def from_confg(cls, config):
        return cls(**config)

def install_head_resnet50_localization_regression(model, ModelType=None):
    out = EvaluateOutputs()(model.output)
    if ModelType is None:
        return Model(inputs = model.input, outputs = out) 
    else:
        return ModelType(inputs = model.input, outputs = out)


class ZoomAndFocusModel(keras.Model):
    def __init__(self, padding_ratio=1.0, **kwargs):
        super(ZoomAndFocusModel, self).__init__(name='zoom_and_focus_model', **kwargs)
        self.padding_ratio = padding_ratio

    def predict_with_zoom_and_focus(self, x, **kwargs):
        y_pred = super(ZoomAndFocusModel, self).predict(x, **kwargs)
        assert x.shape[0] == 1, "Batch prediction not supported."

        height, width = x.shape[1], x.shape[2]   # this is expected to conform to the shape of model.input

        orig_img_size = np.array([height, width]).reshape((1, 2))

        c_x = y_pred[..., 1:2]
        c_y = y_pred[..., 2:3]
        c_r = y_pred[..., 3:4]

        box = np.concatenate([c_y - c_r, c_x - c_r, c_y + c_r, c_x + c_r], axis=-1)   # top, left, bottom, right 
        box_size = np.concatenate([2. * c_r, 2. * c_r], axis=-1)	

        padding_size = box_size * self.padding_ratio
        crop_coords = np.concatenate([
                                      np.maximum(0, box[..., :2] - padding_size), 
                                      np.minimum(1.0, box[...,2:] + padding_size)
                                     ], axis=-1)

        crop_size = np.stack([crop_coords[..., 2] - crop_coords[..., 0], crop_coords[..., 3] - crop_coords[..., 1]], axis=-1)

	# get the absolute coordinate in order to crop the actual image
        abs_crop_coords = np.round((crop_coords * np.concatenate([orig_img_size, orig_img_size], axis=-1))).astype(np.int)

	# crop the box out from the original image
        x_ = x[0]
        cropped_x = x_[abs_crop_coords[0,0]:abs_crop_coords[0,2], abs_crop_coords[0,1]:abs_crop_coords[0,3], :]
        img = PIL.Image.fromarray((cropped_x*225.).astype(np.uint8))
        resize_img = img.resize((height, width), PIL.Image.BICUBIC)   # resize back to what the model input expects
        cropped_resized_x = np.array(resize_img)

	# make a prediction on the cropped and resized image
        cropped_resize_y_pred = self.predict(cropped_resized_x[None]/255.)

        	
	# Modify the objectness and class prediction based on that of cropped image
        y_pred[..., 4:] = cropped_resize_y_pred[..., 4:]
        y_pred[..., 0:1] = cropped_resize_y_pred[..., 0:1]

        # TODO: Figure how to relax this requirement
        # crop_size needs to be a square, if it isnt, we don't update bounding box coordinate
        if np.abs(np.squeeze(crop_size[..., 0] - crop_size[..., 1])) < K.epsilon():
	    # transform the prediction back to coordinate system of the original image
            # print(cropped_resize_yhat[0])
	
            cropped_resize_y_pred[..., 1:3] = cropped_resize_y_pred[..., 1:3] * crop_size + np.array([crop_coords[0, 1], crop_coords[0, 0]])
            cropped_resize_y_pred[..., 3:4] = cropped_resize_y_pred[..., 3:4] * crop_size[..., 0:1] 

            y_pred[..., 1:3] = cropped_resize_y_pred[..., 1:3]
            y_pred[..., 3:4] = cropped_resize_y_pred[..., 3:4]
 
        return y_pred

# For error analysis
def L_acc_by_parts(y_true, y_pred, iou_score_threshold=0.6):
    '''
    Return numpy array of {0., 1.} intermediate indicators for further calculating error/accuracy attributable to various aspect of the predictions.
  
    Parameters:
    y_true:   ground truth in shape of [batch, ?]
    y_pred:   prediction in shape of [batch, ?]
    '''
  
    # Objectness confidence, (background vs. an object)
  
    y_true_conf = y_true[..., 0]
    y_pred_conf = np.round(y_pred[..., 0])
  
    # Class (must be one-hot encoded)
    y_true_classes = y_true[..., 4:]
    y_pred_classes = y_pred[..., 4:]

    classes_accuracy = np.equal(np.argmax(y_true_classes, axis=-1), np.argmax(y_pred_classes, axis=-1)).astype(np.float32)
  
    # IOU driven accuracy
    true_xy = y_true[..., 1:3]
    pred_xy = y_pred[..., 1:3]
  
    true_r = y_true[..., 3:4]
    pred_r = y_pred[..., 3:4]
  
    true_mins = true_xy - true_r    # top, left, bottom, right coordinates
    true_maxes = true_xy + true_r
    
    pred_mins = pred_xy - pred_r
    pred_maxes = pred_xy + pred_r
  
    intersect_mins = np.maximum(pred_mins, true_mins)
    intersect_maxes = np.minimum(pred_maxes, true_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_areas = 4. * pred_r[..., 0] * pred_r[..., 0]   # a square
    true_areas = 4. * true_r[..., 0] * true_r[..., 0]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas
    
    iou_accuracy = (iou_scores > iou_score_threshold).astype(np.float32)

    # Total joint accuracy
    joint_accuracy = (y_true_conf * y_pred_conf * classes_accuracy * iou_accuracy) + \
                         ((1.0 - y_true_conf) * (1.0 - y_pred_conf)) 
  
    # Accuracy attributable to IOU
  
    return joint_accuracy, iou_accuracy, iou_scores, classes_accuracy, y_true_conf, y_pred_conf 




