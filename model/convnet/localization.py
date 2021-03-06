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
from keras.layers import AveragePooling2D

from keras.applications import ResNet50, MobileNet
import keras.backend as K

from keras.utils.generic_utils import get_custom_objects

# Custom Activation for exp(X)

class Exp(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Exp, self).__init__(activation, **kwargs)
        self.__name__ = 'exp_activation'

def exp_activation(X):
  return K.exp(X)


def localization(application='ResNet50', input_shape=None, weights=None, params=None, ModelType=None):
    ''' Build a ResNet50 based convnet that output classification and bounding box info 

    Parameters
    ----------
    input_shape : Input shape of image e.g. (224, 224, 3) 
    conv_base_source : source model file (h5) path from which the 'resnet50' layer will be used
    params : others
    ModelType: the Model type to return (can be a subclass of keras.models.Model

    Returns
    -------
    Keras model

    '''
    
    get_custom_objects().update({'exp_activation': Exp(exp_activation)})

    if weights is None:
        if application == 'ResNet50':
            conv_base = ResNet50(weights='imagenet', include_top=False) 
        elif application == 'MobileNet':
            conv_base = MobileNet(weights='imagenet', include_top=False)
    else:
        src_model = load_model(weights)
        if application == 'ResNet50':
            conv_base = src_model.get_layer('resnet50')
        elif application == 'MobileNet':
            conv_base = src_model.get_layer('mobilenet_1.00_224')

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

def resnet50_localization_regression(input_shape=None, conv_base_source=None, params=None, num_top_conv2d=1, ModelType=None):
    ''' Build a ResNet50 based convnet that output classification and bounding box related info 
        The custom layer EvaluateOutputs is needed as final layer to get the observed prediction.

    Parameters
    ----------
    input_shape : Input shape of image e.g. (224, 224, 3) 
    conv_base_source : source model file (h5) path from which the 'resnet50' layer will be used
    params : others
    ModelType: the Model type to return (can be a subclass of keras.models.Model

    Returns
    -------
    Keras model

    '''
    if conv_base_source is None:
        conv_base = ResNet50(weights='imagenet', include_top=False, input_shape = input_shape) 
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

    for k in range(num_top_conv2d-1):
        X = Dropout(dropout, name="dropout_before_t_conv2d_{}".format(k))(X)
        X = Conv2D(256, (1, 1), activation='relu', name="t_conv2d_{}".format(k))(X)

    # model with linear activation output

    #X = Dropout(dropout)(X)
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
        conv_dims = K.shape(inputs)[1:3]
        conv_height_index = K.arange(0, stop=conv_dims[0])
        conv_width_index = K.arange(0, stop=conv_dims[1])
        conv_height_index = K.tile(conv_height_index, [conv_dims[1]])
        conv_width_index = K.tile(K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
        
        conv_width_index = K.flatten(K.transpose(conv_width_index))
        conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
        conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 2])
        conv_index = K.cast(conv_index, K.dtype(inputs))
        conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 2]), K.dtype(inputs))

        p_o = K.sigmoid(inputs[..., 0:1])
        p_c = K.softmax(inputs[..., 4:])
        
        b_xy = K.sigmoid(inputs[..., 1:3])
        b_r = K.exp(inputs[..., 3:4])

	# adjust prediction to each spatial grid point
        b_xy = (b_xy + conv_index) / conv_dims
        b_r = b_r / conv_dims[..., 0:1]     # conv_dims has height and width, we only need one for a square box assumption
        
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

def _install_head_resnet50_localization_regression(self, ModelType=None):
    '''
    Add a layer to evaluate interpretable predictions (class probabilities, location, size) 

    Parameters
    ----------
    input_shape : Input shape of image e.g. (224, 224, 3) 
    conv_base_source : source model file (h5) path from which the 'resnet50' layer will be used
    params : others
    ModelType: the Model type to return (can be a subclass of keras.models.Model

    Returns
    -------
    Keras model

    '''
    return install_head_resnet50_localization_regression(self, ModelType=ModelType)

Model.with_head = _install_head_resnet50_localization_regression

def install_final_activation_layer(model, activation_layer, ModelType=None):
    out = activation_layer()(model.output)        #TODO: Why can't name be added here??
    if ModelType is None:
        return Model(inputs = model.input, outputs = out)
    else:
        return ModelType(inputs = model.input, outputs = out) 

def _install_final_activation_layer(self, activation_layer, ModelType=None):
    ''' Add a final activation layer to the model and return it '''
    return install_final_activation_layer(self, activation_layer, ModelType=ModelType)


Model.with_final_activation_layer = _install_final_activation_layer

def _avg_pool_stride_one(self):
  input_layer = self.get_layer('resnet50').layers[0]
  layer_right_b4_avg_pool = self.get_layer('resnet50').layers[-2]

  # construct new avg_pool that has a strides of (1, 1)
  avg_pool_layer = self.get_layer('resnet50').get_layer('avg_pool')
  avg_pool_layer_config = avg_pool_layer.get_config()
  avg_pool_layer_config['strides'] = (1, 1)
  new_avg_pool_layer = avg_pool_layer.from_config(avg_pool_layer_config)

  #X = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), name='avg_pool')(layer_right_b4_avg_pool.output)

  X = new_avg_pool_layer(layer_right_b4_avg_pool.output)
  
  # get dropout config
  dropout = self.get_layer("dropout_before_t_conv2d_{}".format(0)).get_config()['rate']

  X = Dropout(dropout, name="dropout_before_t_conv2d_{}".format(0))(X)

  X = Conv2D(256, (1, 1), activation='relu', name="t_conv2d_{}".format(0))(X)

  out = Conv2D(10, (1, 1), name='t_output')(X)

  avg_pool_stride_one_model = Model(inputs=input_layer.input, outputs=out)
  
  # copy the weights for last 2 conv2d layers
  name = "t_conv2d_{}".format(0)
  avg_pool_stride_one_model.get_layer(name).set_weights(self.get_layer(name).get_weights())
  avg_pool_stride_one_model.get_layer("t_output").set_weights(self.get_layer("t_output").get_weights())
  
  return avg_pool_stride_one_model


Model.with_avg_pool_stride_one = _avg_pool_stride_one


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
        img = PIL.Image.fromarray((cropped_x*255.).astype(np.uint8))
        resized_img = img.resize((height, width), PIL.Image.BICUBIC)   # resize back to what the model input expects
        cropped_resized_x = np.array(resized_img)

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

    def predict_zoom_and_focus_raw_image(self, filename, **kwargs):
        img = PIL.Image.open(filename)

	# take care of jpeg img orientation issue of PIL.Image.open
        #print(type(img) == PIL.JpegImagePlugin.JpegImageFile)
        
        if type(img) == PIL.JpegImagePlugin.JpegImageFile:
            exif = dict(img._getexif().items())
            inv_ExifTags = dict([(property, idx) for idx, property in ExifTags.TAGS.items()])	
            orientation = exif[inv_ExifTags['Orientation']]
            if orientation == 3:
                img = img.rotate(180)
            elif orientation == 6:
                img = img.rotate(270)
            elif orientation == 8:
                img = img.rotate(90)
        
        orig_img_size = np.array([img.height, img.width]).reshape((1, 2))    # note this is not 224x224, but large

        x = np.array(img.resize((224, 224), PIL.Image.BICUBIC))              # TODO: 224 shouldnt be hardcoded, need to figure this out

        x = x[:,:,:3]   # ignore the last channel in case there are 4 (224, 224, 4), which is the alpha 

        y_pred = self.predict(x[None]/255., **kwargs)

	# check if this is background, if it is, skip zoom/focus 
        if np.round(y_pred[..., 0]) == 0:
            return y_pred

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
        
        cropped_x = np.array(img)[abs_crop_coords[0,0]:abs_crop_coords[0,2], abs_crop_coords[0,1]:abs_crop_coords[0,3], :]

        cropped_img = PIL.Image.fromarray(cropped_x)

        resized_img = cropped_img.resize((224, 224), PIL.Image.BICUBIC) # resize back to what the model input expects

        cropped_resized_x = np.array(resized_img)
        cropped_resized_x = cropped_resized_x[:, :, :3]

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

def regression_model_with_input_shape(application='ResNet50', input_shape=None, dropout=0.0, avg_pool_stride=(7, 7), ModelType=None):
  ''' Return a model with a specified input_shape. If no input shape is specified, the net effect is removal of
  the last Reshape layer, resulting in a model with output shape like (batch, N, N, 10) compared with original (batch, 10)
 
  '''

  if input_shape is not None:
    width, height, channel = input_shape
  else:
    width, height, channel = None, None, 3

  if application == 'ResNet50':
    conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(height, width, 3))
  elif application == 'MobileNet':
    conv_base = MobileNet(weights='imagenet', include_top=False)

  X_input = Input((height, width, 3), name='input')
  X = conv_base(X_input)

  # Keras 2.2.4, ResNet50(include_top=False) does not have avg_pool in the last layer.
  # we will add it back in here. This is actually better flexible design. 
  X = AveragePooling2D(name='avg_pool', pool_size=(7, 7), strides=avg_pool_stride, padding='valid')(X)

  X = Dropout(dropout, name="dropout_before_t_conv2d_{}".format(0))(X)
  X = Conv2D(256, (1, 1), activation='relu', name="t_conv2d_{}".format(0))(X)

  #X = Dropout(dropout)(X)
  #out = Conv2D(10, (1, 1), kernel_regularizer=keras.regularizers.l2(0.0005), name='t_output')(X)
  out = Conv2D(10, (1, 1), name='t_output')(X)

  if ModelType is None:
    return Model(inputs = X_input, outputs = out) 
  else:
    return ModelType(inputs = X_input, outputs = out)



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

def error_analysis_summary_print(y_true, y_pred, filenames=None, iou_score_threshold=0.6):
    joint_accuracy, iou_accuracy, iou_scores, classes_accuracy, y_true_conf, y_pred_conf = L_acc_by_parts(y_true, y_pred, iou_score_threshold=iou_score_threshold)
    
    num_samples = len(joint_accuracy)

    print("General Error Summary:\n")
    print("----------------------\n")
    print("{} errors and accuracy is {:.2f}%".format(np.sum(1. - joint_accuracy), np.sum(joint_accuracy)/num_samples*100.))
    print("")

    total_num_err = np.sum(1. - joint_accuracy)

    print("% of mistake due to IOU: {0:.2f}%".format(
        np.sum((1. - iou_accuracy) * np.clip(y_pred_conf + y_true_conf, 0, 1)) / total_num_err * 100.   # % of IOU mismatch that matters, everything but background
    ))

    print("% of mistake due to mis-classification: {0:.2f}%".format(
        np.sum((1. - classes_accuracy) * np.clip(y_pred_conf + y_true_conf, 0, 1)) / total_num_err * 100.
    ))

    print("% of mistake due to both: {0:.2f}%".format(
        np.sum(
        (1. - iou_accuracy) * 
        (1. - classes_accuracy) *
        np.clip(y_pred_conf + y_true_conf, 0, 1)
        ) / total_num_err * 100.
    ))

    print("")

    print("Overall error % due to IOU: {:.2f}%".format(
        np.sum((1. - iou_accuracy) * np.clip(y_pred_conf + y_true_conf, 0, 1)) / num_samples * 100.
    )) 

    print("Overall error % due to mis-classification: {:.2f}%".format(
        np.sum((1. - classes_accuracy) * np.clip(y_pred_conf + y_true_conf, 0, 1)) / num_samples * 100.
    ))


    print("Overall error % due to both: {:.2f}%\n".format(  
        np.sum(
        (1. - iou_accuracy) * 
        (1. - classes_accuracy) *
        np.clip(y_pred_conf + y_true_conf, 0, 1)
        ) / num_samples * 100.
    ))

    if filenames is not None:
        num_err_due_far = len([str(filenames[idx]) for idx in np.nonzero(1. - joint_accuracy)[0] 
                          if "FAR" in str(filenames[idx])])

        print("% due to far object: {:.2f}%\n".format(
            num_err_due_far/total_num_err*100.
        ))


