import os
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

#get_custom_objects().update({'exp_activation': Exp(exp_activation)})

def build_model(input_shape=None, conv_base_source=None, params=None):
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
        src_model = load_model(os.path.join(top_model_dir, 'keras_resnet50_far_less_aug_conv1_d0.5_weights_acc_0.9204.h5'))
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

    return Model(inputs=X_input, outputs=out)

