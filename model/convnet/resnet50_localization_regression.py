import os
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Conv2D, Dropout, Concatenate, Reshape
from keras.layers import BatchNormalization, Activation
from keras.layers import Input, Lambda

from keras.applications import ResNet50
import keras.backend as K

from keras.utils.generic_utils import get_custom_objects

def resnet50_localization_regression(input_shape=None, conv_base_source=None, params=None):
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

    return Model(inputs=X_input, outputs=out)
