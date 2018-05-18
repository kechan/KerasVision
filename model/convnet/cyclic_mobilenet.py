from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Average
from keras.layers import BatchNormalization, Activation
from keras.layers import Input, Lambda

from keras.applications import MobileNet
import keras.backend as K


def build_model(input_shape = (64, 64, 3), params=None):
    ''' Adding some cyclic invariance to MobileNet '''

    conv_base = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)

    X_input = Input(input_shape) 

    r0 = conv_base(X_input)

    # rot90
    X1 = Lambda(lambda x: K.reverse(x, axes=2), output_shape=input_shape)(X_input) # left-right flip
    X1 = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)))(X1) # transpose
    r1 = conv_base(X1)

    # rot180
    X2 = Lambda(lambda x: K.reverse(x, axes=2), output_shape=input_shape)(X_input) # left-right flip
    X2 = Lambda(lambda x: K.reverse(x, axes=1), output_shape=input_shape)(X2) # up-down flip
    r2 = conv_base(X2)

    # rot270
    X3 = Lambda(lambda x: K.reverse(x, axes=1), output_shape=input_shape)(X_input) # up-down flip
    X3 = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)))(X3) # transpose
    r3 = conv_base(X3)

    out = Average()([r0, r1, r2, r3])

    model = Model(inputs = X_input, outputs = out)

    return model
