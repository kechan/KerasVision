from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from keras.layers import BatchNormalization

def add_conv_bn_conv_bn_pool_block(model, params):

    if hasattr(params, "batch_norm") and params.batch_norm:
        batch_norm = True
    else:
        batch_norm = False

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, name='conv0'))
    if batch_norm:
        model.add(BatchNormalization(axis=3, name='bn0'))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), padding='same', name='conv1'))
    if batch_norm:
        model.add(BatchNormalization(axis=3, name='bn1'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2)))


def build_model(input_shape=(224, 224, 3), nb_classes=6, nb_blocks=3, params=None):
    ''' Inspired by Happy House from Course 4 Deep Learning Specialization Coursera
    '''
    ''' CP-CP-CP-CP-(DO)-FC-FC
    '''
    ''' with batch norm
    '''
    ''' CONV-BATCHNORM-RELU->POOL 4 times and then DO-FC-FC
    '''
    model = Sequential()

    for i in range(nb_blocks):
        add_conv_bn_conv_bn_pool_block(model, params)
     
    model.add(Flatten())

    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu'))

    model.add(Dense(nb_classes, activation='softmax')) 

    return model
