from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from keras.layers import BatchNormalization

def build_model(input_shape=(224, 224, 3), nb_classes=6, params=None):
    ''' Convnet as introduced in chapter 5 of Chollet's book
    '''
    ''' CP-CP-CP-CP-(DO)-FC-FC
    '''
    ''' with batch norm
    '''
    ''' CONV-BATCHNORM-RELU->POOL 4 times and then DO-FC-FC
    '''

    if hasattr(params, "batch_norm") and params.batch_norm:
        batch_norm = True
    else:
        batch_norm = False

    print("batch_norm = " + str(batch_norm))

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3)))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3)))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    if input_shape[0] == 224 or input_shape[0] == 112:
        model.add(Conv2D(128, (3, 3)))      # if input is 112x224 or 224x224  
    elif input_shape[0] == 64:
        model.add(Conv2D(256, (3, 3)))      # if input is 64x64
    else:
        model.add(Conv2D(128, (3, 3)))

    if batch_norm:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    if params is not None and params.dropout is not None:
        model.add(Dropout(params.dropout))

    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))

    return model
