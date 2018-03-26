from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

def build_model(input_shape=(224, 224, 3), nb_classes=6, params=None):
    ''' Convnet as introduced in chapter 5 of Chollet's book
    '''
    ''' CP-CP-CP-CP-FC-FC
    '''

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())

    if params.dropout is not None:
        model.add(Dropout(params.dropout))

    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))

    return model
