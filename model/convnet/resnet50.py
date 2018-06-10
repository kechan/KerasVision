from keras.applications import ResNet50
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten


def build_model(input_shape=None, nb_classes=6, params=None):

    if input_shape is None:
        conv_base = ResNet50(weights='imagenet', include_top=False)

	model = Sequential()

        model.add(conv_base)
        if params.dropout is not None:
            model.add(Dropout(params.dropout))
        model.add(Conv2D(nb_classes, (1, 1), activation='softmax', name='conv_preds'))

    else:
        conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

        model = Sequential()

        model.add(conv_base)
        model.add(Flatten())
    
        if params.dropout is not None:
            model.add(Dropout(params.dropout))

        model.add(Dense(nb_classes, activation='softmax'))

    return model
