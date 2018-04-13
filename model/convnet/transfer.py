from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers import BatchNormalization, Activation
from keras.applications import VGG16

def build_model(input_shape=(224*224*3,), nb_classes=6, params=None):
    ''' Use everything up to the block5 of VGG16, and then add 
       the custom fc on top (Transfer Learnning).

    Parameters
    ----------
    input_shape : a tuple e.g. (16,)
    nb_classes : # of classes


    Returns
    -------
    A Keras model
   
    '''

    dropout = None
    if hasattr(params, "dropout") and params.dropout is not None:
        dropout = params.dropout

    if hasattr(params, "batch_norm") and params.batch_norm:
        batch_norm = True
    else:
        batch_norm = False

    # define the custom fc layers
    assert hasattr(params, "hidden_layers_config"), "hidden_layers_config is not found." 

    hidden_layers_config = params.hidden_layers_config
    assert type(hidden_layers_config) == list, "hidden_layers_config must be a list."
    assert len(hidden_layers_config) > 0, "hidden_layers_config must not be empty."

    custom_fc = Sequential()

    custom_fc.add(Dense(hidden_layers_config[0], input_shape=input_shape))
    if batch_norm:
        custom_fc.add(BatchNormalization())
    custom_fc.add(Activation('relu'))

    if dropout is not None:
        custom_fc.add(Dropout(dropout))

    for n in hidden_layers_config[1:]:
        custom_fc.add(Dense(n))
	if batch_norm:
	    custom_fc.add(BatchNormalization())
	custom_fc.add(Activation('relu'))

	if dropout is not None:
	    custom_fc.add(Dropout(dropout))

    custom_fc.add(Dense(nb_classes, activation='softmax'))

    return custom_fc

