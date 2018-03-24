from keras.models import Sequential
from keras.layers import Dense

def build_model(input_shape=(224*224*3,), nb_classes=6, params=None):
    ''' return Feedward model for either binary or multiclass classification

   Parameters
   ----------
   input_shape : a tuple e.g. (16,)
   nb_classes : # of classes


   Returns
   -------
   A Keras model
   
   '''

    model = Sequential()

    hidden_layers_config = params.hidden_layers_config

    model.add(Dense(hidden_layers_config[0], activation='relu', input_shape=input_shape))

    for n in hidden_layers_config[1:]:
        model.add(Dense(n, activation='relu'))

    model.add(Dense(nb_classes, activation='softmax'))

    return model
