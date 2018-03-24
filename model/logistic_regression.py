from keras.models import Sequential
from keras.layers import Dense

def build_model(input_shape=(224*224*3,), nb_classes=6, params=None):
   ''' return keras model for logistic regression (both binary or multiclass classification)

   Parameters
   ----------
   input_shape : a tuple e.g. (16,)
   nb_classes : # of classes


   Returns
   -------
   A Keras model
   
   '''

   model = Sequential()
   model.add(Dense(nb_classes, activation='softmax', input_shape=input_shape))

   return model
