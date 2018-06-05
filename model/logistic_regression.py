from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape

def build_model(input_shape=(224, 224, 3), nb_classes=6, params=None):
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
   model.add(Reshape(target_shape=(-1,), input_shape=input_shape))
   #model.add(Dense(nb_classes, activation='softmax', input_shape=input_shape))

   if params.dropout is not None:
       model.add(Dropout(params.dropout))

   model.add(Dense(nb_classes, activation='softmax'))

   return model
