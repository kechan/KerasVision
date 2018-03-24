from keras import optimizers

def compile_model(model, params):
    if params.optimizer is None or params.optimizer == 'rmsprop':
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=params.learning_rate), metrics=['accuracy'])
    elif params.optimizer == 'adam':
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=params.learning_rate), metrics=['accuracy'])

