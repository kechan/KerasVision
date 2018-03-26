from keras import optimizers

def compile_model(model, params):
    if params.optimizer is None or params.optimizer == 'rmsprop':
        if params.learning_rate is not None:
            model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=params.learning_rate), metrics=['accuracy'])
	else:
            model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    elif params.optimizer == 'adam':
        if params.learning_rate is not None:
            model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=params.learning_rate), metrics=['accuracy'])
	else:
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

