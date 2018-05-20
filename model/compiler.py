from keras import optimizers

def compile_model(model, params):

    if not hasattr(params, 'decay'):
        params.decay = 0.0

    if params.optimizer is None or params.optimizer == 'rmsprop':
        if params.learning_rate is not None:
            model.compile(loss=params.loss, optimizer=optimizers.RMSprop(lr=params.learning_rate, decay=params.decay), metrics=['accuracy'])
	else:
            model.compile(loss=params.loss, optimizer='rmsprop', metrics=['accuracy'])

    elif params.optimizer == 'adam':
        if params.learning_rate is not None:
	    print("model.compile got here")
            model.compile(loss=params.loss, optimizer=optimizers.Adam(lr=params.learning_rate, decay=params.decay), metrics=['accuracy'])
	else:
            model.compile(loss=params.loss, optimizer='adam', metrics=['accuracy'])

