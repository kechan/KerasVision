def train_and_evaluate_with_fit(model, train_set_x, train_set_y, dev_set_x, dev_set_y, params, callbacks_list):

    print("model.fit got here")

    return model.fit(train_set_x, train_set_y, epochs=params.num_epochs, batch_size=params.batch_size, validation_data=(dev_set_x, dev_set_y), callbacks=callbacks_list)


def train_and_evaluate_with_fit_generator(model, params, callbacks_list):

    train_generator = params.train_generator
    validation_generator = params.validation_generator
    num_epochs = params.num_epochs

    batch_size = params.batch_size
    train_sample_size = params.train_sample_size
    validation_sample_size = params.validation_sample_size

    return model.fit_generator(train_generator,
                               steps_per_epoch = train_sample_size // batch_size,
			       epochs=num_epochs,
			       callbacks=callbacks_list,
			       validation_data=validation_generator,
			       validation_steps = validation_sample_size // batch_size)

       
