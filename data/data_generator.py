""" For configuring Keras data generator 
"""

import os
from keras.preprocessing.image import ImageDataGenerator

def configure_generator(data_dir, params):
    '''
    for .flow_from_directory(...)
    '''

    logging.info("Using ImageDataGenerator and .flow_from_directory")

    assert hasattr(params, "image_size"), "image_size must be defined in params.json"

    size = params.image_size
    classes = params.classes
    batch_size = params.batch_size

    train_datagen_args = dict(rescale=1./255)
    assign_more_params(train_datagen_args, params)

    train_datagen = ImageDataGenerator(**train_datagen_args)    # image rescale
    test_datagen = ImageDataGenerator(rescale=1./255)

    # checking up on data directories
    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        train_dir = os.path.join(data_dir, "train")
        assert os.path.exists(train_dir) and os.path.isdir(train_dir), "train dir not found"

        validation_dir = os.path.join(data_dir, "validation")
	assert os.path.exists(validation_dir) and os.path.isdir(validation_dir), "validation dir not found"

	test_dir = os.path.join(data_dir, "test")
	assert os.path.exists(test_dir) and os.path.isdir(test_dir), "test dir not found"

    train_generator = train_datagen.flow_from_directory(train_dir, 
                                                        target_size=(size, size), 
                                                        batch_size=batch_size, 
							classes=classes) 


    validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                            target_size=(size, size),
							    batch_size=batch_size,
							    classes=classes) 
 
    params.train_generator = train_generator
    params.validation_generator = validation_generator

    already_normalized = True

def configure_generator(train_set_x, train_set_y, dev_set_x, dev_set_y, params):

    '''
    for .flow(...)
    '''

    logging.info("Using ImageDataGenerator and .flow")

    classes = params.classes
    batch_size = params.batch_size

    train_datagen_args = dict(rescale=1./255)
    assign_more_params(train_datagen_args, params)

    train_datagen = ImageDataGenerator(**train_datagen_args)    # image rescale
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow(train_set_x, train_set_y, batch_size=batch_size)

    validation_generator = test_datagen.flow(dev_set_x, dev_set_y, batch_size=batch_size)

    params.train_generator = train_generator
    params.validation_generator = validation_generator

    already_normalized = True


def assign_more_params(train_datagen_args, params):

    aug_params = ["rotation_range", "width_shift_range", "height_shift_range", "shear_range", "zoom_range"]

    for aug_param in aug_params:
        if hasattr(params, aug_param):
            value = params.dict[aug_param]
            train_datagen_args[aug_param] = value 
