""" For configuring Keras data generator 
"""

import os
from keras.preprocessing.image import ImageDataGenerator
from augmentation.CustomImageDataGenerator import * 


def configure_generator_for_dir(data_dir, params):
    '''
    for .flow_from_directory(...)
    '''

    print("Using ImageDataGenerator and .flow_from_directory")

    assert hasattr(params, "image_size"), "image_size must be defined in params.json"

    size = params.image_size
    #classes = params.classes
    batch_size = params.batch_size

    train_datagen_args = dict(rescale=1./255)
    assign_more_params(train_datagen_args, params)

    train_datagen = CustomImageDataGenerator(**train_datagen_args)    # image rescale
    test_datagen = CustomImageDataGenerator(rescale=1./255)

    # checking up on data directories
    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        train_dir = os.path.join(data_dir, "train")
        assert os.path.exists(train_dir) and os.path.isdir(train_dir), "train dir not found"

        validation_dir = os.path.join(data_dir, "validation")
	assert os.path.exists(validation_dir) and os.path.isdir(validation_dir), "validation dir not found"

	test_dir = os.path.join(data_dir, "test")
	assert os.path.exists(test_dir) and os.path.isdir(test_dir), "test dir not found"

    class_mode = 'categorical'
    if hasattr(params, 'optimizer'):
        if params.optimizer.startswith('binary'):
	    class_mode = 'binary'
	elif params.optimizer.startswith('categorical'):
	    class_mode = 'categorical'

    if params.model_type.endswith('.noshape'):
        train_generator = train_datagen.flow_from_directory_with_1x1_conv_target(train_dir, 
                                                        target_size=(size, size), 
							class_mode=class_mode,
                                                        batch_size=batch_size)


        validation_generator = test_datagen.flow_from_directory_with_1x1_conv_target(validation_dir,
                                                            target_size=(size, size),
							    class_mode=class_mode,
							    batch_size=batch_size)
    else:
        train_generator = train_datagen.flow_from_directory(train_dir, 
                                                        target_size=(size, size), 
							class_mode=class_mode,
                                                        batch_size=batch_size)


        validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                            target_size=(size, size),
							    class_mode=class_mode,
							    batch_size=batch_size)
 
    already_normalized = True

    return train_generator, validation_generator

    

def configure_generator(train_set_x, train_set_y, dev_set_x, dev_set_y, params):

    '''
    for .flow(...)
    '''

    print("Using ImageDataGenerator and .flow")

    batch_size = params.batch_size

    train_datagen_args = dict(rescale=1./255)
    assign_more_params(train_datagen_args, params)

    train_datagen = CustomImageDataGenerator(**train_datagen_args)    # image rescale
    test_datagen = ImageDataGenerator(rescale=1./255)

    if params.model_type.endswith('.noshape'):
        reshaped_train_set_y = train_set_y[:,None,None,:]
	reshaped_dev_set_y = dev_set_y[:,None,None,:]
	train_generator = train_datagen.flow(train_set_x, reshaped_train_set_y, batch_size=batch_size)
        validation_generator = test_datagen.flow(dev_set_x, reshaped_dev_set_y, batch_size=batch_size)
    else:
        train_generator = train_datagen.flow(train_set_x, train_set_y, batch_size=batch_size)
        validation_generator = test_datagen.flow(dev_set_x, dev_set_y, batch_size=batch_size)

    already_normalized = True

    return train_generator, validation_generator


def assign_more_params(train_datagen_args, params):

    aug_types = ["rotation_range", "width_shift_range", "height_shift_range", "shear_range", "zoom_range", "rot90", "color_shift", "contrast_stretching", "gaussian_blur_range"]

    for aug_type in aug_types:
        if hasattr(params, aug_type):
            value = params.dict[aug_type]
            train_datagen_args[aug_type] = value 
