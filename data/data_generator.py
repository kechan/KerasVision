""" For configuring Keras data generator 
"""

import os
from keras.preprocessing.image import ImageDataGenerator

def configure_generator(data_dir, params):

    size = params.image_size
    classes = params.classes
    batch_size = params.batch_size
    
    train_datagen = ImageDataGenerator(rescale=1./255)    # image rescale
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
