import numpy as np
from numpy.random import seed


import argparse
import logging
import os, shutil, glob
import random
import pickle

import tensorflow as tf
import keras
import numpy as np

from utils import Params
import params_extension

from utils import set_logger
from utils import save_dict_to_json

from data.load_data import from_splitted_hdf5
from data.data_generator import configure_generator

from model.compiler import compile_model
from model.training import train_and_evaluate_with_fit
from model.training import train_and_evaluate_with_fit_generator

from preprocessing.general import one_hot_encode_y

description = """Kick off training by reading from model_dir (with params.json) and data_dir\n
E.g. \n
python train.py --model_dir experiments/logistic_regression --data_dir data/64x64_cropped_hdf5\n
"""


parser = argparse.ArgumentParser(description)

parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")

parser.add_argument('--data_dir', default='data/64x64_SIGNS',
                    help="Directory containing the dataset")

parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")

already_normalized = False

def create_indice_to_classes_dictionary(classes):
    '''
    Input example: 
    		['cat', 'non-cat']

    Output example:
    		['0': 'cat', '1': 'non-cat']

    '''

    assert type(classes) == list or type(classes) == np.ndarray, "classes is not a list or np.ndarray"
    #print(type(classes[0]))
    assert len(classes) > 1 and (type(classes[0]) == str or type(classes[0]) == unicode or type(classes[0]) == np.string_), "number of classes must be bigger than 1 and must be a string"

    indice_classes = {}
    for i, c in enumerate(classes):
        indice_classes[str(i)] = c

    return indice_classes

def plot(history, model_dir, params):
    import matplotlib.pyplot as plt

    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(len(acc))

    # Plotting Accuracy vs Epoch curve
    plt.figure(figsize=(8, 8))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend(loc='upper left')
    plt.grid()

    plt.savefig(os.path.join(model_dir, "acc.png"))

    # Plotting Loss vs Epoch curve
    plt.figure(figsize=(8, 8))

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(model_dir, "loss.png"))

if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    # Note: this randomness may sometimes not work if training on GPU
    #seed(123)
    #tf.set_random_seed(230) 

    #Load the parameters from json file
    args = parser.parse_args()
    model_dir = args.model_dir
    restore_from = args.restore_from

    # Read params.json
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    # Validate on params
    params.validate()
    
    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    overwritting = model_dir_has_best_weights and args.restore_from is None
    assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    model_loaded_from_archive = False

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Load the data
    data_dir = args.data_dir

    logging.info("#################################################################")

    logging.info("python train.py --model_dir " + args.model_dir + " --data_dir " + args.data_dir)

    logging.info("params: " + str(params.dict))

    logging.info("Loading datasets...")
    train_set_x, train_set_y, dev_set_x, dev_set_y, test_set_x, test_set_y, classes = None, None, None, None, None, None, None

    if params.data_format == 'splitted_hdf5':
	
        train_set_x, train_set_y, dev_set_x, dev_set_y, test_set_x, test_set_y, classes = from_splitted_hdf5(data_dir = data_dir)

	logging.info("Dataset shape:")
        logging.info("\ttrain_set_x: " + str(train_set_x.shape))
        logging.info("\tdev_set_x: " + str(dev_set_x.shape))
        logging.info("\ttest_set_x: " + str(test_set_x.shape))
        logging.info("\ttrain_set_y: " + str(train_set_y.shape))
        logging.info("\tdev_set_y: " + str(dev_set_y.shape))
        logging.info("\ttest_set_y: " + str(test_set_y.shape))

	train_set_y, dev_set_y, test_set_y = one_hot_encode_y(train_set_y, dev_set_y, test_set_y)

	if hasattr(params, "use_data_gen") and params.use_data_gen:
	   logging.info("Using data generator .flow")  
           configure_generator(train_set_x, train_set_y, dev_set_x, dev_set_y, params)

        params.train_sample_size = len(train_set_y)
	params.validation_sample_size = len(dev_set_y)

    elif params.data_format == 'splitted_dirs':    # Using Keras generator

        classes = params.classes
	configure_generator(data_dir, params)

    else:
        print("Other data input format") 

    indice_classes = create_indice_to_classes_dictionary(classes)

    logging.info("\tlabel/target to class mappings: " + str(indice_classes))


    if params.model_type == "logistic_regression" or params.model_type == "feedforward":

        from preprocessing.feedforward import preprocess   # Flatten and normalize

	train_set_x, train_set_y, dev_set_x, dev_set_y, test_set_x, test_set_y = preprocess(train_set_x, train_set_y, dev_set_x, dev_set_y, test_set_x, test_set_y, params)

	logging.info("Using preprocessing.feedforward.preprocess")
	logging.info("Dataset shape after preprocessing:")
     
        logging.info("\ttrain_set_x: " + str(train_set_x.shape))
        logging.info("\tdev_set_x: " + str(dev_set_x.shape))
        logging.info("\ttest_set_x: " + str(test_set_x.shape))
        logging.info("\ttrain_set_y: " + str(train_set_y.shape))
        logging.info("\tdev_set_y: " + str(dev_set_y.shape))
        logging.info("\ttest_set_y: " + str(test_set_y.shape))

    elif params.model_type == "convnet.transfer":

        from preprocessing.transfer import preprocess     # Flatten

        train_set_x, train_set_y, dev_set_x, dev_set_y, test_set_x, test_set_y = preprocess(train_set_x, train_set_y, dev_set_x, dev_set_y, test_set_x, test_set_y, params)
        

    # Define the model
    logging.info("Creating the model...")
      
    # if restore, get it from the pre-existing .h5
    model = None
    if restore_from is not None:
        model_h5 = os.path.join(restore_from, "model_and_weights.h5") 	
	if os.path.exists(model_h5):
	    from keras.models import load_model

	    logging.info("loading model from existing .h5")
	    model = load_model(model_h5)
	    model_loaded_from_archive = True
        
    if model is None:     

        if train_set_x is None:
	    dim = params.image_size
	else:
	    dim = train_set_x.shape[1]

	if params.model_type == "logistic_regression" or params.model_type == "feedforward":

            if params.model_type == "logistic_regression":
	        from model.logistic_regression import build_model
	    elif params.model_type == "feedforward":
	        from model.feedforward import build_model

	    model = build_model(input_shape=(dim,), nb_classes=len(indice_classes), params=params)

	elif params.model_type == "convnet.chollet":

	    from model.convnet.chollet import build_model
            model = build_model(input_shape=(dim, dim, 3), nb_classes=len(indice_classes), params=params)

	elif params.model_type == "convnet.transfer":

	    from model.convnet.transfer import build_model
	    model = build_model(input_shape=train_set_x.shape[1:], nb_classes=len(indice_classes), params=params)


    logging.info("Input: " + str(model.inputs))
    model.summary(print_fn=logging.info)

    # .compile
    model_recompiled = False
    if not model_loaded_from_archive:   # don't recompile if loaded from *weights*.h5 or lose opt-zer states.
        compile_model(model, params)
	model_recompiled = True

    if not model_recompiled and hasattr(params, "always_recompile") and params.always_recompile:
        compile_model(model, params)
	model_recompiled = True

    if model_recompiled:
        logging.info("Model is re-compiled (or compiled first time).")

    # configure callbacks
    # 1) checkpoints: save the best dev/val loss weights

    # prepare weights dir to save the .h5 weights.
    num_weight_dirs = len(glob.glob(os.path.join(model_dir, "weights*")))
    weights_dir = os.path.join(model_dir, "weights_" + str(num_weight_dirs+1))
    os.mkdir(weights_dir)

    model_checkpt_callback = keras.callbacks.ModelCheckpoint(
                                                    filepath=os.path.join(weights_dir, "weights.{epoch:02d}-{val_loss:.2f}.h5"),
	                                            monitor='val_loss',
	                                            save_best_only=True
						   )

    # 2) learning rate scheduling
    model_lr_callback = keras.callbacks.ReduceLROnPlateau(
                                               monitor='val_loss',
					       factor=0.1,
					       patience=20,
					       )

    callbacks_list = [
        model_checkpt_callback,
        model_lr_callback
    ] 

    # .fit**
    if params.data_format == 'splitted_hdf5':

        if hasattr(params, "use_data_gen") and params.use_data_gen:
            history = train_and_evaluate_with_fit_generator(model, params, callbacks_list)
	else:
            history = train_and_evaluate_with_fit(model, train_set_x, train_set_y, dev_set_x, dev_set_y, params, callbacks_list)

    elif params.data_format == 'splitted_dirs':
        history = train_and_evaluate_with_fit_generator(model, params, callbacks_list)

    # save history
    history = history.history
    if restore_from is not None and model_dir == restore_from:	
        history_pickle_file = os.path.join(model_dir, "history.pickle")
	if os.path.exists(history_pickle_file):
	    with open(history_pickle_file, 'rb') as f:
                all_history = pickle.load(f)
	    for measure in history.keys():
	        if measure in all_history:
                    all_history[measure] += history[measure]
		else:
		    all_history[measure] = history[measure]

            history = all_history

    with open(os.path.join(model_dir, "history.pickle"), 'wb') as f:
        pickle.dump(history, f)

    # save model and weights
    model.save(os.path.join(model_dir, "model_and_weights.h5"), overwrite=True, include_optimizer=True)

    # put an end marker to the log
    logging.info("########################## THE END ##########################")

    plot(history, model_dir, params)
    

