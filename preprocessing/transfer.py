from keras.utils import to_categorical
import numpy as np

def preprocess(train_set_x, train_set_y, dev_set_x, dev_set_y, test_set_x, test_set_y, params):
    ''' return flatten input X for train, dev, and test set.


    Parameters
    ----------
    train_set_x : train set X inputs
    train_set_y : train set Y targets (scalar indices for the label)
    dev_set_x : dev/validation set X inputs
    dev_set_y : dev/validation set Y targets
    test_set_x : test set X inputs
    test_set_y : test set Y targets

    Returns
    -------
    train_set_x : Flatten
    train_set_y : 
    dev_set_x : Flatten
    dev_set_y : 
    test_set_x : Flatten
    test_set_y : 

    '''

    dim = np.prod(train_set_x.shape[1:])

    # reshape/flatten
    train_set_x = train_set_x.reshape((-1, dim))
    dev_set_x = dev_set_x.reshape((-1, dim))
    test_set_x = test_set_x.reshape((-1, dim))

    return train_set_x, train_set_y, dev_set_x, dev_set_y, test_set_x, test_set_y

