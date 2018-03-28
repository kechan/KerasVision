from keras.utils import to_categorical

def preprocess(train_set_x, train_set_y, dev_set_x, dev_set_y, test_set_x, test_set_y):
    ''' return flatten and normalize input X for train, dev, and test set.

    Assumptions:
    
    X is a square image data with Channel Last ordering
    All images are equal height and width.


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
    train_set_x : Flatten and normalize
    train_set_y : one-hot encoded
    dev_set_x : Flatten and normalize
    dev_set_y : one-hot encoded
    test_set_x : Flatten and normalize
    test_set_y : one-hot encoded

    '''

    height = train_set_x.shape[1]
    width = train_set_x.shape[2]

    assert height == width

    dim = height*width*3

    # reshape/flatten
    train_set_x = train_set_x.reshape((-1, dim))
    dev_set_x = dev_set_x.reshape((-1, dim))
    test_set_x = test_set_x.reshape((-1, dim))

    # normalize 
    train_set_x = train_set_x.astype('float32') / 255.
    dev_set_x = dev_set_x.astype('float32') / 255.
    test_set_x = test_set_x.astype('float32') / 255.

    return train_set_x, train_set_y, dev_set_x, dev_set_y, test_set_x, test_set_y

