from keras.utils import to_categorical

def one_hot_encode_y(train_set_y, dev_set_y, test_set_y):
    # one-hot encoding of Y
    train_set_y = to_categorical(train_set_y)
    dev_set_y = to_categorical(dev_set_y)
    test_set_y = to_categorical(test_set_y)

    return train_set_y, dev_set_y, test_set_y

def renorm(train_set_x, dev_set_x, test_set_x):
    train_set_x = train_set_x.astype('float32') / 255.
    dev_set_x = dev_set_x.astype('float32') / 255.
    test_set_x = test_set_x.astype('float32') / 255.

    return train_set_x, dev_set_x, test_set_x
