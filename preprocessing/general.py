from keras.utils import to_categorical

def one_hot_encode_y(train_set_y, dev_set_y, test_set_y):
    # one-hot encoding of Y
    train_set_y = to_categorical(train_set_y)
    dev_set_y = to_categorical(dev_set_y)
    test_set_y = to_categorical(test_set_y)

    return train_set_y, dev_set_y, test_set_y
