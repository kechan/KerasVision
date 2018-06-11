from utils import Params

'''
This method performs a 1st line sanity check on params.json, as well as
acting like a documentation on whats available

'''


def _validate(self):

    model_type(self.model_type)

    hidden_layers_config(self)

    classes(self)

    optimizer(self.optimizer)

    learning_rate(self.learning_rate)

    dropout(self)

    train_sample_size(self)
    
    validation_sample_size(self)
    
    batch_size(self.batch_size)

    num_epochs(self.num_epochs)

    image_size(self)

Params.validate = _validate

def model_type(value):
    '''
    	logistic_regression
	feedforward
	convnet.chollet: Convnet as introduced in chapter 5 of Chollet's book
	convnet.galaxy: convnet used in galaxy kaggle competition
    '''

    assert (value == 'logistic_regression' or 
            value == 'feedforward' or 
	    value == 'convnet.chollet' or
	    value == 'convnet.galaxy' or 
	    value == 'convnet.transfer' or 
	    value == 'convnet.resnet50' or
	    value == 'convnet.resnet50.noshape'), "Invalid params.json value for key model_type."

def hidden_layers_config(params):
    '''
    Optional:
    This is only mandatory if model_type is feedforward
    '''

    if hasattr(params, "hidden_layers_config"):
        value = params.hidden_layers_config

    if params.model_type == 'feedforward':
        assert value is not None, "hidden_layers_config is missing in params.json."
        assert type(value) == list, "Invalid params.json value for key hidden_layers_config, must be a list."


def classes(params):
    '''
    Optional:
    This is only mandatory if data_format is splitted_dirs
    '''

    if hasattr(params, "classes"):
        value = params.classes


def optimizer(value):
    '''
    rmsprop
    adam

    TODO: collect more from Keras doc
    '''

    assert (value == 'rmsprop' or 
            value == 'adam'), "Invalid params.json value for key optimizer."


def learning_rate(value):
    '''
    '''
    return

def dropout(value):
    '''
    Optional
    '''
    return


def train_sample_size(params): 
    '''
    Optional: 

    This is mandatory if data_format is splitted_dirs
    '''

    if hasattr(params, "train_sample_size"):
        value = params.train_sample_size


def validation_sample_size(params):
    '''
    Optional: 

    This is mandatory if data_format is splitted_dirs
    '''

    if hasattr(params, "validation_sample_size"):
        value = params.validation_sample_size

def batch_size(value):
    assert value is not None, "batch_size is missing in params.json"

def num_epochs(value):
    assert value is not None, "num_epochs is missing in params.json"


def image_size(params):
    '''
    Optional:

    This is mandatory if data_format is splitted_dir and we are using Keras data generator
    '''

    if hasattr(params, "image_size"):
        value = params.image_size



