from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout
from keras.applications import ResNet50

def transform_pretrained_resnet_fc_2_conv(resnet_model, dropout=None):
    ''' Transform a resnet50 with fc as last layer to using 1x1 conv'''
    ''' this will allow flexible input (ie. image size) '''

    assert resnet_model.get_layer('resnet50') is not None, "Expecting a layer named resnet50"
    
    # serialized the weight for the conv_base part of the fine tuned model
    resnet_model.get_layer('resnet50').save_weights('tmp_weights')
    
    # Instantiate a topless model with flexible input shape
    conv_base = ResNet50(weights='imagenet', include_top=False)

    # load the conv_base weights into the model with flexible input shape
    conv_base.load_weights('tmp_weights')

    # Construct a new model with conv 1x1 as last layer, instead of FC
    model = Sequential()

    model.add(conv_base)
    if dropout is not None:
        model.add(Dropout(dropout))
    model.add(Conv2D(7, (1, 1), activation='softmax', name='conv_preds'))

    # "Transfer" the weight from FC layer to 1x1 Conv2D layer
    assert resnet_model.get_layer('dense_1') is not None, "Expecting a layer named dense_1"

    dense_1_weights = resnet_model.get_layer('dense_1').get_weights()[0]
    dense_1_bias = resnet_model.get_layer('dense_1').get_weights()[1]

    # assign the weights of the 1x1 conv2d layer from the Dense layer of resnet_model
    conv_preds = model.get_layer('conv_preds')
    conv_preds.set_weights([dense_1_weights[None, None,:,:], dense_1_bias])
    
    return model

def transform_resnet50_to_sliding_window_conv(resnet_model, dropout=None):

    conv_preds = resnet_model.get_layer['conv_preds']
  
    assert conv_preds is not None, "Expecting a layer named conv_preds"

    # get the weight from the last conv1x1 layer 'conv_preds', we need this later.
    conv_preds_weights = conv_preds.get_weights() 

    input_layer = resnet_model.layers[0].layers[0]
    layer_before_avgpool = resnet_model.layers[0].layers[-2]

    model_before_avgpool = Model(inputs = input_layer.input, outputs=layer_before_avgpool.output)

    sliding_window_model = Sequential()
    
    sliding_window_model.add(model_before_avgpool)
    sliding_window_model.add(AveragePooling2D(pool_size=(7, 7), strides=(1, 1)))  # swap in (1,1) stride

    if dropout is not None:
        sliding_window_model.add(Dropout(dropout))

    sliding_window_model.add(Conv2D(7, (1, 1), activation='softmax', name='conv_preds'))

    # load back in the weights for last conv1x1 layer 'conv_preds'

    new_conv_preds = sliding_window_model.get_layer('conv_preds')

    new_conv_preds.set_weights(conv_preds_weights)

    return sliding_window_model
