from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


def obj_localization_loss(y_true, y_pred):
    class_scale = 1.

    y_true_confidence = y_true[:, 0:1]
    y_pred_confidence = y_pred[:, 0:1]
    
    #y_true_coord = y_true[:, 1:4]
    #y_pred_coord = y_pred[:, 1:4]
    
    y_true_xy = y_true[:, 1:3]
    y_pred_xy = y_pred[:, 1:3]
    
    y_true_r = y_true[:, 3]
    y_pred_r = y_pred[:, 3]
    
    
    y_true_classes = y_true[:, 4:]
    y_pred_classes = y_pred[:, 4:]
    
    confidence_loss = K.mean(K.binary_crossentropy(y_true_confidence, y_pred_confidence), axis=-1)
    
    #coord_loss = K.mean(K.square(y_pred_coord - y_true_coord), axis=-1)
    
    coord_loss = K.mean(K.square(y_pred_xy - y_true_xy), axis=-1) + 8 * K.mean(K.square(y_true_r - y_pred_r), axis=-1)
    
    classes_loss = class_scale * K.categorical_crossentropy(y_true_classes, y_pred_classes)
    
    y_true_confidence_squeeze = K.squeeze(y_true_confidence, axis=-1)
    
    total_loss = confidence_loss + y_true_confidence_squeeze * (coord_loss + classes_loss)        
    
    '''
    total_loss = tf.Print(total_loss, [K.shape(confidence_loss), 
                                       K.shape(coord_loss),
                                       K.shape(classes_loss),
                                       K.shape(classes_loss)], message='loss info = ')
    '''
    
    return total_loss

get_custom_objects().update({'obj_localization_loss': obj_localization_loss})

