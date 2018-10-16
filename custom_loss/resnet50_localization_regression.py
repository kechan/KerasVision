from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

def obj_localization_loss(y_true, y_pred):
    y_true_confidence = y_true[:, 0:1]
    y_pred_confidence = K.sigmoid(y_pred[:, 0:1])
    
    y_true_xy = y_true[:, 1:3]
    y_pred_xy = K.sigmoid(y_pred[:, 1:3])
    
    y_true_r = K.log(y_true[:, 3:4] + K.epsilon())     # 1e-7 to ensure 
    y_pred_r = y_pred[:, 3:4]
    
    y_true_classes = y_true[:, 4:]
    y_pred_classes = K.softmax(y_pred[:, 4:])
    
    confidence_loss = K.mean(K.square(y_true_confidence - y_pred_confidence), axis=-1)
    
    coord_loss = K.mean(K.square(y_pred_xy - y_true_xy), axis=-1) + 8 * K.mean(K.square(y_true_r - y_pred_r), axis=-1)
    
    classes_loss = K.mean(K.square(y_true_classes - y_pred_classes), axis=-1)
    
    y_true_confidence_squeeze = K.squeeze(y_true_confidence, axis=-1)
    
    total_loss = confidence_loss + y_true_confidence_squeeze * (coord_loss + classes_loss)        
    
    return total_loss
  
#get_custom_objects().update({'obj_localization_loss': obj_localization_loss})

def obj_localization_loss_2(y_true, y_pred):
    y_true_confidence = y_true[:, 0:1] 
    y_pred_confidence = K.sigmoid(y_pred[:, 0:1])
    
    y_true_xy = K.log(y_true[:, 1:3] / (1. - y_true[:, 1:3]) + K.epsilon())             # inverse of sigmoid
    y_pred_xy = y_pred[:, 1:3]
    
    y_true_r = K.log(y_true[:, 3:4] + K.epsilon())     # 1e-7 to ensure 
    y_pred_r = y_pred[:, 3:4]
    
    y_true_classes = y_true[:, 4:]
    y_pred_classes = K.softmax(y_pred[:, 4:])
    
    confidence_loss = K.mean(K.square(y_true_confidence - y_pred_confidence), axis=-1)
    
    coord_loss = K.mean(K.square(y_pred_xy - y_true_xy), axis=-1) + 8 * K.mean(K.square(y_true_r - y_pred_r), axis=-1)
    
    classes_loss = K.mean(K.square(y_true_classes - y_pred_classes), axis=-1)
    
    y_true_confidence_squeeze = K.squeeze(y_true_confidence, axis=-1)
    
    total_loss = confidence_loss + y_true_confidence_squeeze * (coord_loss + classes_loss)        
    
    return total_loss

