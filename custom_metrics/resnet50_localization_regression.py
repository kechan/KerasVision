from keras import backend as K
from keras.metrics import categorical_accuracy
from keras.utils.generic_utils import get_custom_objects

def L_acc(y_true, y_pred):
    # Confidence component (aka objectness)
    
    y_true_confidence = y_true[:, 0]
    y_pred_confidence = K.round(K.sigmoid(y_pred[:, 0]))
        
    # Class component 
    
    y_true_classes = y_true[:, 4:]
    y_pred_classes = K.softmax(y_pred[:, 4:])
        
    classes_accuracy = categorical_accuracy(y_true_classes, y_pred_classes)
    
    true_xy = y_true[:, 1:3] 
    pred_xy = K.sigmoid(y_pred[:, 1:3])
    
    true_r = y_true[:, 3:4]
    pred_r = K.exp(y_pred[:, 3:4])

    # compute IOU, using the top,left,bottom,right representation.
    true_mins = true_xy - true_r
    true_maxes = true_xy + true_r
    
    pred_mins = pred_xy - pred_r
    pred_maxes = pred_xy + pred_r
    
    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    pred_areas = 4. * pred_r[..., 0] * pred_r[..., 0]   # a square
    true_areas = 4. * true_r[..., 0] * true_r[..., 0]
    
    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas
    
    iou_accuracy = K.cast(iou_scores > 0.6, K.floatx())
    
    
    joint_accuracy = (y_true_confidence * y_pred_confidence * classes_accuracy * iou_accuracy) + \
                     ((1.0 - y_true_confidence) * (1.0 - y_pred_confidence))     # if background, dont care about class or iou
    
    
    return joint_accuracy

