import colorsys, PIL, random

import numpy as np

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


def generate_colors(class_names):
    hsv_tuples = [(x / float(len(class_names)), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
    
    #font = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    #font = ImageFont.truetype(font='/Library/Fonts/Microsoft/Arial.ttf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    #font=ImageFont.truetype(font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    font=ImageFont.truetype(font='/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    
    
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        
        print("{} ({}, {}), ({}, {})".format(label, left, top, right, bottom))
        
        if predicted_class != "UNK":    # don't draw background prediction
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw


def visualize_prediction(x, yhat=None, y=None, classes=None, filename='', model=None, figsize=8, colors=None, return_image=False):
  ''' Handy function to visualize an image and the localization information from prediction and ground truth
  
  Parameters:
  -----------
  x : Single image with shape (height, width, channel)
  yhat: Single predction with shape (10,) or Multi prediction with shape (n, n, 10), the expected prediction from resnet50_localization_*** model
  y: ground truth, must be same shape as yhat
  classes: list of classes, in the order of the corresponding label index 
  filename: optional string to display as title of plot 
  model: optional model to perform model.evaluate(...)
  figsize: figure size
  colors: return from calling generate_colors(classes)
  return_image: a boolean indicating if the image with bounding boxes should be returned
  '''

  assert classes is not None, "classes should be provided."
  assert colors is not None, "colors should be provided."
    
  def is_background(y_):
    if y_[0] >= 0.5:
      return False
    else:
      return True
    
  def iou(mins1, maxes1, mins2, maxes2):
  
    intersect_mins = np.maximum(mins1, mins2)
    intersect_maxes = np.minimum(maxes1, maxes2)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
  
    wh1 = maxes1 - mins1
    wh2 = maxes2 - mins2
      
    areas1 = wh1[..., 0] * wh1[..., 1]
    areas2 = wh2[..., 0] * wh2[..., 1]
    
    union_areas = areas1 + areas2 - intersect_areas
    
    iou_scores = intersect_areas / union_areas
    
    return iou_scores
  
  
  def get_drawing_info(y_):
    if not is_background(y_):
      label_indice = np.argmax(y_[4:]) + 1
      #label_string = indice_classes[str(label_indice)]
      out_classes = np.array([label_indice])      
      #out_scores = np.array([np.max(y_[4:])])
      #title = "{} \n {}".format(filename, label_string)
    else:
      out_classes = np.array([0])
      #out_scores = np.array([1. - y_[0]])
      #title = "{} \n {}".format(filename, 'This is a background')      
    
    confidence_scores = y_[0:1] * y_[4:]
    out_scores = np.max(confidence_scores, axis=-1, keepdims=True)
    
    c_x, c_y, c_r = y_[1:4] * x.shape[0]  #c_x, c_y, c_r and rescale back to original size     
    
    c_d = 2.*c_r

    #out_scores = np.array([y_[0]])
    out_boxes = np.array([c_y - c_d/2., c_x - c_d/2., c_y + c_d/2., c_x + c_d/2.]).reshape((1,4))

    return out_scores, out_boxes, out_classes
  
  # start visualization
  
  plt.figure(figsize=(figsize, figsize))
  
  title_string = str(filename)
    
  image = PIL.Image.fromarray(x) 
  
  # need to differentiate reshaped 1-dim output vs. n by n conv2d outputs
  
  if yhat is not None and y is None:
    if yhat.ndim == 1:
      pred_score, pred_box, pred_class = get_drawing_info(yhat)
      iou_score = [1.0]
      draw_boxes(image, pred_score, pred_box, pred_class, classes, colors)
    
    elif yhat.ndim == 2:
      n, _ = yhat.shape
      for i in range(n):
        yhat_ = yhat[i]

	pred_score, pred_box, pred_class = get_drawing_info(yhat_)
        iou_score = [1.0]
        draw_boxes(image, pred_score, pred_box, pred_class, classes, colors)
      
    elif yhat.ndim == 3:
      nrow, ncol, _ = yhat.shape 
      for i in range(nrow):
        for j in range(ncol):
        
          yhat_ = yhat[i][j]        
        
          pred_score, pred_box, pred_class = get_drawing_info(yhat_)
          iou_score = [1.0]
          draw_boxes(image, pred_score, pred_box, pred_class, classes, colors)
      
  elif yhat is not None and y is not None:
  
    if yhat.ndim == 1 and y.ndim == 1:
      pred_score, pred_box, pred_class = get_drawing_info(yhat)
      true_score, true_box, true_class = get_drawing_info(y)
  
      out_scores = np.concatenate([pred_score, true_score], axis=0)
      out_boxes = np.concatenate([pred_box, true_box], axis=0)
      out_classes = np.concatenate([pred_class, true_class], axis=0)
  
      iou_score = iou(pred_box[..., :2], pred_box[..., 2:], true_box[..., :2], true_box[..., 2:])
  
      draw_boxes(image, out_scores, out_boxes, out_classes, classes, colors)
    
    elif yhat.ndim == 3 and y.ndim == 3:
      nrow, ncol, _ = yhat.shape 
      for i in range(nrow):
        for j in range(ncol):
         
          yhat_ = yhat[i][j]
          y_ = y[i][j]
        
          pred_score, pred_box, pred_class = get_drawing_info(yhat_)
          true_score, true_box, true_class = get_drawing_info(y_)
  
          out_scores = np.concatenate([pred_score, true_score], axis=0)
          out_boxes = np.concatenate([pred_box, true_box], axis=0)
          out_classes = np.concatenate([pred_class, true_class], axis=0)
  
          iou_score = iou(pred_box[..., :2], pred_box[..., 2:], true_box[..., :2], true_box[..., 2:])
  
          draw_boxes(image, out_scores, out_boxes, out_classes, classes, colors)
        
    else:
      print("Unexpected dimensions for yhat and/or y. Please ensure they have same ndim of either 1 or 3")
      return

  if model is not None:
    score = model.evaluate((x[None]/255.).astype(np.float32), y[None])
    #print("score: {}".format(score))
    title_string += "\n score: {}".format(score)
    if score[1] == 1:
      plt.ylabel("CORRECT Prediction")
    else:
      plt.ylabel("WRONG Prediction")
      
  plt.title(title_string)
  plt.xlabel("iou = {}".format(iou_score[0]))
  plt.imshow(image)
  plt.grid(None)

  if return_image:
    return image
