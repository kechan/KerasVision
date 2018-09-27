import argparse
import os, shutil, glob, h5py, random
from subprocess import check_call
import sys

from utils import Params

from keras import backend as K

from keras.utils import to_categorical
from data.load_data import from_splitted_hdf5
from data.augmentation.CustomImageDataGenerator import * 
from data.augmentation.ImageDataGeneratorObjectDetection import *
from data.data_util import preview_data_aug

from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator
from keras.preprocessing.image import transform_matrix_offset_center, apply_transform, array_to_img

import matplotlib.pyplot as plt
import colorsys

import PIL 
from PIL import Image, ImageDraw, ImageFont

PYTHON = sys.executable
parser = argparse.ArgumentParser()

def get_data(data_dir):
    
    train_file = os.path.join(data_dir, "train_224_224.hdf5")
    dataset = h5py.File(train_file, 'r')
    classes = dataset['list_classes'][:]

    # shuffle and select

    train_set_x = dataset['train_set_x'][:]
    train_set_y = dataset['train_set_y'][:]
    train_idx_filenames = dataset['train_idx_filenames'][:]

    indice_classes = {}
    for i, c in enumerate(classes):
        indice_classes[str(i)] = c

    dev_file = os.path.join(data_dir, "validation_224_224.hdf5")
    dataset = h5py.File(dev_file, 'r')

    dev_set_x = dataset['dev_set_x'][:]   #128
    dev_set_y = dataset['dev_set_y'][:]   #128
    dev_idx_filenames = dataset['dev_idx_filenames'][:]   #128

    test_file = os.path.join(data_dir, "test_224_224.hdf5")
    dataset = h5py.File(test_file, 'r')

    test_set_x = dataset['test_set_x'][:]
    test_set_y = dataset['test_set_y'][:]
    test_idx_filenames = dataset['test_idx_filenames'][:]

    return train_set_x, train_set_y, train_idx_filenames, dev_set_x, dev_set_y, dev_idx_filenames, test_set_x, test_set_y, test_idx_filenames, classes, indice_classes


def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
    
    #font = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    font = ImageFont.truetype(font='/Library/Fonts/Microsoft/Arial.ttf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    #font=ImageFont.truetype(font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    
    
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
        print(label, (left, top), (right, bottom))

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

def visualize_prediction(x, yhat, y, filename, model=None):
  
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
            label_string = indice_classes[str(label_indice)]
            out_classes = np.array([label_indice])      
            out_scores = np.array([np.max(y_[4:])])
         
        else:
            out_classes = np.array([0])
            out_scores = np.array([1. - y_[0]])
      
        c_x, c_y, c_r = y_[1:4] * 224.0  #c_x, c_y, c_r and rescale back to original size     
  
        c_d = 2.*c_r

        out_boxes = np.array([c_y - c_d/2., c_x - c_d/2., c_y + c_d/2., c_x + c_d/2.]).reshape((1,4))

        return out_scores, out_boxes, out_classes
  
   # start visualization
  
    plt.figure(figsize=(6, 6))
    plt.title(filename)
  
    image = PIL.Image.fromarray(x) 
  
    pred_score, pred_box, pred_class = get_drawing_info(yhat)
    true_score, true_box, true_class = get_drawing_info(y)
  
    out_scores = np.concatenate([pred_score, true_score], axis=0)
    out_boxes = np.concatenate([pred_box, true_box], axis=0)
    out_classes = np.concatenate([pred_class, true_class], axis=0)
  
    iou_score = iou(pred_box[..., :2], pred_box[..., 2:], true_box[..., :2], true_box[..., 2:])
    print("iou = {}".format(iou_score))
  
    draw_boxes(image, out_scores, out_boxes, out_classes, classes, colors)

    plt.imshow(image)
    plt.grid(None)
  
    if model is not None:
        score = model.evaluate((x[None]/255.).astype(np.float32), y[None])
        print("score: {}".format(score))
 
def test_data_gen_for_obj_detection(train_set_x, train_set_y, dev_set_x, dev_set_y, save_to_dir=None):

    data_gen = ImageDataGeneratorObjectDetection(rescale=1./255, 
                                                 height_shift_range=0.3, 
						 width_shift_range=0.3,
						 rot90=True)

    num_previews = 5
    num_samples = 5

    index = np.random.randint(len(train_set_x))
    #index = 2505
    print("index: {}".format(index))

    image = train_set_x[index:index+num_samples, :]
    image_label = train_set_y[index:index+num_samples, :]

    i = 0
    for img_batch, label_batch in data_gen.flow(image, image_label, batch_size=num_samples):
    
        for img, label in zip(img_batch, label_batch):
        
            #print("label: {}".format(label))

            img = (img*255).astype(np.uint8)
            visualize_prediction(img, label, label, '')

            plt.show()
        
        i += 1  
        if i >= num_previews:
            break 


if __name__ == "__main__":
    args = parser.parse_args()

    home_dir = os.getenv("HOME")
    mount_dir = "/Volumes/My Mac Backup"

    project_dir = os.path.join(home_dir, "Documents", "CoinSee")
    #project_dir = os.path.join(mount_dir, "CoinSee")
    
    top_data_dir = os.path.join(project_dir, "data")
    data_dir = os.path.join(top_data_dir, '224x224_original_and_cropped_merged_heads_bbox')
    tmp_dir = os.path.join(top_data_dir, 'tmp')

    train_set_x, train_set_y, train_idx_filenames, dev_set_x, dev_set_y, dev_idx_filenames, test_set_x, test_set_y, test_idx_filenames, classes, indice_classes = get_data(data_dir)

    colors = generate_colors(classes)  

    test_data_gen_for_obj_detection(train_set_x, train_set_y, dev_set_x, dev_set_y) 


