import os, sys, h5py, random, shutil, glob, io
from random import shuffle
import PIL

import numpy as np
import pandas as pd

import keras
from keras.utils import to_categorical

#from visualization.localization import generate_colors, draw_boxes, visualize_prediction

class MultiCoinImageGenerator:
  
  def __init__(self, coin_collection_config, template_dir, canvas_file_path, canvas_size=None, iou_threshold=0.2):
        
    coin_files = [os.path.join(template_dir, c['filename']) for c in coin_collection_config]
    
    if 'recommended_resize' in coin_collection_config[0]:
      coin_size = coin_collection_config[0]['recommended_resize']
    else:
      # figure this out from one of the coin image
      tmp_img = PIL.Image.open(coin_files[0])
      coin_size = (tmp_img.height, tmp_img.width)
    
    # numpy instance of template image set (png with alpha channel)
    coin_imageset = np.zeros((len(coin_files), coin_size[0], coin_size[1], 4), dtype=np.uint8) 
    for idx, filename in enumerate(coin_files):
      img = PIL.Image.open(filename)
      img = img.resize(coin_size, PIL.Image.BICUBIC)

      coin_imageset[idx] = np.array(img)
      
    # numpy instance of canvas
    canvas_img = PIL.Image.open(canvas_file_path)
    if canvas_size is not None:
      canvas_img = canvas_img.resize(canvas_size, PIL.Image.BICUBIC)
    
    canvas = np.array(canvas_img)
    canvas_size = np.array(canvas.shape[:2]).reshape((2, 1))
       
    # instance variable that don't change per image and should be computed only once.
    self.coin_collection_config = coin_collection_config
    self.coin_imageset = coin_imageset
    self.canvas = canvas
    self.canvas_size = canvas_size
    self.iou_threshold = iou_threshold
    
  def generate_single_image(self, num_of_coins_in_image):
    
    #num_of_coins = np.random.randint(max_num_of_coins) + 1
    
    # for tracking all rectangles to ensure overlap of no more than the threshold
    top_left_all = np.zeros((2, num_of_coins_in_image), dtype=np.float32)
    bottom_right_all = np.zeros((2, num_of_coins_in_image), dtype=np.float32)
    
    #  y (ground truth target)
    y = np.zeros(num_of_coins_in_image*9, dtype=np.float32)

    #  y (true identity without 5_10_25c_heads)
    z = np.zeros(num_of_coins_in_image, dtype=np.int)
    
    # create a local copy of canvas (otherwise, the instance canvas will be overwritten)
    canvas = self.canvas.copy()
    
    for i in range(num_of_coins_in_image):
    
      while True:
        # choose a random position on canvas (rescaled to 1.0)
        c_xy = np.random.uniform(0.1, 0.9, size=(2, 1))

        # choose a random coin 
        idx = np.random.randint(len(self.coin_collection_config))
        chosen_coin = self.coin_imageset[idx]                      #plt.imshow(chosen_coin); plt.grid(); plt.show()
  
        # for debugging #c_xy = np.array([0.0, 1.0]).reshape((2, 1))   # position is a column vector (in linear algebra term)
  
        coin_size = np.array(chosen_coin.shape[:2]).reshape((2, 1))
  
        # scale up to canvas_size
        g_xy = np.round(c_xy * self.canvas_size)
  
        # figure out the rectangle on canvas that will be replaced by the coin
        pt_at_top_left = np.maximum(g_xy - coin_size/2., 0)
        pt_at_bottom_right = np.minimum(self.canvas_size, g_xy + coin_size/2.)
    
        if not self.has_overlapped(top_left_all, bottom_right_all, pt_at_top_left, pt_at_bottom_right):
          break

      # print("{}th coin chosen...".format(i))
      # 1) figure out if the choosen coin is overlapping with an existing coin
      # 2) figure out if chosen_coin is indeed truncated and how to crop it 
  
      x1, y1 = np.squeeze(g_xy - coin_size/2.).astype(np.int)
      x2, y2 = np.squeeze(g_xy + coin_size/2.).astype(np.int)
  
      coin_col_crop_start = np.maximum(-x1, 0)
      coin_col_crop_end = coin_size[1, 0] - np.maximum(x2 - self.canvas_size[1, 0], 0)
  
      coin_row_crop_start = np.maximum(-y1, 0)
      coin_row_crop_end = coin_size[0, 0] - np.maximum(y2 - self.canvas_size[0, 0], 0)

      # randomly rotate chosen coin
      k = np.random.randint(4)
  
      alpha_coin = (np.rot90(chosen_coin, k=k)[coin_row_crop_start:coin_row_crop_end, coin_col_crop_start:coin_col_crop_end, 3] / 255.).astype(np.float32)
      alpha_canvas = 1. - alpha_coin

      alpha_canvas = alpha_canvas[..., None]
      alpha_coin = alpha_coin[..., None]
  
      top_left = np.squeeze(pt_at_top_left).astype(np.uint)
      bottom_right = np.squeeze(pt_at_bottom_right).astype(np.uint)
    
      top_left_all[..., i] = np.squeeze(pt_at_top_left)
      bottom_right_all[..., i] = np.squeeze(pt_at_bottom_right)
  
      canvas[top_left[1]: bottom_right[1], top_left[0]: bottom_right[0], :] = \
      (alpha_coin * np.rot90(chosen_coin, k=k)[coin_row_crop_start:coin_row_crop_end, coin_col_crop_start:coin_col_crop_end, :3] + 
       alpha_canvas * canvas[top_left[1]:bottom_right[1], top_left[0]: bottom_right[0], :]).astype(np.uint8)
  
      #plt.figure(figsize=(10, 10)); plt.imshow(canvas); plt.grid(); plt.show()
   
      # fill in the y (ground truth target)
      bbox_side_ratio = self.coin_collection_config[idx]['bbox_side_ratio']
      c_r = 0.5 * bbox_side_ratio * float(coin_size[0]) / self.canvas_size[0, 0]
      label_index = self.coin_collection_config[idx]['label_index']
    
      y[i*9:i*9+2] = c_xy[:, 0]
      y[i*9+2] = c_r
      y[i*9+3:i*9+9] = to_categorical(label_index, num_classes=6) 

      # this is to keep track of the identity of 5_10_25c_heads, we may need this eventually
      z[i] = self.coin_collection_config[idx]['z_index']

    return canvas, y, z
  
  def has_overlapped(self, top_left_all, bottom_right_all, top_left_proposal, bottom_right_proposal):
    ''' Detect if there is overlap among the proposed rectangle vs. all the rectangles defined by top_left_all, bottom_right_all '''
   
    # This is vector implementation of IOU 
    intersect_mins = np.maximum(top_left_proposal, top_left_all)
    intersect_maxes = np.minimum(bottom_right_proposal, bottom_right_all)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[0, ...] * intersect_wh[1, ...]

    wh_all = bottom_right_all - top_left_all
    wh_proposal = bottom_right_proposal - top_left_proposal

    areas_all = wh_all[0, ...] * wh_all[1, ...]
    areas_proposal = wh_proposal[0, ...] * wh_proposal[1, ...]
  
    union_areas = areas_all + areas_proposal - intersect_areas

    iou_scores = intersect_areas / union_areas
  
    return np.sum(iou_scores) > self.iou_threshold
  
  def debug(self):
    print("coin image shape: {}".format(self.coin_imageset[0].shape))
    print("canvas_size: {}".format(self.canvas_size))
    plt.imshow(self.canvas); plt.grid()
    


def generate_h5_dataset(image_generator, dataset_size, desired_image_size, outfile, set_x_name='train_set_x', set_y_name='train_set_y', set_z_name=None, max_coins_per_image=10):

  data_shape = (dataset_size, desired_image_size, desired_image_size, 3)
  vlen_int_dt = h5py.special_dtype(vlen=np.dtype(int))  # variable length default int
  vlen_np_float32_dt = h5py.special_dtype(vlen=np.dtype(np.float32))
 
  h5_file = h5py.File(outfile, mode='w')

  set_x = h5_file.create_dataset(set_x_name, data_shape, np.uint8, 
                                 maxshape=(None, data_shape[1], data_shape[2], data_shape[3]),
                                 chunks=data_shape
                                )
  
  set_y = h5_file.create_dataset(set_y_name, shape=(dataset_size, ), dtype=vlen_np_float32_dt)

  if set_z_name is not None:
    use_z = True
  else:
    use_z = False 

  if use_z:
    set_z = h5_file.create_dataset(set_z_name, shape=(dataset_size, ), dtype=vlen_int_dt)

  for i in range(dataset_size):
    img, y, z = image_generator.generate_single_image(np.random.randint(max_coins_per_image)+1)
    img = PIL.Image.fromarray(img).resize((desired_image_size, desired_image_size), PIL.Image.BICUBIC)
    img = np.array(img)

    set_x[i, ...] = img
    set_y[i, ...] = y

    if use_z:
      set_z[i, ...] = z
 
  h5_file.close()
  
