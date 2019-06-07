import glob, os, PIL
import xml.etree.ElementTree as ElementTree

import numpy as np

from keras import backend as K

from keras.utils import to_categorical
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator, DirectoryIterator, Iterator
from keras.preprocessing.image import array_to_img

from PIL import ImageDraw, ImageFont, ImageFilter
from skimage import exposure, img_as_int

from model.convnet.detection import preprocess_true_boxes

#from .CustomImageDataGenerator import perform_rot90_with_tracking, perform_gaussian_blur_range, perform_color_shift, perform_contrast_stretching, perform_histogram_equalization, perform_adaptive_equalization, perform_cut_out

def apply_rot90(x, k):
    return np.rot90(x, k, axes=[0, 1])

def apply_gaussian_blur(x, blur_radius):
    pil_img = image.array_to_img(x)

    blurred_img = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return np.array(blurred_img)

def apply_color_shift(x, color_shift):
    color_shifted_x = x + color_shift
    x = np.maximum(np.minimum(color_shifted_x, 255), 0).astype('uint8')
    return x


def apply_contrast_stretching(x):
    p2, p98 = np.percentile(x, (2, 98))
    x = exposure.rescale_intensity(x, in_range=(p2, p98))
    return x

def apply_histogram_equalization(x):
    # TODO: Debug this, could be dtype issue for x
    x = exposure.equalize_hist(x)
    return x

def apply_adaptive_equalization(x):
    # TODO: Debug this, could be dtype issue for x
    x = exposure.equalize_adapthist(x, clip_limit=0.03)
    return x

def apply_cut_out(im, cut_out):
    h, w, _ = im.shape 
    n_holes, length = cut_out

    mask = np.ones((h, w), np.uint8)
    
    for n in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = int(np.clip(y - length / 2, 0, h))
        y2 = int(np.clip(y + length / 2, 0, h))
        x1 = int(np.clip(x - length / 2, 0, w))
        x2 = int(np.clip(x + length / 2, 0, w))
        mask[y1: y2, x1: x2] = 0
     
    mask = mask[:,:,None]
    im = (im * mask).astype('uint8')

    return im


class NumpyArrayIteratorObjectDetection(NumpyArrayIterator):
    def __init__(self, *args, **kwargs):
        NumpyArrayIterator.__init__(self, *args, **kwargs)

    # TODO: This is a very fragile override, should find a way to do this without a destructive overrride 
    def _get_batches_of_transformed_samples(self, index_array):

	# Added in this override
        def swap(a, i, j):
	    # swap the values of a[i] and a[j]
            tmp = a[i]                  
            a[i] = a[j]
            a[j] = tmp
  
	# output = super(NumpyArrayIteratorObjectDetection, self)._get_batches_of_transformed_samples(index_array)

        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]),
                           dtype=self.dtype)
        batch_y = np.zeros(tuple([len(index_array)] + list(self.y.shape)[1:]), dtype=self.dtype)    # Added in this override

        for i, j in enumerate(index_array):
            x = self.x[j]
            params = self.image_data_generator.get_random_transform(x.shape)
            x = self.image_data_generator.apply_transform(
                x.astype(self.dtype), params)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

            y = self.y[j]                  # Added in this override
            batch_y[i] = np.copy(y)
            if y[0] > 0.:

	        # 4-rotation (np.rot90)
                rot = params['rot90']
                if rot == 1:
                    batch_y[i][1] = 1. - batch_y[i][1]     # left/right flip
                    swap(batch_y[i], 1, 2)

                elif rot == 2:
                    batch_y[i][1] = 1. - batch_y[i][1]     # left/right flip
                    batch_y[i][2] = 1. - batch_y[i][2]     # up/down flip

                elif rot == 3: 
                    batch_y[i][2] = 1. - batch_y[i][2]     # up-down flip
                    swap(batch_y[i], 1, 2)

	        # translations
                batch_y[i][1] -= (params['ty'] / x.shape[1])
                batch_y[i][2] -= (params['tx'] / x.shape[0])
	        
		# scale/zoom
                batch_y[i][1] = ((batch_y[i][1] - 0.5) / params['zy']) + 0.5
                batch_y[i][2] = ((batch_y[i][2] - 0.5) / params['zx']) + 0.5  

                batch_y[i][3] /= params['zx']

		# check if the center c_x, c_y has fallen out of bound, then zeros all the y.
                if batch_y[i][1] < 0.0 or batch_y[i][1] > 1.0 or batch_y[i][2] < 0.0 or batch_y[i][2] > 1.0:
                    batch_y[i] = np.zeros(tuple([1] + list(self.y.shape)[1:]), dtype=K.floatx())


        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        batch_x_miscs = [xx[index_array] for xx in self.x_misc]
        output = (batch_x if batch_x_miscs == []
                  else [batch_x] + batch_x_miscs,)
        if self.y is None:
            return output[0]
        #output += (self.y[index_array],)
        output += (batch_y,)                          # Added in this override
        if self.sample_weight is not None:
            output += (self.sample_weight[index_array],)

        return output
    

class DirectoryIteratorObjectDetection(DirectoryIterator):
    def __init__(self, *args, **kwargs):
        DirectoryIterator.__init__(self, *args, **kwargs)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            x, transform_params = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e7),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_classes), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        elif self.class_mode == 'categorical_with_bounding_box':
	    # to be implemented 
            assert false, "Not implemented."
            batch_y = np.zeros((len(batch_x), self.num_classes), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y


class ImageDataGeneratorObjectDetection(ImageDataGenerator):

    def __init__(self, 
                 rot90=False, 
                 gaussian_blur_range=None, 
                 color_shift=None, 
                 contrast_stretching=False, 
                 histogram_equalization=False, 
                 adaptive_equalization=False,
                 cut_out=None, 
                 *args, **kwargs):

        self.rot90 = rot90
        self.gaussian_blur_range = gaussian_blur_range
        self.color_shift = color_shift
        self.contrast_stretching = contrast_stretching
        self.histogram_equalization = histogram_equalization
        self.adaptive_equalization = adaptive_equalization
        self.cut_out = cut_out

        #print("Instantiating ImageDataGeneratorObjectDetection")

        ImageDataGenerator.__init__(self, *args, **kwargs)

    def get_random_transform(self, img_shape, seed=None):

        #print("ImageDataGeneratorObjectDetection.get_random_transform(...)")
        transform_params = super(ImageDataGeneratorObjectDetection, self).get_random_transform(img_shape, seed=seed)

	# For object detection, we like to work with equal zoom in x and y direction 
        if transform_params.get('zx') is not None and transform_params.get('zy') is not None:
            transform_params['zy'] = transform_params['zx']

	# Add extra custom transform params
        if self.rot90:
            k = np.random.randint(4)
            transform_params['rot90'] = k
        else:
            transform_params['rot90'] = 0

        if self.gaussian_blur_range:
            blur_radius = self.gaussian_blur_range * np.random.rand()
            transform_params['blur_radius'] = blur_radius

        if self.color_shift:
            if np.random.random() < 0.5:
                r_shift = np.random.randint(-self.color_shift[0], self.color_shift[0])
                g_shift = np.random.randint(-self.color_shift[1], self.color_shift[1])
                b_shift = np.random.randint(-self.color_shift[2], self.color_shift[2])

                transform_params['color_shift'] = np.array([r_shift, g_shift, b_shift])

        if self.contrast_stretching:
            if np.random.random() < 0.5:
                transform_params['contrast_stretching'] = True 

        if self.histogram_equalization:
            if np.random.random() < 0.5:
                transform_params['histogram_equalization'] = True

        if self.adaptive_equalization:
            if np.random.random() < 0.5:
                transform_params['adaptive_equalization'] = True

        if self.cut_out:
            if np.random.random() < 1.0:
                transform_params['cut_out'] = self.cut_out

	#print("transform_params: {}".format(transform_params))
        return transform_params

    def apply_transform(self, x, transform_parameters):

        #print("ImageDataGeneratorObjectDetection.apply_transform(...)")

        need_uint8_temporarily = False
        if self.contrast_stretching or self.histogram_equalization or self.adaptive_equalization or self.color_shift or self.cut_out or self.gaussian_blur_range:
            need_uint8_temporarily = True

        if need_uint8_temporarily:
            x = x.astype('uint8')    

	# apply custom first 
        if transform_parameters.get('rot90') is not None:
            k = transform_parameters.get('rot90') 	
            x = apply_rot90(x, k)
        else:
            k = 0

        if transform_parameters.get('blur_radius') is not None:
            blur_radius = transform_parameters.get('blur_radius')
            x = apply_gaussian_blur(x, blur_radius)

        if transform_parameters.get('color_shift') is not None:
            color_shift = transform_parameters.get('color_shift')
            x = apply_color_shift(x, color_shift)

        if transform_parameters.get('contrast_stretching') is not None and transform_parameters.get('contrast_stretching'):
            x = apply_contrast_stretching(x)

        if transform_parameters.get('histogram_equalization') is not None and transform_parameters.get('histogram_equalization'):
            x = apply_histogram_equalization(x)

        if transform_parameters.get('adaptive_equalization') is not None and transform_parameters.get('adaptive_equalization'):
            x = apply_adaptive_equalization(x)

        if transform_parameters.get('cut_out'):
            cut_out = transform_parameters.get('cut_out')
            x = apply_cut_out(x, cut_out)
	
        if need_uint8_temporarily:    # convert back to float32 for the rest of random transforms
            x = x.astype('float32')

        x = super(ImageDataGeneratorObjectDetection, self).apply_transform(x, transform_parameters)

        return x

    
    # override flow to return a NumpyArrayIteratorObjectDetection instead of the normal NumpyArrayIterator
    def flow(self, x,
             y=None, batch_size=32, shuffle=True,
             sample_weight=None, seed=None,
             save_to_dir=None, save_prefix='', save_format='png', subset=None):

        return NumpyArrayIteratorObjectDetection(
            x, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            sample_weight=sample_weight,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset)
    
    # new
    def flow_from_dir_with_random_selection_cropping(self, directory,
                                                           target_size=(224, 224),
                                                           classes=None, 
                                                           max_object_per_image=1,
                                                           bounding_box_adjustment=1.,
                                                           conv_height=9,
                                                           conv_width=9,
                                                           crop_range=(0.3, 0.9),
                                                           class_mode=None,
                                                           batch_size=32,
                                                           shuffe=True,
                                                           seed=None,
                                                           save_to_dir=None,
                                                           debug=False
                                                           ):
        return DirectoryWithRandomSelectionCroppingIterator(directory, self, target_size=target_size, classes=classes,
                                                            max_object_per_image=max_object_per_image,
                                                            bounding_box_adjustment=bounding_box_adjustment,
                                                            conv_height=conv_height,
                                                            conv_width=conv_width,
                                                            crop_range=crop_range,
	                                                    class_mode=class_mode, 
	                                                    batch_size=batch_size, 
	                                                    shuffle=True, 
                                                            seed=seed,
                                                            save_to_dir=save_to_dir,
                                                            debug=debug)


def get_boxes_for_id(voc_path, image_name, list_classes):
    """Get object bounding boxes annotations for given image.

    Parameters
    ----------
    voc_path : str
        Path to VOC directory.
    image_id : str
        Pascal VOC identifier for given image.

    Returns
    -------
    boxes : array of int
        bounding box annotations of class label, xmin, ymin, xmax, ymax as a
        5xN array.
    """
    fname = os.path.join(voc_path, '{}.xml'.format(image_name))
    with open(fname) as in_file:
        xml_tree = ElementTree.parse(in_file)
    root = xml_tree.getroot()
    boxes = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        label = obj.find('name').text
        if label not in list_classes or int(
                difficult) == 1:  # exclude difficult or unlisted classes
            continue
        xml_box = obj.find('bndbox')
        bbox = (int(xml_box.find('xmin').text),
                int(xml_box.find('ymin').text), int(xml_box.find('xmax').text),
                int(xml_box.find('ymax').text), list_classes.index(label))
        
        boxes.extend(bbox)
        
    boxes = np.reshape(np.array(boxes), (-1, 5))
        
    return boxes


class DirectoryWithRandomSelectionCroppingIterator(Iterator):

    def __init__(self, directory, image_data_generator,
                 target_size=(224, 224),
                 classes=None, 
                 max_object_per_image=1,
                 bounding_box_adjustment=1.,
                 conv_height=9,
                 conv_width=9,
		 crop_range=(0.3, 0.9),
                 class_mode=None,
                 batch_size=32,
                 shuffle=True,
		 seed=None,
                 save_to_dir=None,
                 debug=False):

        self.filenames = glob.glob(os.path.join(directory, '*.jpg')) 
        self.filenames.extend(glob.glob(os.path.join(directory, '*.JPG')))
        self.filenames.extend(glob.glob(os.path.join(directory, '*.png')))
        self.filenames.extend(glob.glob(os.path.join(directory, '*.PNG')))

        self.num_samples = len(self.filenames)

        self.directory = directory
        self.image_data_generator = image_data_generator 
        self.target_size = target_size
        self.classes = classes   # list of classes
        self.max_object_per_image = max_object_per_image
        self.bounding_box_adjustment = bounding_box_adjustment
        self.conv_height = conv_height
        self.conv_width = conv_width
        self.crop_range = crop_range
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.debug = debug

	# retrieve and store img_boxes_cache
        self.img_boxes_cache = {}
	
        img_names = [os.path.splitext(os.path.basename(img))[0] for img in self.filenames]

        for img_name in img_names:
            boxes = get_boxes_for_id(self.directory, img_name, classes)
            boxes[..., :4] = boxes[..., :4] * self.bounding_box_adjustment
            self.img_boxes_cache[img_name] = boxes


        super(DirectoryWithRandomSelectionCroppingIterator, self).__init__(self.num_samples, 
                                                                           batch_size, 
                                                                           shuffle, 
                                                                           seed
                                                                           )

    def _get_batches_of_transformed_samples(self, index_array):

        #print("index_array: {}".format(index_array))

        def swap(a, i, j):    # swap the column i and j of a[]
            tmp = np.copy(a[..., i])
            a[..., i] = a[..., j]
            a[..., j] = tmp

        #print(index_array)
        batch_x = np.zeros((len(index_array),) + self.target_size + (3,), dtype='float32')
        batch_y = np.empty((len(index_array)), dtype=object)

	# build batch of image data
        for i, j in enumerate(index_array):
	    # select the big image
            fname = self.filenames[j]
            img_name = os.path.splitext(os.path.basename(fname))[0]
            boxes = self.img_boxes_cache[img_name]
            img = PIL.Image.open(os.path.join(self.directory, img_name+".JPG"))
            img_data = np.array(img)

	    # crop-sample a sub-image
            W = img.width
            H = img.height
            crop_lower_range, crop_higher_range = self.crop_range
            ratio = (crop_higher_range - crop_lower_range) * np.random.random_sample() + crop_lower_range
            h = int(ratio * np.min([W, H]))
            w = h
            #h = 1024     # TODO: need to parameterize this, a square whose side is 1/2 of min(W, H)
            #w = 1024

            x_offset = np.random.randint(0, W-w)
            y_offset = np.random.randint(0, H-h)

            sub_img_data = img_data[y_offset: y_offset+h, x_offset: x_offset+w]
	    
	    # find which coins fall into this img
	    # (c_x, c_y) is centers of boxes
            c_x = (boxes[:, 2] + boxes[:, 0]) * 0.5
            c_y = (boxes[:, 3] + boxes[:, 1]) * 0.5

            mask = (x_offset < c_x) * (c_x < x_offset + w) * (y_offset < c_y) * (c_y < y_offset + h)

            coins_in_sub_img = boxes[mask]

	    # transform coordinates of coins_in_sub_img
            offset = np.reshape(np.array([x_offset, y_offset, x_offset, y_offset, 0]), (1, 5))
            sub_boxes = coins_in_sub_img - offset

	    # convert to c_xyr 
            c_r = np.minimum((sub_boxes[..., 2] - sub_boxes[..., 0]) * 0.5, (sub_boxes[..., 3] - sub_boxes[..., 1]) * 0.5)
            c_r = np.reshape(c_r, (-1, 1))

            c_xy = sub_boxes[..., 0:2] + c_r

            # renorm to img size
            c_xyr = np.concatenate([c_xy, c_r], axis=-1)
            c_xyr = c_xyr / np.array([[sub_img_data.shape[1], sub_img_data.shape[0], sub_img_data.shape[0]]])

            
            # resize sub image to desired output size
            sub_img = PIL.Image.fromarray(sub_img_data)
            sub_img = sub_img.resize(self.target_size, PIL.Image.BICUBIC)
            x = np.array(sub_img)

            params = self.image_data_generator.get_random_transform(x.shape)
            #print("params: {}".format(params))
            x = self.image_data_generator.apply_transform(x, params)
            x = x.astype(K.floatx())
            x = self.image_data_generator.standardize(x)

	    # Apply transform to c_xyr

	    # 4-rotation (np.rot90)
            rot = params['rot90']
            if rot == 1:
                c_xyr[..., 0] = 1. - c_xyr[..., 0]   # flip left/right (i.e. x)
                swap(c_xyr, 0, 1)                    # transpose
            elif rot == 2:
                c_xyr[..., 0] = 1. - c_xyr[..., 0]     # left/right flip
                c_xyr[..., 1] = 1. - c_xyr[..., 1]     # up/down flip
            elif rot == 3:
                c_xyr[..., 1] = 1. - c_xyr[..., 1]     # up-down flip
                swap(c_xyr, 0, 1)

	    # translation     
            c_xyr[..., 0] -= (params['ty'] / x.shape[1])
            c_xyr[..., 1] -= (params['tx'] / x.shape[0])

	    # scale/zoom
            c_xyr[..., 0] = ((c_xyr[..., 0] - 0.5) / params['zy']) + 0.5
            c_xyr[..., 1] = ((c_xyr[..., 1] - 0.5) / params['zx']) + 0.5  

            c_xyr[..., 2] /= params['zx']

	    # check if the center c_x, c_y has fallen out of bound, then zeros all the entries.
            mask = ((c_xyr[..., 0] < 0.0) + (c_xyr[..., 0] > 1.0) + (c_xyr[..., 1] < 0.0) + (c_xyr[..., 1] > 1.0) > 0)
            mask = (1. - mask).astype('bool')
            c_xyr = c_xyr[mask]

            #mask = mask[..., np.newaxis]
	    #c_xyr = mask * c_xyr      #TODO: fix this, can't just zero them out, they need to be removed.
	    #print("c_xyr: {}".format(c_xyr))

	    # add one-hot class to the end, we need the -1 because the 1st class UNK is the background which we don't track here
            one_hot_class = to_categorical(sub_boxes[..., -1]-1, len(self.classes)-1) 
            one_hot_class = one_hot_class[mask]

            c_xyrc = np.concatenate([c_xyr, one_hot_class], axis=-1)

            # flatten c_xyrc
            c_xyrc = np.reshape(c_xyrc, (-1,)).astype(np.float32)

            batch_x[i] = x
            batch_y[i] = c_xyrc
	
        batch_y_for_training = preprocess_true_boxes(batch_y,
	                                             self.max_object_per_image,
	                                             self.conv_height,
	                                             self.conv_width)

        if self.debug:
            return batch_x, batch_y    # return batch_y for debugging affine related data augmentation
        else:
            return batch_x, batch_y_for_training

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

