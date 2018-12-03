import numpy as np
import PIL
from skimage import exposure, img_as_int
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from PIL import ImageDraw, ImageFont, ImageFilter
from data.CustomDirectoryIterator import DirectoryIteratorWith1x1ConvTarget

from .ImageDataGeneratorObjectDetection import apply_rot90, apply_gaussian_blur, apply_color_shift, apply_contrast_stretching, apply_histogram_equalization, apply_adaptive_equalization, apply_cut_out

class CustomImageDataGenerator(ImageDataGenerator):

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

        ImageDataGenerator.__init__(self, *args, **kwargs)

    def get_random_transform(self, img_shape, seed=None):

        #print("ImageDataGeneratorObjectDetection.get_random_transform(...)")
        transform_params = super(ImageDataGeneratorObjectDetection, self).get_random_transform(img_shape, seed=seed)

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
    
    def perform_center_crop(self, x, min_size=None):
        """ Returns a center crop of an image"""
        n_row, n_col, _ = x.shape

        if min_size is None:
            min_size = np.min(n_row, n_col)

        start_r = np.ceil((n_row - min_size)/2.).astype('int')
        start_c = np.ceil((n_col - min_size)/2.).astype('int')
    
        return self.crop(x, start_r, start_c, min_size)
        
    def perform_top_right_crop(self, x, min_size=None):
        """ Returns a top right crop of an image"""
        n_row, n_col, _ = x.shape

        if min_size is None:
            min_size = np.min(n_row, n_col)
    
        start_r = 0
        start_c = n_col - min_size
    
        return self.crop(x, start_r, start_c, min_size)
        
    def perform_top_left_crop(self, x, min_size=None):
        """ Returns a top right crop of an image"""
        n_row, n_col, _ = x.shape
        if min_size is None:
            min_size = np.min(n_row, n_col)
    
        start_r = 0
        start_c = 0
    
        return self.crop(x, start_r, start_c, min_size)

    def perform_bottom_left_crop(self, x, min_size=None):
        """ Returns a top right crop of an image"""
        n_row, n_col, _ = x.shape
        if min_size is None:
            min_size = np.min(n_row, n_col)
    
        start_r = n_row - min_size
        start_c = 0
    
        return self.crop(x, start_r, start_c, min_size)

    def perform_bottom_right_crop(self, x, min_size=None):
        """ Returns a top right crop of an image"""
        n_row, n_col, _ = x.shape
        if min_size is None:
            min_size = np.min(n_row, n_col)
    
        start_r = n_row - min_size
        start_c = n_col - min_size
    
        return self.crop(x, start_r, start_c, min_size)

    def crop(self, x, r, c, size):
        return x[r:r+size, c:c+size]


    def perform_pca_color_shift(self, x, prob=0.5):
        ''' please refer to the paper '''
        if np.random.random() < prob:
            renorm_x = np.reshape(x, (x.shape[0] * x.shape[1], 3)) 

            renorm_x = renorm_x.astype('float32')
            renorm_x -= np.mean(renorm_x, axis=0)
            renorm_x /= np.std(renorm_x, axis=0)

            cov = np.cov(renorm_image, rowvar=False)
            lambdas, p = np.linalg.eig(cov)

            alphas = np.random.normal(0, 0.1, 3)

            #delta = p[:,0]*alphas[0]*lambdas[0] + p[:,1]*alphas[1]*lambdas[1] + p[:,2]*alphas[2]*lambdas[2]
            delta = np.dot(p, alphas*lambdas)

            delta = (delta*255.).astype('int8')

            x = np.maximum(np.minimum(x + delta, 255), 0).astype('uint8')

        return x

    def flow_from_directory_with_1x1_conv_target(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest'):

        return DirectoryIteratorWith1x1ConvTarget(
	    directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation)


def perform_center_crop(x, min_size=None):
    """ Returns a center crop of an image"""
    n_row, n_col, _ = x.shape

    if min_size is None:
        min_size = np.min(n_row, n_col)

    start_r = np.ceil((n_row - min_size)/2.).astype('int')
    start_c = np.ceil((n_col - min_size)/2.).astype('int')
    
    return crop(x, start_r, start_c, min_size)


def crop(x, r, c, size):
    return x[r:r+size, c:c+size]

def _pil_center_crop(self):

    width, height = self.size
    min_size = min(height, width)

    start_r = int(np.ceil((height - min_size)/2.))
    start_c = int(np.ceil((width - min_size)/2.))

    return self.crop((start_c, start_r, start_c+min_size, start_r+min_size))

PIL.Image.Image.center_crop = _pil_center_crop

