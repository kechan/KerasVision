import numpy as np
from skimage import exposure, img_as_int
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from PIL import ImageDraw, ImageFont, ImageFilter
from data.CustomDirectoryIterator import DirectoryIteratorWith1x1ConvTarget

# import matplotlib.pyplot as plt


class CustomImageDataGenerator(ImageDataGenerator):

    def __init__(self, rot90=False, gaussian_blur_range=None, color_shift=None, contrast_stretching=False, histogram_equalization=False, adaptive_equalization=False, cut_out=None, *args, **kwargs):

        self.rot90 = rot90

        self.gaussian_blur_range = gaussian_blur_range

        self.color_shift = color_shift

        self.contrast_stretching = contrast_stretching
        self.histogram_equalization = histogram_equalization
        self.adaptive_equalization = adaptive_equalization

        self.cut_out = cut_out

        ImageDataGenerator.__init__(self, *args, **kwargs)


    def random_transform(self, x, seed=None):
        
        need_to_operate_on_no_norm_image = False
        if self.contrast_stretching or \
           self.histogram_equalization or \
           self.adaptive_equalization or \
           self.color_shift or \
           self.cut_out or \
           self.gaussian_blur_range:
            need_to_operate_on_no_norm_image = True

        if need_to_operate_on_no_norm_image:
            x = x.astype('uint8')

        x = self.perform_custom_transform(x)
    
        if need_to_operate_on_no_norm_image:
            x = x.astype('float32')
    
        x = super(CustomImageDataGenerator, self).random_transform(x, seed=seed)

        return x


    def perform_custom_transform(self, x):

        if self.rot90:
            x = self.perform_rot90(x)

        if self.gaussian_blur_range:
            x = self.perform_gaussian_blur_range(x, self.gaussian_blur_range)

        if self.color_shift:
            x = self.perform_color_shift(x, rgb_shift=self.color_shift, prob=1.0)

        if self.contrast_stretching:
            x = self.perform_contrast_stretching(x, prob=1.0)

        if self.histogram_equalization:
            x = self.perform_histogram_equalization(x)

        if self.adaptive_equalization:
            x = self.perform_adaptive_equalization(x)

        if self.cut_out:
            x = self.perform_cut_out(x, n_holes=self.cut_out[0], length=self.cut_out[1])

        return x

    def perform_rot90(self, x):
        return np.rot90(x, k=np.random.randint(4), axes=[0, 1])


    def perform_color_shift(self, x, rgb_shift=None, prob=0.5):
        if np.random.random() < prob:
            if rgb_shift is None:
                rgb_shift = [10, 10, 10]
    
            r_shift = np.random.randint(-rgb_shift[0], rgb_shift[0])
            g_shift = np.random.randint(-rgb_shift[1], rgb_shift[1])
            b_shift = np.random.randint(-rgb_shift[2], rgb_shift[2])
    
            color_shifted_x = x + np.array([r_shift, g_shift, b_shift])
    
            x = np.maximum(np.minimum(color_shifted_x, 255), 0).astype('uint8')
    
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

    def perform_gaussian_blur_range(self, x, blur_range):
        pil_img = image.array_to_img(x)

        radius = blur_range * np.random.rand()
        blurred_img = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))

        return np.array(blurred_img)


    def perform_contrast_stretching(self, x, prob=0.5):
        if np.random.random() < prob:
            p2, p98 = np.percentile(x, (2, 98))
            x = exposure.rescale_intensity(x, in_range=(p2, p98))

        return x

    def perform_histogram_equalization(self, x, prob=0.5):
        if np.random.random() < prob:
            x = exposure.equalize_hist(x)

        return x

    def perform_adaptive_equalization(self, x, prob=0.5):
        if np.random.random() < prob:
            x = exposure.equalize_adapthist(x, clip_limit=0.03)

        return x

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

    def perform_cut_out(self, im, n_holes=0, length=0, prob=0.5):
        ''' randomly cut out some squares '''
        if np.random.random() < prob:
            h, w, _ = im.shape
            mask = np.ones((h, w), np.int32)
            for n in range(n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = int(np.clip(y - length / 2, 0, h))
                y2 = int(np.clip(y + length / 2, 0, h))
                x1 = int(np.clip(x - length / 2, 0, w))
                x2 = int(np.clip(x + length / 2, 0, w))
                mask[y1: y2, x1: x2] = 0.
    
            mask = mask[:,:,None]
            im = (im * mask).astype('uint8')

        return im


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
