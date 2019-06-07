import numpy as np
from keras.preprocessing.image import Iterator, ImageDataGenerator

#from datasets.coin_detection import MultiCoinArtificialImageGenerator
from model.convnet.detection import preprocess_true_boxes


import PIL

class SyntheticImageDataGenerator(object):

    ''' 
    Parameters:
    -----------
    n: Integer, total number of samples (per epoch)
    image_generators: either a single instance of ImageGenerator or a tuple of them. In case of tuple, ImageGenerators will be sampled uniformly.

    max_object_per_image: maximum number of object (coins) per image
    image_size: size of generated images
    conv_height: number of row of grid cell, output of conv model height dim 
    conv_width: number of col of grid cell, output of conv model width dim

    '''

    def __init__(self, n, image_generators, max_object_per_image=1, image_size=453, conv_height=9, conv_width=9):
	self.n = n
        self.image_generators = image_generators
	self.max_object_per_image = max_object_per_image
	self.image_size = image_size
	self.conv_height = conv_height
	self.conv_width = conv_width

	self.num_of_image_generators = len(image_generators) if type(image_generators) == tuple else 1

    def dynamically_return_generators(self, n):
        if n == 1:
	   return self.image_generators
	else:
	   img_gen_idx = np.random.randint(n)
	   return self.image_generators[img_gen_idx]

    def generate_single_image(self):
        ''' Generate a simgle image 

	Returns:
	-----------
	img: numpy array representing the image
	y: 1-dim array of box params (with dtype='<f4'
	
	'''
        img_generator = self.dynamically_return_generators(self.num_of_image_generators)
        #img, y, z = img_generator.generate_single_image(np.random.randint(self.max_object_per_image) + 1)
        img, y, z = img_generator.generate_single_image(np.random.randint(self.max_object_per_image) + 1)
        img = PIL.Image.fromarray(img).resize((self.image_size, self.image_size), PIL.Image.BICUBIC)
	img = np.array(img)
	
	return img, y

    def flow(self, batch_size=32, save_to_dir=None, save_prefix='', save_format='png'):
        return NumpyArrayIterator(self, 
	                          batch_size=batch_size, 
				  save_to_dir=save_to_dir, 
				  save_prefix=save_prefix, 
				  save_format=save_format)


class NumpyArrayIterator(Iterator):

    def __init__(self, image_data_generator, batch_size=32, save_to_dir=None, save_prefix='', save_format='png', dtype='float32'):
        self.image_data_generator = image_data_generator
	self.save_to_dir = save_to_dir
	self.save_prefix = save_prefix
	self.save_format = save_format
	self.dtype = dtype
	super(NumpyArrayIterator, self).__init__(image_data_generator.n, 
	                                         batch_size,
                                                 False,   # shuffle does not matter
                                                 0)       # seed does not matter 

    def _get_batches_of_transformed_samples(self, index_array):

       #print(index_array)
       n = len(index_array)
       image_size = self.image_data_generator.image_size

       batch_x = np.zeros((n, image_size, image_size, 3), dtype=self.dtype)
       batch_y = np.empty((n), dtype=object)

       for k in range(len(index_array)):
          img, y = self.image_data_generator.generate_single_image()
          batch_x[k] = img
	  batch_y[k] = y
  
       batch_y_for_training = preprocess_true_boxes(batch_y, 
                                                    self.image_data_generator.max_object_per_image, 
						    self.image_data_generator.conv_height, 
						    self.image_data_generator.conv_width)

       return batch_x, batch_y_for_training

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

