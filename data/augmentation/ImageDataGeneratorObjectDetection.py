import numpy as np

from keras import backend as K

from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator, DirectoryIterator
from keras.preprocessing.image import transform_matrix_offset_center, apply_transform, array_to_img

class NumpyArrayIteratorObjectDetection(NumpyArrayIterator):
    def __init__(self, *args, **kwargs):
        NumpyArrayIterator.__init__(self, *args, **kwargs)

    def _get_batches_of_transformed_samples(self, index_array):
        #print("index_array: {}".format(index_array))
        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]),
                           dtype=K.floatx())

        batch_y = np.zeros(tuple([len(index_array)] + list(self.y.shape)[1:]), dtype=K.floatx())

        for i, j in enumerate(index_array):
            x = self.x[j]
            x, transform_params = self.image_data_generator.random_transform(x.astype(K.floatx()))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

	    #print("transform_params: {}".format(transform_params))
	    y = self.y[j]
	    batch_y[i] = np.copy(y)
	    if y[0] > 0.:

	        # translations
	        batch_y[i][1] -= (transform_params['ty'] / x.shape[1])
	        batch_y[i][2] -= (transform_params['tx'] / x.shape[0])
	        
		# scale/zoom
		batch_y[i][1] = ((batch_y[i][1] - 0.5) / transform_params['zy']) + 0.5
		batch_y[i][2] = ((batch_y[i][2] - 0.5) / transform_params['zx']) + 0.5  

		batch_y[i][3] /= transform_params['zx']

		# check if the center c_x, c_y has fallen out of bound, then zeros all the y.
		if batch_y[i][1] < 0.0 or batch_y[i][1] > 1.0 or batch_y[i][2] < 0.0 or batch_y[i][2] > 1.0:
                    batch_y[i] = np.zeros(tuple([1] + list(self.y.shape)[1:]), dtype=K.floatx())


        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        #if self.y is None:
        #    return batch_x
        #batch_y = self.y[index_array]

        return batch_x, batch_y

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
            batch_y = np.zeros((len(batch_x), self.num_classes), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y


class ImageDataGeneratorObjectDetection(ImageDataGenerator):

    def __init__(self, *args, **kwargs):
        ImageDataGenerator.__init__(self, *args, **kwargs)

    def random_transform(self, x, seed=None):
        """Randomly augment a single image tensor.

        # Arguments
            x: 3D tensor, single image.
            seed: random seed.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.deg2rad(np.random.uniform(-self.rotation_range, self.rotation_range))
        else:
            theta = 0

        if self.height_shift_range:
            try:  # 1-D array-like or int
                tx = np.random.choice(self.height_shift_range)
                tx *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                tx = np.random.uniform(-self.height_shift_range,
                                       self.height_shift_range)
            if np.max(self.height_shift_range) < 1:
                tx *= x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            try:  # 1-D array-like or int
                ty = np.random.choice(self.width_shift_range)
                ty *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                ty = np.random.uniform(-self.width_shift_range,
                                       self.width_shift_range)
            if np.max(self.width_shift_range) < 1:
                ty *= x.shape[img_col_axis]
        else:
            ty = 0

	# hack to debug
	#tx = -0.45 * x.shape[img_row_axis]
	#ty = -0.25 * x.shape[img_col_axis]

        if self.shear_range:
            shear = np.deg2rad(np.random.uniform(-self.shear_range, self.shear_range))
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else: # for object detection, we may not want to distort the object and bounding box like this.
            # zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
	    zx = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
            zy = zx

	    # hack to debug
	    #zx = 0.8
	    #zy = 0.8

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

	#print("tx, ty = ({}, {})".format(tx, ty))
        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)

        if self.channel_shift_range != 0:
            x = random_channel_shift(x,
                                     self.channel_shift_range,
                                     img_channel_axis)
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)

        if self.brightness_range is not None:
            x = random_brightness(x, self.brightness_range)

	transform_parameters = {'theta': theta,
	                        'tx': tx,
				'ty': ty,
				'zx': zx,
				'zy': zy}

        return x, transform_parameters

    def fit(self, x,
            augment=False,
            rounds=1,
            seed=None):
        """Compute the internal data stats related to the data-dependent transformations, based on an array of sample data.
        Only required if featurewise_center or featurewise_std_normalization or zca_whitening.

        # Arguments
            x: sample data. Should have rank 4.
             In case of grayscale data,
             the channels axis should have value 1, and in case
             of RGB data, it should have value 3.
            augment: Boolean (default: False). Whether to fit on randomly augmented samples.
            rounds: int (default: 1). If augment, how many augmentation passes over the data to use.
            seed: int (default: None). Random seed.
       """
        x = np.asarray(x, dtype=K.floatx())
        if x.ndim != 4:
            raise ValueError('Input to `.fit()` should have rank 4. '
                             'Got array with shape: ' + str(x.shape))
        if x.shape[self.channel_axis] not in {1, 3, 4}:
            warnings.warn(
                'Expected input to be images (as Numpy array) '
                'following the data format convention "' + self.data_format + '" '
                '(channels on axis ' + str(self.channel_axis) + '), i.e. expected '
                'either 1, 3 or 4 channels on axis ' + str(self.channel_axis) + '. '
                'However, it was passed an array with shape ' + str(x.shape) +
                ' (' + str(x.shape[self.channel_axis]) + ' channels).')

        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)
        if augment:
            ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]), dtype=K.floatx())
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]], _ = self.random_transform(x[i])
            x = ax

        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= (self.std + K.epsilon())

        if self.zca_whitening:
            flat_x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
            sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = linalg.svd(sigma)
            s_inv = 1. / np.sqrt(s[np.newaxis] + self.zca_epsilon)
            self.principal_components = (u * s_inv).dot(u.T)

    def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='png', subset=None):

        return NumpyArrayIteratorObjectDetection(
            x, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset)

