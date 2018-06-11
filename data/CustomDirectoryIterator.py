from keras.preprocessing.image import DirectoryIterator

class DirectoryIteratorWith1x1ConvTarget(DirectoryIterator):

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x, batch_y = super(DirectoryIteratorWith1x1ConvTarget, self)._get_batches_of_transformed_samples(index_array)

        reshaped_batch_y = batch_y[:, None, None, :]

	return batch_x, reshaped_batch_y
