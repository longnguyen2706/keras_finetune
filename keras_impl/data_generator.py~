import numpy as np

from slim_utils import get_random_decoded_images


class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self):
    return

  def generate(self, sess, image_lists, batch_size, category,
                          image_dir, jpeg_data_tensor, decoded_image_tensor):
      # Infinite loop
      while 1:
          # Generate order of exploration of dataset

          (data, ground_truth, _) = get_random_decoded_images(sess,
                                                              image_lists, batch_size, category,
                                                              image_dir, jpeg_data_tensor,
                                                              decoded_image_tensor)
          yield data, ground_truth

  # def __get_exploration_order(self, list_IDs):
  #     'Generates order of exploration'
  #     # Find exploration order
  #     indexes = np.arange(len(list_IDs))
  #     if self.shuffle == True:
  #         np.random.shuffle(indexes)
  #
  #     return indexes



# def sparsify(y):
#   'Returns labels in binary NumPy array'
#   n_classes = # Enter number of classes
#   return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
#                    for i in range(y.shape[0])])
