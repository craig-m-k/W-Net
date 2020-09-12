"""
Data pipeline tools for WNet.
"""

import tensorflow as tf
import os
import numpy as np
import glob

AUTOTUNE = tf.contrib.data.AUTOTUNE

class DataReader:
    def __init__(self, file_path, batch_size=1):
        self.file_path = file_path
        self.batch_size = batch_size

    def flip(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        return x

    def color(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.image.random_hue(x, 0.08)
        x = tf.image.random_saturation(x, 0.6, 1.6)
        x = tf.image.random_brightness(x, 0.05)
        x = tf.image.random_contrast(x, 0.7, 1.3)
        return x

    def rotate(self, x: tf.Tensor) -> tf.Tensor:
        degrees = np.random.randint(-5,5)
        return tf.contrib.image.rotate(x, degrees * np.pi / 180, interpolation='BILINEAR')

    def parse_image(self, filename):
        image_string = tf.read_file(self.file_path + '/' + filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3, dct_method='INTEGER_ACCURATE')
        image_resized = tf.image.resize_images(image_decoded, [224, 224])
        image_resized = color(image_resized)
        image_resized = self.flip(image_resized)
        image_resized = rotate(image_resized)
        return filename, image_resized

    def get_filenames(self):
        filenames_g = sorted(glob.glob(self.file_path + '/*.jpg'))
        filenames = [os.path.basename(x) for x in filenames_g]
        return filenames

    def input_data(self, num_images=None):
        filenames = self.get_filenames()
        lf = len(filenames)
        print("Number of images: ", lf)
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(self.parse_image)
        dataset = dataset.shuffle(lf)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset.make_initializable_iterator()


