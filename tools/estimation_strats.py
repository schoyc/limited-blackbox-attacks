import tensorflow as tf
import numpy as np

from enum import Enum

class ConfidenceEstimationStrategy():

    def generate_samples(self, eval_points, n, img_shape):
        return eval_points

class UniformSampling(ConfidenceEstimationStrategy):

    def __init__(self, radius):
        self.radius = radius

    def generate_samples(self, eval_points, n, img_shape):
        tiled_points = tf.tile(tf.expand_dims(eval_points, 0), [n, 1, 1, 1, 1])
        noised_eval_im = tiled_points + \
                       tf.random_uniform(tf.shape(tiled_points), minval=-1, \
                                         maxval=1) * self.radius

        return noised_eval_im


class ImageTranslation(ConfidenceEstimationStrategy):

    def __init__(self, translation_limit, noise=None):
        self.translation_limit = translation_limit
        self.noise = noise

    def generate_samples(self, eval_points, n, img_shape):
        tiled_points = tf.tile(tf.expand_dims(eval_points, 0), [n, 1, 1, 1, 1])

        images = tf.reshape(tiled_points, (-1) + img_shape)
        translations = tf.random_uniform((images.shape[0], 2), -self.translation_limit, self.translation_limit + 1, dtype=tf.int32)

        translated_points = tf.contrib.image.translate(images, tf.cast(translations, tf.float32))

        points = tf.reshape(translated_points, tf.shape(tiled_points))
        if self.noise is not None:
            points = points + \
                    tf.random_uniform(tf.shape(tiled_points), minval=-1, \
                                         maxval=1) * self.noise

        return points

class ImageAdjustment(ConfidenceEstimationStrategy):

    class Adjustment(Enum):
        BRIGHTNESS = 0
        HUE = 1
        CONTRAST = 2
        SATURATION = 3

    adjustment_to_op = {
        Adjustment.BRIGHTNESS: tf.image.random_brightness,
        Adjustment.CONTRAST: tf.image.random_contrast,
        Adjustment.HUE: tf.image.random_hue,
        Adjustment.SATURATION: tf.image.random_saturation
    }

    def __init__(self, adjustment, noise=None, lower=None, upper=None, max_delta=None):
        self.adjustment = self.adjustment_to_op(adjustment)
        if adjustment == self.Adjustment.BRIGHTNESS or adjustment == self.Adjustment.HUE:
            self.kwargs = {"max_delta": max_delta}
        elif adjustment == self.Adjustment.CONTRAST or adjustment == self.Adjustment.SATURATION:
            self.kwargs = {"lower": lower, "upper": upper}

    def generate_samples(self, eval_points, n, img_shape):
        tiled_points = tf.tile(tf.expand_dims(eval_points, 0), [n, 1, 1, 1, 1])

        images = tf.reshape(tiled_points, (-1) + img_shape)
        adjusted_points = self.adjustment(images, **self.kwargs)

        points = tf.reshape(adjusted_points, tf.shape(tiled_points))
        if self.noise is not None:
            points = points + \
                     tf.random_uniform(tf.shape(tiled_points), minval=-1, \
                                       maxval=1) * self.noise
        return points
