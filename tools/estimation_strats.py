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

class ImageTransformation(ConfidenceEstimationStrategy):
    def __init__(self, limit, noise=None):
        self.limit = limit
        self.noise = noise

    def generate_samples(self, eval_points, n, img_shape):
        tiled_points = tf.tile(tf.expand_dims(eval_points, 0), [n, 1, 1, 1, 1])

        images = tf.reshape(tiled_points, (-1,) + img_shape)
        transformed_points = self.transform_points(images, n)
        points = tf.reshape(transformed_points, tf.shape(tiled_points))
        if self.noise is not None:
            points = points + \
                    tf.random_uniform(tf.shape(tiled_points), minval=-1, \
                                         maxval=1) * self.noise

        return points

    def transform_points(self, images, n):
        return images



class ImageTranslation(ImageTransformation):
    def transform_points(self, images, n):
        translations = tf.random_uniform((tf.shape(images)[0], 2), -self.limit, self.limit)

        translated_points = tf.contrib.image.translate(images, translations, interpolation='BILINEAR')
        return translated_points

class ImageRotation(ImageTransformation):
    def transform_points(self, images, n):
        rotations = tf.random_uniform((tf.shape(images)[0],), -self.limit, self.limit)

        rotated_points = tf.contrib.image.rotate(images, rotations, interpolation='BILINEAR')
        return rotated_points

class ImagePixelScale(ImageTransformation):
    def transform_points(self, images, n):
        k = tf.shape(images)[0]
        scalings = tf.random_uniform((k,), -self.limit, self.limit)
        scalings = tf.reshape((1 + scalings), (k,) + (1, 1, 1))
        scaled_by_pixel_points = tf.clip_by_value(images * scalings, clip_value_min=0, clip_value_max=1)
        return scaled_by_pixel_points

class ImageCropAndResize(ImageTransformation):
    def transform_points(self, images, n):
        k = tf.shape(images)[0]
        bounds = tf.random_uniform((k, 1), 0, self.limit)
        boxes = tf.concat([bounds, bounds, 1 - bounds, 1 - bounds], axis=1)
        cropped_images = tf.image.crop_and_resize(images, boxes, tf.range(k), [32, 32])
        return cropped_images


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
        self.adjustment = self.adjustment_to_op[adjustment]
        self.noise = noise
        if adjustment == self.Adjustment.BRIGHTNESS or adjustment == self.Adjustment.HUE:
            self.kwargs = {"max_delta": max_delta}
        elif adjustment == self.Adjustment.CONTRAST or adjustment == self.Adjustment.SATURATION:
            self.kwargs = {"lower": lower, "upper": upper}

    def generate_samples(self, eval_points, n, img_shape):
        tiled_points = tf.tile(tf.expand_dims(eval_points, 0), [n, 1, 1, 1, 1])

        images = tf.reshape(tiled_points, (-1, ) + img_shape)
        adjusted_points = self.adjustment(images, **self.kwargs)

        points = tf.reshape(adjusted_points, tf.shape(tiled_points))
        if self.noise is not None:
            points = points + \
                     tf.random_uniform(tf.shape(tiled_points), minval=-1, \
                                       maxval=1) * self.noise
        return points


def get_strat(strat, strat_param):
    if strat == "uniform":
        return UniformSampling(strat_param)
    elif strat == "translate":
        return ImageTranslation(strat_param)
    elif strat == "rotate":
        return ImageRotation(strat_param)
    elif strat == "pixel_scale":
        return ImagePixelScale(strat_param)
    elif strat == "crop_resize":
        return ImageCropAndResize(strat_param)
    return None
