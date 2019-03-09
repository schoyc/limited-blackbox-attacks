from tools.utils import optimistic_restore
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, InputLayer
from tensorflow.keras.models import load_model, Model

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import functools
import os

SIZE = 32

CIFAR_10_MODEL_PATH = "./models/cifar10_ResNet20v1_model.h5"

def _get_model(reuse):
    # arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    func = load_cifar_model()
    @functools.wraps(func)
    def network_fn(images):
        # with slim.arg_scope(arg_scope):
        return func(images)
    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size
    return network_fn

def _preprocess(image, height, width, scope=None):
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
        # image = tf.subtract(image, 0.5)
        # image = tf.multiply(image, 2.0)
        return image

# input is [batch, 256, 256, 3], pixels in [0, 1]
# output is [batch, 10]
_cifar_initialized = False
def model(sess, image):
    global _cifar_initialized
    network_fn = _get_model(reuse=_cifar_initialized)
    size = SIZE

    preprocessed = _preprocess(image, size, size)
    logits = network_fn(preprocessed)
    # logits = logits[:,1:] # ignore background class
    predictions = tf.argmax(logits, 1)

    # if not _cifar_initialized:
    #     optimistic_restore(sess, CIFAR_CHECKPOINT_PATH)
    #     _cifar_initialized = True

    return logits, predictions

def load_cifar_model():
    model = load_model(CIFAR_10_MODEL_PATH)
    logits = Model(inputs=model.input, outputs=model.layers[-2].output)
    return logits

