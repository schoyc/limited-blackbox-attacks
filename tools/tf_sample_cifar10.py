from tools.utils import optimistic_restore
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, InputLayer

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import functools
import os

SIZE = 32

# to make this work, you need to download:
# http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
# and decompress it in the `data` directory

_CIFAR10_CHECKPOINT_NAME = 'keras_cifar10_trained_model_logits.ckpt'
CIFAR_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__),
    'models',
    _CIFAR10_CHECKPOINT_NAME
)

CIFAR_10_WEIGHTS_PATH = "./models/keras_cifar10_trained_model_weights.h5"

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
        image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image

# input is [batch, 256, 256, 3], pixels in [0, 1]
# output is [batch, 10]
_cifar_initialized = False
def model(sess, image):
    global _cifar_initialized
    network_fn = _get_model(reuse=_cifar_initialized)
    size = SIZE

    preprocessed = _preprocess(image, size, size)
    logits, _ = network_fn(preprocessed)
    # logits = logits[:,1:] # ignore background class
    predictions = tf.argmax(logits, 1)

    # if not _cifar_initialized:
    #     optimistic_restore(sess, CIFAR_CHECKPOINT_PATH)
    #     _cifar_initialized = True

    return logits, predictions

def load_cifar_model():
    cifar10 = cifar10_logits()
    cifar10.load_weights(CIFAR_10_WEIGHTS_PATH, by_name=True)
    return cifar10


def cifar10_logits():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', name='conv2d_1', input_shape=(32, 32, 3)))
    model.add(Activation('relu', name='activation_1'))
    model.add(Conv2D(32, (3, 3), name='conv2d_2'))
    model.add(Activation('relu', name='activation_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1'))
    model.add(Dropout(0.25, name='dropout_1'))

    model.add(Conv2D(64, (3, 3), padding='same', name='conv2d_3'))
    model.add(Activation('relu', name='activation_3'))
    model.add(Conv2D(64, (3, 3), name='conv2d_4'))
    model.add(Activation('relu', name='activation_4'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_2'))
    model.add(Dropout(0.25, name='dropout_2'))

    model.add(Flatten(name='flatten_1'))
    model.add(Dense(512, name='dense_1'))
    model.add(Activation('relu', name='activation_5'))
    model.add(Dropout(0.5, name='dropout_3'))
    model.add(Dense(10, name='dense_2'))
    model.add(Activation('linear', name='logits'))

    return model
