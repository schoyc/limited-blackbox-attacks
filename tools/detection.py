import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, InputLayer

class Detector(object):

    def __init__(self, threshold, K=50, size=None, chunk_size=1000, weights_path="./encoder_1.h5"):
        self.threshold = threshold
        self.K = K
        self.size = size
        self.num_queries = 0

        self.buffer = []
        self.memory = []
        self.chunk_size = chunk_size

        self.history = [] # Tracks number of queries (t) when attack was detected
        self.history_by_attack = []
        self.detected_dists = [] # Tracks knn-dist that was detected

        self._init_encoder(weights_path)

    def _init_encoder(self, weights_path):
        # if encoder is None:
        #     self.encode = lambda x: x
        # else:
        #     # Restore model from tf session
        #     # encoder = SiameseEncoder(margin=np.sqrt(10), learning_rate=1e-4)
        #     # encoder.init_sess()
        encoder = cifar10_encoder()
        encoder.load_weights(weights_path, by_name=True)
        self.encoder = encoder
        self.encode = lambda x : encoder.predict(np.expand_dims(x, axis=0))


    def process(self, queries, num_queries_so_far):
        for query in queries:
            self.process_query(query, num_queries_so_far)
            num_queries_so_far += 1

    def process_query(self, query, num_queries_so_far):

        query = np.squeeze(self.encode(query))

        if len(self.memory) == 0 and len(self.buffer) < self.K:
            self.buffer.append(query)
            self.num_queries += 1
            return False

        k = self.K
        all_dists = []

        if len(self.buffer) > 0:
            queries = np.stack(self.buffer, axis=0)
            dists = np.linalg.norm(queries - query, axis=-1)
            all_dists.append(dists)

        for queries in self.memory:
            dists = np.linalg.norm(queries - query, axis=-1)
            all_dists.append(dists)

        dists = np.concatenate(all_dists)
        k_nearest_dists = np.partition(dists, k - 1)[:k, None]
        k_avg_dist = np.mean(k_nearest_dists)

        self.buffer.append(query)
        self.num_queries += 1

        if len(self.buffer) >= self.chunk_size:
            self.memory.append(np.stack(self.buffer, axis=0))
            self.buffer = []

        # print("[debug]", num_queries_so_far, k_avg_dist)
        is_attack = k_avg_dist < self.threshold
        if is_attack:
            self.history.append(self.num_queries)
            self.history_by_attack.append(num_queries_so_far + 1)
            self.detected_dists.append(k_avg_dist)
            # print("[encoder] Attack detected:", str(self.history), str(self.detected_dists))
            self.clear_memory()

    def clear_memory(self):
        self.buffer = []
        self.memory = []

    def get_detections(self):
        history = self.history
        epochs = []
        for i in range(len(history) - 1):
            epochs.append(history[i + 1] - history[i])

        return epochs


class SiameseEncoder(object):
    def __init__(self,
                 margin,
                 learning_rate=5e-6,
                 momentum=0.9,
                 decay=5e-4):

        self.learning_rate = learning_rate
        self.margin = margin

        self.image_1 = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.image_2 = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.labels = tf.placeholder(tf.float32, [None])  # 0 for negative, 1 for positive

        self.cifar_encoder = cifar10_encoder()
        self.encoding_1 = self.cifar_encoder(self.image_1)
        self.encoding_2 = self.cifar_encoder(self.image_2)

        #         self.encoding_1 = cnn_func(self.image_1, reuse=tf.AUTO_REUSE)
        #         self.encoding_2 = cnn_func(self.image_2, reuse=tf.AUTO_REUSE)

        with tf.variable_scope('training', reuse=tf.AUTO_REUSE) as scope:
            self.l2_distance_squared = tf.square(
                tf.norm(tf.reshape(self.encoding_1 - self.encoding_2, (tf.shape(self.labels)[0], -1)), axis=-1))
            self.positives_loss = tf.reduce_mean(self.labels * self.l2_distance_squared)
            self.negatives_loss = tf.reduce_mean(
                (1 - self.labels) * tf.maximum(0., margin ** 2 - self.l2_distance_squared))
            self.loss = self.positives_loss + self.negatives_loss
            self.update_op = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(self.loss)

    def init_sess(self):
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True  # may need if using GPU
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__()  # equivalent to `with self.sess:`
        tf.global_variables_initializer().run()  # pylint: disable=E1101

    def encode(self, x):
        encoding = self.sess.run(self.encoding_1, feed_dict={self.image_1: x})
        return encoding

    def load_weights(self, weights_path):
        self.cifar_encoder.load_weights(weights_path, by_name=True)



def cifar10_encoder(encode_dim=256):
    model = Sequential()
#     model.add(InputLayer(input_tensor=input_placeholder,
#                      input_shape=(32, 32, 3)))

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
    model.add(Dense(encode_dim, name='dense_encode'))
    model.add(Activation('linear', name='encoding'))

    return model

