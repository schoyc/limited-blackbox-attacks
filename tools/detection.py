import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, InputLayer
from collections import OrderedDict

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
        raise NotImplementedError("Must implement your own encode function!")

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

class L2Detector(Detector):
    def _init_encoder(self, weights_path):
        self.encode = lambda x : x.flatten()

class SimilarityDetector(Detector):
    def _init_encoder(self, weights_path):
        encoder = cifar10_encoder()
        encoder.load_weights(weights_path, by_name=True)
        self.encoder = encoder
        self.encode = lambda x : encoder.predict(np.expand_dims(x, axis=0))

class ExperimentDetectors():
    def __init__(self):
        detectors = [
            ("similarity", SimilarityDetector(threshold=1.44, K=50, weights_path="./encoders/encoder_all.h5")),
            ("l2", L2Detector(threshold=5.069, K=50))
        ]

        self.detectors = OrderedDict({})
        for d_name, detector in detectors:
            self.detectors[d_name] = detector

    def process(self, queries, num_queries_so_far):
        for _, detector in self.detectors.items():
            detector.process(queries, num_queries_so_far)

    def process_query(self, query, num_queries_so_far):
        for _, detector in self.detectors.items():
            detector.process_query(query, num_queries_so_far)



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

