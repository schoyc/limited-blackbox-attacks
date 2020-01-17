import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, InputLayer
from collections import OrderedDict

class Detector(object):

    def __init__(self, threshold, K=50, size=None, chunk_size=1000, weights_path="./encoder_1.h5", ith_query=1):
        self.threshold = threshold
        self.K = K
        self.size = size
        self.num_queries = 0
        self.ith_query = ith_query

        self.buffer = []
        self.memory = []
        self.chunk_size = chunk_size

        self.history = [] # Tracks number of queries (t) when attack was detected
        self.history_by_attack = []
        self.detected_dists = [] # Tracks knn-dist that was detected

        self._init_encoder(weights_path)

    def _init_encoder(self, weights_path):
        raise NotImplementedError("Must implement your own encode function!")

    def process(self, queries, num_queries_so_far, encoded=False):
        if not encoded:
            queries = self.encode(queries)
        for query in queries:
            self.process_query(query, num_queries_so_far)
            num_queries_so_far += 1

    def process_query(self, query, num_queries_so_far):
        if self.num_queries % self.ith_query != 0:
            return

        if len(self.memory) == 0 and len(self.buffer) < self.K:
            self.buffer.append(query)
            self.num_queries += 1
            return

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
        self.encode = lambda x : x.reshape((x.shape[0], -1))

class SimilarityDetector(Detector):
    def _init_encoder(self, weights_path):
        encoder = cifar10_encoder()
        encoder.load_weights(weights_path, by_name=True)
        self.encoder = encoder
        self.encode = lambda x : encoder.predict(x)

class ExperimentDetectors():
    def __init__(self, active=True, detectors=None):
        self.active = active

        if detectors is None:
            detectors = [
                ("similarity", SimilarityDetector(threshold=1.44, K=50, weights_path="./encoders/encoder_all.h5")),
                ("l2", L2Detector(threshold=5.069, K=50)),
                ("sim-no-brightness", SimilarityDetector(threshold=1.56, K=50, weights_path="./encoders/encoder_no_brightness.h5")),
            ]

        self.detectors = OrderedDict({})
        for d_name, detector in detectors:
            self.detectors[d_name] = detector

    def process(self, queries, num_queries_so_far):
        if not self.active:
            return

        for _, detector in self.detectors.items():
            detector.process(queries, num_queries_so_far)

    def process_query(self, query, num_queries_so_far):
        if not self.active:
            return

        for _, detector in self.detectors.items():
            detector.process_query(query, num_queries_so_far)


class MultiAttackDetectors(ExperimentDetectors):
    def __init__(self, active=True, detectors=None):
        detectors = [
            # ("sim-k=50-i=1", SimilarityDetector(threshold=1.44, K=50, weights_path="./encoders/encoder_all.h5")),
            # ("sim-k=25-i=1", SimilarityDetector(threshold=1.26, K=25, weights_path="./encoders/encoder_all.h5")),
            # ("sim-k=10-i=1", SimilarityDetector(threshold=1.02, K=10, weights_path="./encoders/encoder_all.h5")),
            # ("sim-k=50-i=50", SimilarityDetector(threshold=1.44, K=50, weights_path="./encoders/encoder_all.h5", ith_query=50)),
            # ("sim-k=50-i=100", SimilarityDetector(threshold=1.44, K=50, weights_path="./encoders/encoder_all.h5", ith_query=100)),
            # ("sim-k=10-i=50", SimilarityDetector(threshold=1.02, K=10, weights_path="./encoders/encoder_all.h5", ith_query=50)),

            ("PRADA-k=50-t=093", PRADADetector(threshold=0.93))
        ]
        # self.encode_once = detectors[0].encode
        super.__init__(self, detectors=detectors)


    def process(self, queries, num_queries_so_far):
        # queries = self.encode_once(queries)
        # super.process(queries, num_queries_so_far)
        ExperimentDetectors.process(self, queries, num_queries_so_far)

from scipy.stats import shapiro
class PRADADetector():
    def __init__(self, threshold, model_path='./models/cifar10_ResNet20v1_model.h5', num_classes=10, min_D_size=50, clear_after_detection=True):
        self.model = load_model(model_path)
        self.num_classes = num_classes
        self.threshold = threshold
        self.clear_after_detection = clear_after_detection
        self.min_D_size = min_D_size

        self.detected_stats = []
        self.num_queries = 0
        self.history = []
        self.detected_dists = []
        self.W_s = []

        self._init_buffers()

    def _init_buffers(self):
        num_classes = self.num_classes
        # D
        self.D = []

        # G_c
        self.G_c = [[] for c in range(num_classes)]

        # D_G_c
        self.D_c = [[] for c in range(num_classes)]

        # T_c
        self.T_c = [0 for c in range(num_classes)]

        # W
        self.W = []

    def process(self, queries, num_queries_so_far, encoded=False, c_s=None):
        if c_s is None:
            c_s = self.model.predict(queries).argmax(axis=-1)

        for c, query in zip(c_s, queries):
            self.process_query(query, c, num_queries_so_far)
            num_queries_so_far += 1

    def process_query(self, query, c, num_queries_so_far):
        G_c, D_c, T_c = self.G_c[c], self.D_c[c], self.T_c[c]
        D = self.D
        if len(G_c) == 0:
            G_c.append(query)
            D_c.append(0.)
        else:
            d_min = np.min(np.linalg.norm(query - G_c))
            D.append(d_min)

            if d_min > T_c:
                G_c.append(query)
                D_c.append(d_min)
                D_c = np.array(D_c)
                T_c = max(T_c, D_c.mean() - D_c.std())
                self.T_c[c] = T_c

        self.num_queries += 1
        if len(D) >= self.min_D_size - 1:
            D = np.array(D)
            lower, upper = D.mean() - 3 * D.std(), D.mean() + 3 * D.std()
            D_ = [d for d in D if d >= lower and d <= upper]
            W, p = shapiro(D_)
            self.W.append(W)

            is_attack = W < self.threshold
            if is_attack:
                self.detected_dists.append(W)
                self.history.append(self.num_queries)
                print("[PRADA] Detected:", W)

                if self.clear_after_detection:
                    self.W_s.append(self.W)
                    self._init_buffers()

            if len(D) % 1000 == 0:
                print("[PRADA] Num. queries so far:", self.num_queries)

    def get_detections(self):
        history = self.history
        epochs = []
        for i in range(len(history) - 1):
            epochs.append(history[i + 1] - history[i])

        return epochs


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

