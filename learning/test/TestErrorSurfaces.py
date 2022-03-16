import tensorflow as tf
import numpy as np


class TestErrorSurfaces:

    def __init__(self):
        pass

    def make_net(self, input, weights1, weights2):

        layer1 = tf.max(tf.tensordot(input, weights1, [1, 1]), 0)

        layer2 = tf.max(tf.tensordot(layer1, weights2, [1, 1]), 0)

        return layer2

    def generate_dataset_with_local_minima(self):

        pass

    def train_optimizer(self, batch_size, optimizer, error_surface):

        pass