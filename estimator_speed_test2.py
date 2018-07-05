from input.SmallNorb import SmallNorb
from learning.MatrixCapsNetEstimator import MatrixCapsNetEstimator
import dill as pickle
import config
import numpy as np
import tensorflow as tf
import os
import shutil
import time


def speed_test():

    tf.logging.set_verbosity(tf.logging.INFO)

    sn = SmallNorb.from_cache()

    print("loaded data")

    mcne = MatrixCapsNetEstimator().init()

    batch_size = 25
    epoch_count = 1
    max_steps = 100

    def create_small_input_fn(fn):
        def input_fn():
            set = fn()
            return tf.data.Dataset.from_tensor_slices(set).\
                shuffle(10000, reshuffle_each_iteration=True).\
                take(tf.cast(tf.shape(set[0])[0] / 3, dtype=tf.int64)).\
                batch(batch_size)
        return input_fn

    estimator = mcne.create_estimator(sn, config.TF_DEBUG_MODEL_PATH, epoch_count)
    validation_fn = create_small_input_fn(sn.default_validation_set)

    print("loaded data")

    print("evaluating estimator")
    t0 = time.time()
    estimator.evaluate(validation_fn)
    t1 = time.time()
    print(t1 - t0)
    print("estimator evaluated")


if __name__ == "__main__":
    speed_test()
