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
    if os.path.isdir(config.TF_TRASH_PATH):
        shutil.rmtree(config.TF_TRASH_PATH)

    os.makedirs(config.TF_TRASH_PATH)

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
            return tf.data.Dataset.from_tensor_slices(set).batch(batch_size)
        return input_fn

    estimator = mcne.create_estimator(sn, config.TF_DEBUG_MODEL_PATH, epoch_count)
    train_fn = create_small_input_fn(sn.default_training_set)
    validation_fn = create_small_input_fn(sn.default_validation_set)
    test_fn = create_small_input_fn(sn.default_test_set)

    print("loaded data")

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_fn,
        max_steps=max_steps
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=validation_fn,
        start_delay_secs=250*60,  # do not evaluate to test only training time
        steps=1,  # use throttle and start delay instead
        throttle_secs=250*60  # do not evaluate to test only training time

    )
    print("training estimator")
    t0 = time.time()
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    t1 = time.time()
    print(t1 - t0)
    print("estimator trained")


if __name__ == "__main__":
    speed_test()
