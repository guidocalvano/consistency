from input.SmallNorb import SmallNorb
from learning.MatrixCapsNetEstimator import MatrixCapsNetEstimator
import dill as pickle
import config
import numpy as np
import tensorflow as tf
import os
import shutil
import time

def run():
    if os.path.isdir(config.TF_TRASH_PATH):
        shutil.rmtree(config.TF_TRASH_PATH)

    os.makedirs(config.TF_TRASH_PATH)
    tf.logging.set_verbosity(tf.logging.INFO)

    sn = SmallNorb.from_cache()

    print("loaded data")

    mcne = MatrixCapsNetEstimator().init()

    batch_size = 25
    epoch_count = 1
    max_steps = 150

    def create_input_fn(fn):
        def input_fn():
            set = fn()
            return tf.data.Dataset.from_tensor_slices(set).batch(batch_size)
        return input_fn

    def create_reduced_input_fn(fn):
        def input_fn():
            set = fn()
            return tf.data.Dataset.from_tensor_slices(set).\
                shuffle(10000, reshuffle_each_iteration=True).\
                take(tf.cast(tf.shape(set[0])[0] / 7, dtype=tf.int64)).\
                batch(batch_size)
        return input_fn

    estimator = mcne.create_estimator(sn, config.TF_DEBUG_MODEL_PATH, epoch_count)
    train_fn = create_input_fn(sn.default_training_set)
    validation_fn = create_reduced_input_fn(sn.default_validation_set)
    test_fn = create_input_fn(sn.default_test_set)

    print("loaded data")

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_fn,
        max_steps=max_steps
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=validation_fn,
        start_delay_secs=1*60,  # evaluate every 20 minutes on a random third of the evaluation set. Evaluation takes about 5 minutes
        steps=1,  # use throttle and start delay instead
        throttle_secs=7*60  # evaluate every 20 minutes on a random third of the evaluation set

    )
    print("training estimator")
    t0 = time.time()
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    t1 = time.time()
    print(t1 - t0)
    print("estimator trained")

    print("testing estimator")
    t0 = time.time()
    estimator.evaluate(test_fn)
    t1 = time.time()
    print(t1 - t0)
    print("testing evaluated")


if __name__ == "__main__":
    run()