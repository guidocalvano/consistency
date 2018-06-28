from input.SmallNorb import SmallNorb
from learning.MatrixCapsNetEstimator import MatrixCapsNetEstimator
import dill as pickle
import config
import numpy as np
import tensorflow as tf
import os
import shutil


def profile():
    if os.path.isdir(config.TF_TRASH_PATH):
        shutil.rmtree(config.TF_TRASH_PATH)

    os.makedirs(config.TF_TRASH_PATH)

    sn = SmallNorb.from_cache()

    print("loaded data")

    mcne = MatrixCapsNetEstimator().init()

    batch_size = 4
    epoch_count = 1
    max_steps = 3

    def create_small_input_fn(fn):
        def input_fn():
            set = fn()
            return tf.data.Dataset.from_tensor_slices((tf.gather(set[0], list(range(batch_size))), tf.gather(set[1], list(range(batch_size))))).batch(batch_size)
        return input_fn

    estimator = mcne.create_estimator(sn, config.TF_DEBUG_MODEL_PATH, epoch_count)
    train_fn = create_small_input_fn(sn.default_training_set)
    validation_fn = create_small_input_fn(sn.default_validation_set)
    test_fn = create_small_input_fn(sn.default_test_set)

    print("loaded data")

    training_profiler = tf.train.ProfilerHook(
        save_steps=1,
        save_secs=None,
        output_dir=config.TF_DEBUG_TRAINING_TIMELINE_FILE_PATH,
        show_dataflow=True,
        show_memory=True
    )

    eval_profiler = tf.train.ProfilerHook(
        save_steps=1,
        save_secs=None,
        output_dir=config.TF_DEBUG_EVAL_TIMELINE_FILE_PATH,
        show_dataflow=True,
        show_memory=True
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_fn,
        max_steps=max_steps,
        hooks=[training_profiler]
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=validation_fn,
        hooks=[eval_profiler]
    )
    print("training estimator")
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    print("estimator trained")


if __name__ == "__main__":
    profile()