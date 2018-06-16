import tensorflow as tf
from learning.MatrixCapsNetEstimator import MatrixCapsNetEstimator
from input.SmallNorb import SmallNorb
import numpy as np


class TestMatrixCapsNetEstimator(tf.test.TestCase):

    def test_run_default(self):
        sn = SmallNorb.from_cache()

        sn.training = sn.filter_data_on_row_ids(sn.training, np.arange(10))
        sn.validation = sn.filter_data_on_row_ids(sn.validation, np.arange(10))
        sn.test = sn.filter_data_on_row_ids(sn.test, np.arange(10))

        mcne = MatrixCapsNetEstimator().init()

        first_results = mcne.run_default(sn, 5, 1, 1)
        test_result, validation_result, training_result, test_predictions = first_results

        # assert that first result have correct shape and do not contain nans

        second_results = mcne.run_default(sn, 50, 1, 1)

        # assert correct shape and no nans

        # assert improvement on training set

        self.assertTrue(False, "assertions still missing")

    def test_spread_loss(self):

        mcne = MatrixCapsNetEstimator()

        batch_size = 6
        class_count = 5

        mocked_correct = tf.constant(np.random.random([batch_size, class_count]), dtype=tf.float32)
        mocked_predict = tf.Variable(np.random.random([batch_size, class_count]), dtype=tf.float32)
        mocked_margin  = tf.constant(.5, dtype=tf.float32)

        sl = mcne.spread_loss(mocked_correct, mocked_predict, mocked_margin)

        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(sl, global_step=tf.train.get_global_step())

        sess = tf.Session()
        with sess.as_default():
            sess.run(tf.global_variables_initializer())

            sl_output = sl.eval()

            self.assertFiniteAndShape(sl_output, [], "spread loss output must be finite and contain a single value")

            sess.run([train_op])
            sl_output = sl.eval()

            self.assertFiniteAndShape(sl_output, [], "spread loss output must be finite and contain a single value after training")


    def test_model_fn_training(self):

        mcne = MatrixCapsNetEstimator()

        examples = tf.constant(np.random.normal(np.zeros([1, 32, 32, 1])), tf.float32)
        labels = tf.constant(np.ones([1]), tf.int32)
        mode = tf.estimator.ModeKeys.TRAIN

        params = {
            'total_example_count': 1,
            'iteration_count': 3,
            'label_count': 5
        }

        training_specs = mcne.default_model_function(examples, labels, mode, params)

        sess = tf.Session()
        with sess.as_default():
            sess.run(tf.global_variables_initializer())

            weights = sess.run([tf.trainable_variables()])

            self.assertTrue(np.isfinite(weights))

            sess.run([training_specs.train_op])

        pass

    def assertFiniteAndShape(self, tensor_array, tensor_shape, message):
        self.assertTrue(np.isfinite(tensor_array).all(), message + ": does not have finite data")
        self.assertTrue((np.array(tensor_array.shape) ==
                        np.array(tensor_shape)).all(), message + ": does not have correct shape")