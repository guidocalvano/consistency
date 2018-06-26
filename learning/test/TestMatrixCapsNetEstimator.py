import tensorflow as tf
from learning.MatrixCapsNetEstimator import MatrixCapsNetEstimator
from input.SmallNorb import SmallNorb
import numpy as np
import config
import os
import shutil


class TestMatrixCapsNetEstimator(tf.test.TestCase):

    def setUp(self):
        self.tearDown()  # in case a test did not shut down correctly
        os.makedirs(config.TF_DEBUG_MODEL_PATH)

    def tearDown(self):
        if os.path.isdir(config.TF_DEBUG_MODEL_PATH):
            shutil.rmtree(config.TF_DEBUG_MODEL_PATH)

    def test_run_default(self):
        sn = SmallNorb.from_cache()
        sn.limit_by_split(30)
        print("data loaded")
        mcne = MatrixCapsNetEstimator().init()

        first_results = mcne.train_and_test(sn, batch_size=3, epoch_count=1, max_steps=1, model_path=config.TF_DEBUG_MODEL_PATH)
        test_result, validation_result, test_predictions = first_results

        self.assertTrue(np.isfinite(test_result['accuracy']), "test accuracy must be finite")
        self.assertTrue(np.isfinite(test_result['loss']), "test loss must be finite")
        self.assertTrue(np.isfinite(test_result['global_step']), "test global_step must be finite")

        self.assertTrue(np.isfinite(validation_result['accuracy']), "validation accuracy must be finite")
        self.assertTrue(np.isfinite(validation_result['loss']), "validation loss must be finite")
        self.assertTrue(np.isfinite(validation_result['global_step']), "validation global_step must be finite")

        tp = list(test_predictions)

        self.assertTrue(len(tp) == 30, 'there must be predictions for every test set element')

        for p in tp:
            self.assertFiniteAndShape(p['class_ids'], [1], "class ids must be finite and of correct shape")
            self.assertFiniteAndShape(p['probabilities'], [5, 1, 1], "probabilities must be finite and of correct shape")
            self.assertFiniteAndShape(p['activations'], [5, 1, 1], "activations must be finite and of correct shape")
            self.assertFiniteAndShape(p['poses'], [5, 1, 4, 4], "poses must be finite and of correct shape")


        print("first test completed")
        # assert that first result have correct shape and do not contain nans

        second_results = mcne.train_and_test(sn, batch_size=3, epoch_count=1, max_steps=5, model_path=config.TF_DEBUG_MODEL_PATH)

        test_result, validation_result, test_predictions = second_results


        self.assertTrue(np.isfinite(test_result['accuracy']), "test accuracy must be finite")
        self.assertTrue(np.isfinite(test_result['loss']), "test loss must be finite")
        self.assertTrue(np.isfinite(test_result['global_step']), "test global_step must be finite")

        self.assertTrue(np.isfinite(validation_result['accuracy']), "validation accuracy must be finite")
        self.assertTrue(np.isfinite(validation_result['loss']), "validation loss must be finite")
        self.assertTrue(np.isfinite(validation_result['global_step']), "validation global_step must be finite")

        tp = list(test_predictions)

        self.assertTrue(len(tp) == 30, 'there must be predictions for every test set element')

        for p in tp:
            self.assertFiniteAndShape(p['class_ids'], [1], "class ids must be finite and of correct shape")
            self.assertFiniteAndShape(p['probabilities'], [5, 1, 1], "probabilities must be finite and of correct shape")
            self.assertFiniteAndShape(p['activations'], [5, 1, 1], "activations must be finite and of correct shape")
            self.assertFiniteAndShape(p['poses'], [5, 1, 4, 4], "poses must be finite and of correct shape")

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


    # def test_model_fn_training(self):
    #
    #     mcne = MatrixCapsNetEstimator()
    #
    #     examples = tf.constant(np.random.normal(np.zeros([1, 32, 32, 1])), tf.float32)
    #     labels = tf.constant(np.ones([1]), tf.int32)
    #     mode = tf.estimator.ModeKeys.TRAIN
    #
    #     params = {
    #         'total_example_count': 1,
    #         'iteration_count': 3,
    #         'label_count': 5
    #     }
    #
    #     training_specs = mcne.default_model_function(examples, labels, mode, params)
    #
    #     sess = tf.Session()
    #     with sess.as_default():
    #         sess.run(tf.global_variables_initializer())
    #
    #         weights = sess.run([tf.trainable_variables()])
    #
    #         self.assertTrue(np.isfinite(weights))
    #
    #         sess.run([training_specs.train_op])
    #
    #     pass

    def assertFiniteAndShape(self, tensor_array, tensor_shape, message):
        self.assertTrue(np.isfinite(tensor_array).all(), message + ": does not have finite data")
        self.assertTrue((np.array(tensor_array.shape) ==
                        np.array(tensor_shape)).all(), message + ": does not have correct shape")