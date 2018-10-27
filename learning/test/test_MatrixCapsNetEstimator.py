import tensorflow as tf
from learning.MatrixCapsNetEstimator import MatrixCapsNetEstimator
from input.SmallNorb import SmallNorb
import numpy as np
import config
import os
import shutil
import sys


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
            self.assertFiniteAndShape(p['probabilities'], [5, 1], "probabilities must be finite and of correct shape")
            self.assertFiniteAndShape(p['activations'], [5, 1], "activations must be finite and of correct shape")
            self.assertFiniteAndShape(p['poses'], [5, 4, 4], "poses must be finite and of correct shape")


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
            self.assertFiniteAndShape(p['probabilities'], [5, 1], "probabilities must be finite and of correct shape")
            self.assertFiniteAndShape(p['activations'], [5, 1], "activations must be finite and of correct shape")
            self.assertFiniteAndShape(p['poses'], [5, 4, 4], "poses must be finite and of correct shape")

    def test_train_spread_loss(self):

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

    def test_train_two_examples_quick(self):

        sn = SmallNorb.from_cache()
        sn.reduce_to_two_examples(.3, .3)

        mcne = MatrixCapsNetEstimator().init(architecture="build_simple_architecture")

        first_session_max_steps = 1
        batch_size = 17
        epoch_count = 2

        mcne.train(sn, config.TF_DEBUG_MODEL_PATH, batch_size, epoch_count, first_session_max_steps)

        untrained_performance = mcne.test(sn, config.TF_DEBUG_MODEL_PATH, batch_size)[0]["accuracy"]

        second_session_max_steps = 2

        mcne.train(sn, config.TF_DEBUG_MODEL_PATH, batch_size, epoch_count, second_session_max_steps)

        trained_performance = mcne.test(sn, config.TF_DEBUG_MODEL_PATH, batch_size)[0]["accuracy"]

        print("untrained performance" + str(untrained_performance))

        print("trained performance" + str(trained_performance))

        self.assertTrue(True, "This only tested if the code actually runs, run longer test to see if learning takes place")


    def test_train_two_examples(self):

        sn = SmallNorb.from_cache()
        sn.reduce_to_two_examples(.3, .3)

        mcne = MatrixCapsNetEstimator().init(architecture="build_simple_architecture")

        first_session_max_steps = 1
        batch_size = 17
        epoch_count = 50

        mcne.train(sn, config.TF_DEBUG_MODEL_PATH, batch_size, epoch_count, first_session_max_steps)

        untrained_performance = mcne.test(sn, config.TF_DEBUG_MODEL_PATH, batch_size)[0]["accuracy"]

        second_session_max_steps = 1000

        mcne.train(sn, config.TF_DEBUG_MODEL_PATH, batch_size, epoch_count, second_session_max_steps)

        trained_performance = mcne.test(sn, config.TF_DEBUG_MODEL_PATH, batch_size)[0]["accuracy"]

        print("untrained performance" + str(untrained_performance))

        print("trained performance" + str(trained_performance))

        self.assertTrue(trained_performance > untrained_performance, "performance must increase")


    def test_train_two_examples_without_spread_loss(self):

        sn = SmallNorb.from_cache()
        sn.reduce_to_two_examples(.3, .3)

        def mean_squared_loss_adapter(
                correct_one_hot,  # [batch, class]
                predicted_one_hot,  # [batch, class]
                params  # dictionary
        ):
            return tf.losses.mean_squared_error(correct_one_hot, predicted_one_hot)

        mcne = MatrixCapsNetEstimator().init(loss_fn=mean_squared_loss_adapter, architecture="build_simple_architecture")

        first_session_max_steps = 1
        batch_size = 17
        epoch_count = 50

        mcne.train(sn, config.TF_DEBUG_MODEL_PATH, batch_size, epoch_count, first_session_max_steps)

        untrained_performance = mcne.test(sn, config.TF_DEBUG_MODEL_PATH, batch_size)[0]["accuracy"]

        second_session_max_steps = 1000
        batch_size = 17

        mcne.train(sn, config.TF_DEBUG_MODEL_PATH, batch_size, epoch_count, second_session_max_steps)

        trained_performance = mcne.test(sn, config.TF_DEBUG_MODEL_PATH, batch_size)[0]["accuracy"]

        print("untrained performance" + str(untrained_performance))

        print("trained performance" + str(trained_performance))

        self.assertTrue(trained_performance > untrained_performance, "performance must increase")

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

    def test_spread_loss_margin(self):
        sess = tf.Session()

        example_counter = tf.placeholder(dtype=tf.float32)

        spread_margin = MatrixCapsNetEstimator.spread_loss_margin(example_counter)

        one_step_original_margin = 0.2 + .79 * tf.sigmoid(tf.minimum(10.0, 1.0 / 50000.0 - 4.0))
        three_step_original_margin = 0.2 + .79 * tf.sigmoid(tf.minimum(10.0, 3.0 / 50000.0 - 4.0))
        three_and_half_step_original_margin = 0.2 + .79 * tf.sigmoid(tf.minimum(10.0, 3.5 / 50000.0 - 4.0))


        with sess.as_default():

            actual, expected = sess.run([spread_margin, one_step_original_margin], {example_counter: 64})

            self.assertTrue(actual == expected, "margin should be the same for both paper and this implementation for one step in the paper")

            actual, expected = sess.run([spread_margin, three_step_original_margin], {example_counter: 3 * 64})

            self.assertTrue(actual == expected, "margin should be the same for both paper and this implementation for three steps in the paper")

            actual, expected = sess.run([spread_margin, three_and_half_step_original_margin], {example_counter: int(3.5 * 64)})

            self.assertTrue(actual == expected, "margin should be the same for both paper and this implementation for 3.5 steps in the paper")

    def test_spread_loss(self):
        sess = tf.Session()

        m = tf.placeholder(shape=[], dtype=tf.float32)
        correct = tf.placeholder(shape=[None, 2], dtype=tf.float32)
        predicted = tf.placeholder(shape=[None, 2], dtype=tf.float32)

        spread_loss = MatrixCapsNetEstimator.spread_loss(correct, predicted, m)

        correct_np = np.array([
            [0.0, 1.0]
        ])

        correct_but_low_margin_np = np.array([
            [.3, .5]
        ])

        correct_high_margin_np = np.array([
            [.2, .7]
        ])

        wrong_output_np = np.array([
            [.5, .3]
        ])

        with sess.as_default():
            no_loss = sess.run(spread_loss, {
                correct: correct_np,
                predicted: correct_np,
                m: .4
            })

            self.assertTrue(no_loss == 0.0, 'perfectly correct output should not incur any loss')

            low_margin_output_no_loss = sess.run(spread_loss, {
                correct: correct_np,
                predicted: correct_but_low_margin_np,
                m: .2
            })
            self.assertTrue(abs(low_margin_output_no_loss) < sys.float_info.epsilon * 100, 'low margin correct output should not incur any loss')

            low_margin_output_loss = sess.run(spread_loss, {
                correct: correct_np,
                predicted: correct_but_low_margin_np,
                m: .3
            })

            self.assertTrue(abs(low_margin_output_loss - 0.01) < .000001, 'high margin correct output should not incur any loss')


            high_output_no_loss = sess.run(spread_loss, {
                correct: correct_np,
                predicted: correct_high_margin_np,
                m: .4
            })
            self.assertTrue(abs(high_output_no_loss)  < sys.float_info.epsilon * 100, 'high margin correct output should not incur any loss')


            wrong_output_loss = sess.run(spread_loss, {
                correct: correct_np,
                predicted: wrong_output_np,
                m: .2
            })
            self.assertTrue(abs(wrong_output_loss - (.4 * .4)) < .00001, 'high margin correct output should not incur any loss')


    def test_counter(self):

        sess = tf.Session()

        zero_counter = MatrixCapsNetEstimator.counter(tf.constant(0.0), tf.constant(True))
        increase_one_counter = MatrixCapsNetEstimator.counter(tf.constant(1.0), tf.constant(True))

        with sess.as_default():

            sess.run(tf.global_variables_initializer())

            v = sess.run(zero_counter)

            self.assertTrue(v == 0.0, "counter should be zero")

            v = sess.run(increase_one_counter)

            self.assertTrue(v == 1.0, "counter should be one")

            v = sess.run(increase_one_counter)

            self.assertTrue(v == 2.0, "counter should be two")

            v = sess.run([increase_one_counter, increase_one_counter])[1]

            self.assertTrue(v == 3.0, "counter should be three")

    def assertFiniteAndShape(self, tensor_array, tensor_shape, message):
        self.assertTrue(np.isfinite(tensor_array).all(), message + ": does not have finite data")
        self.assertTrue((np.array(tensor_array.shape) ==
                        np.array(tensor_shape)).all(), message + ": does not have correct shape")