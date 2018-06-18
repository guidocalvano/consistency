import tensorflow as tf
from learning import MatrixCapsNet
from input.SmallNorb import SmallNorb
import config


class MatrixCapsNetEstimator:
    def init(self):
        return self

    def default_model_function(self, examples, labels, mode, params):

        total_example_count = params["total_example_count"]
        iteration_count = params["iteration_count"]
        label_count = params["label_count"]

        is_training = tf.constant(mode==tf.estimator.ModeKeys.TRAIN)
        routing_state = None

        network_output, spread_loss_margin, reset_routing_configuration_op = \
            MatrixCapsNet().build_default_architecture(examples, total_example_count, iteration_count, routing_state, is_training)

        (activations, poses) = network_output

        predicted_classes = tf.reshape(tf.argmax(activations, 1), [-1])
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': predicted_classes[:, tf.newaxis],
                'probabilities': tf.nn.softmax(activations),
                'activations': activations,
                'poses': poses
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        loss = self.spread_loss(tf.one_hot(labels, label_count), tf.reshape(activations, [-1, label_count]), spread_loss_margin)

        # Compute evaluation metrics.
        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=predicted_classes,
                                       name='acc_op')

        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        # Create training op.
        assert mode == tf.estimator.ModeKeys.TRAIN

        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        class TrainingHooks(tf.train.SessionRunHook):

            def __init__(self, reset_routing_configuration_op):
                self.reset_routing_configuration_op = reset_routing_configuration_op

            def after_create_session(self, session, coord):
                self.reset_routing_configuration_op.eval(session)

        training_hooks = TrainingHooks(reset_routing_configuration_op)

        #@TODO figure out how reinitialization should work
        train_spec = tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op # ,
            # training_hooks=[training_hooks]
        )

        return train_spec


    def spread_loss(
            self,
            correct_one_hot,  # [batch, class]
            predicted_one_hot,   # [batch, class]
            margin: float  # float
    ):

        difference_loss_per_class = correct_one_hot - predicted_one_hot

        margin_loss_per_class = -difference_loss_per_class + margin  # order switched to make operator overloading work correctly

        class_axis = 1
        spread_loss_per_class = tf.square(tf.maximum(tf.constant(0.0), margin_loss_per_class))
        spread_loss = tf.reduce_sum(spread_loss_per_class) # everything should be summed

        return spread_loss

    def run_default(self, small_norb, batch_size=50, epoch_count=50, max_steps=2500):

        total_example_count = small_norb.training_example_count() * epoch_count

        train_fn = lambda: tf.data.Dataset.from_tensor_slices(small_norb.default_training_set())\
            .shuffle(100000)\
            .repeat(epoch_count)\
            .batch(batch_size)
        validation_fn = lambda: tf.data.Dataset.from_tensor_slices(small_norb.default_validation_set()).batch(batch_size)
        test_fn = lambda: tf.data.Dataset.from_tensor_slices(small_norb.default_test_set()).batch(batch_size)

        estimator = tf.estimator.Estimator(
            lambda features, labels, mode, params: self.default_model_function(features, labels, mode, params),
            params={
                'total_example_count': total_example_count,
                'iteration_count': 3,
                'label_count': small_norb.label_count()
            },
            model_dir=config.TF_MODEL_PATH)

        train_spec = tf.estimator.TrainSpec(input_fn=train_fn, max_steps=max_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=validation_fn)

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

        training_result = estimator.evaluate(input_fn=train_fn)

        validation_result = estimator.evaluate(input_fn=validation_fn)

        test_result = estimator.evaluate(input_fn=test_fn)

        test_predictions = estimator.predict(input_fn=test_fn)

        return test_result, validation_result, training_result, test_predictions

    def evaluate(self, small_norb, batch_size=50):
        total_example_count = small_norb.training_example_count()

        train_fn = lambda: tf.data.Dataset.from_tensor_slices(small_norb.default_training_set())\
            .shuffle(100000)\
            .batch(batch_size)
        validation_fn = lambda: tf.data.Dataset.from_tensor_slices(small_norb.default_validation_set()).batch(batch_size)
        test_fn = lambda: tf.data.Dataset.from_tensor_slices(small_norb.default_test_set()).batch(batch_size)

        estimator = tf.estimator.Estimator(
            lambda features, labels, mode, params: self.default_model_function(features, labels, mode, params),
            params={
                'total_example_count': total_example_count,
                'iteration_count': 3,
                'label_count': small_norb.label_count()
            },
            model_dir=config.TF_MODEL_PATH)

        training_result = estimator.evaluate(input_fn=train_fn)

        validation_result = estimator.evaluate(input_fn=validation_fn)

        test_result = estimator.evaluate(input_fn=test_fn)

        test_predictions = estimator.predict(input_fn=test_fn)

        return test_result, validation_result, training_result, test_predictions
