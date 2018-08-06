import tensorflow as tf
from learning import MatrixCapsNet
from input.SmallNorb import SmallNorb
import config


class MatrixCapsNetEstimator:


    @staticmethod
    def spread_loss(
            correct_one_hot,  # [batch, class]
            predicted_one_hot,   # [batch, class]
            margin # float
    ):
        class_axis = 1

        activation_target_class = tf.reduce_sum(correct_one_hot * predicted_one_hot, axis=class_axis, keepdims=True)

        difference_loss_per_class = activation_target_class - predicted_one_hot

        margin_loss_per_class = -difference_loss_per_class + margin

        incorrect_class = -correct_one_hot + 1.0

        spread_loss_per_class = tf.square(tf.maximum(tf.constant(0.0), margin_loss_per_class))
        spread_loss = tf.reduce_sum(spread_loss_per_class * incorrect_class)  # everything should be summed, but only for incorrect classes

        return spread_loss

    @staticmethod
    def spread_loss_adapter(
            correct_one_hot,  # [batch, class]
            predicted_one_hot,  # [batch, class]
            params  # dictionary
    ):
        return MatrixCapsNetEstimator.spread_loss(correct_one_hot, predicted_one_hot, params["margin"])

    def init(self, loss_fn=spread_loss_adapter.__func__, architecture="build_default_architecture"):
        self.loss_fn = loss_fn
        self.architecture = architecture
        return self

    @staticmethod
    def counter(next_number, is_training):
        number_counter = tf.Variable(tf.constant(float(0.0)), trainable=False, dtype=tf.float32)

        return tf.cond(is_training,
                lambda: number_counter.assign(number_counter + next_number, use_locking=True),
                lambda: number_counter)


    @staticmethod
    def spread_loss_margin(processed_example_counter):
        return 0.2 + .79 * tf.sigmoid(tf.minimum(10.0, (processed_example_counter / 64.0) / 50000.0 - 4.0))

    def model_function(self, examples, labels, mode, params):

        total_example_count = params["total_example_count"]
        iteration_count = params["iteration_count"]
        label_count = params["label_count"]
        batch_size = tf.cast(tf.gather(tf.shape(examples), 0), tf.float32)

        is_training = tf.constant(mode == tf.estimator.ModeKeys.TRAIN)
        routing_state = None

        network_output, reset_routing_configuration_op = \
            getattr(MatrixCapsNet(), self.architecture)(examples, iteration_count, routing_state)

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

        processed_example_counter = MatrixCapsNetEstimator.counter(batch_size, is_training)

        spread_loss_margin = MatrixCapsNetEstimator.spread_loss_margin(processed_example_counter)

        loss = self.loss_fn(tf.one_hot(labels, label_count), tf.reshape(activations, [-1, label_count]), {
            "margin": spread_loss_margin})

        # Compute evaluation metrics.
        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=predicted_classes,
                                       name='acc_op')

        metrics = {'accuracy': accuracy}
        # tf.summary.scalar('accuracy', accuracy[1])

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

    def train_and_test(self, small_norb, batch_size=30, epoch_count=10, max_steps=None, model_path=config.TF_MODEL_PATH):

        if max_steps is None:
            batch_count_per_epoch = small_norb.training_example_count() / batch_size
            max_steps = batch_count_per_epoch * epoch_count

        estimator = self.create_estimator(small_norb, model_path, epoch_count)

        train_fn = lambda: tf.data.Dataset.from_tensor_slices(small_norb.default_training_set())\
            .shuffle(100000)\
            .repeat(epoch_count)\
            .batch(batch_size)
        validation_fn = lambda: tf.data.Dataset.from_tensor_slices(small_norb.default_validation_set()).batch(batch_size)
        test_fn = lambda: tf.data.Dataset.from_tensor_slices(small_norb.default_test_set()).batch(batch_size)

        train_spec = tf.estimator.TrainSpec(input_fn=train_fn, max_steps=max_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=validation_fn)

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        print("training complete")
        validation_result = estimator.evaluate(input_fn=validation_fn)
        print("validation results computed")

        test_result = estimator.evaluate(input_fn=test_fn)
        print("test results computed")

        test_predictions = estimator.predict(input_fn=test_fn)
        print("test predictions computed")

        return test_result, validation_result, test_predictions

    def create_estimator(self, small_norb, model_path, epoch_count=1.0):
        total_example_count = small_norb.training_example_count() * epoch_count

        run_config = tf.estimator.RunConfig(
            # msave_checkpoints_secs=60 * 60,  # Save checkpoints every hour minutes.
            keep_checkpoint_max=20  # Retain the 20 most recent checkpoints.
        )

        estimator = tf.estimator.Estimator(
            lambda features, labels, mode, params: self.model_function(features, labels, mode, params),
            params={
                'total_example_count': total_example_count,
                'iteration_count': 3,
                'label_count': small_norb.label_count()
            },
            model_dir=model_path,
            config=run_config
        )

        return estimator

    def train(self, small_norb, model_path, batch_size, epoch_count, max_steps):
        if max_steps is None:
            batch_count_per_epoch = small_norb.training_example_count() / batch_size
            max_steps = batch_count_per_epoch * epoch_count

        estimator = self.create_estimator(small_norb, model_path, epoch_count)

        train_fn = lambda: tf.data.Dataset.from_tensor_slices(small_norb.default_training_set())\
            .shuffle(100000)\
            .repeat(epoch_count)\
            .batch(batch_size)
        validation_fn = lambda: tf.data.Dataset.from_tensor_slices(small_norb.default_validation_set()).batch(batch_size)

        train_spec = tf.estimator.TrainSpec(input_fn=train_fn, max_steps=max_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=validation_fn)

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        print("training complete")


    def validate(self, small_norb, model_path, batch_size=30):

        validation_fn = lambda: tf.data.Dataset.from_tensor_slices(small_norb.default_validation_set()).batch(batch_size)

        estimator = self.create_estimator(small_norb, model_path)

        validation_result = estimator.evaluate(input_fn=validation_fn)
        print("validation results computed")

        validation_predictions = estimator.predict(input_fn=validation_fn)
        print("validation predictions computed")

        return validation_result, validation_predictions

    def test(self, small_norb, model_path, batch_size=30):

        test_fn = lambda: tf.data.Dataset.from_tensor_slices(small_norb.default_test_set()).batch(batch_size)

        estimator = self.create_estimator(small_norb, model_path)

        test_result = estimator.evaluate(input_fn=test_fn)
        print("test results computed")

        test_predictions = estimator.predict(input_fn=test_fn)
        print("test predictions computed")

        return test_result, test_predictions
