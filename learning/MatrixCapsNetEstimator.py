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

    def init(self,
             loss_fn=spread_loss_adapter.__func__,
             architecture="build_default_architecture",
             initialization=None,
             regularization=None,
             save_summary_steps=500,
             eval_steps=100
        ):
        self.loss_fn = loss_fn
        self.architecture = architecture
        self.inialization = initialization
        self.regularization = regularization

        self.save_summary_steps = save_summary_steps
        self.eval_steps = eval_steps

        return self

    @staticmethod
    def counter(next_number, is_training):

        number_counter = tf.get_variable('counter', initializer=tf.constant(float(0.0)), trainable=False, dtype=tf.float32)
        return tf.cond(is_training,
                lambda: number_counter.assign_add(next_number),
                lambda: number_counter)


    @staticmethod
    def spread_loss_margin(processed_example_counter):
        return 0.2 + .79 * tf.sigmoid(tf.minimum(10.0, (processed_example_counter / 64.0) / 50000.0 - 4.0))

    def model_function(self, examples, labels, mode, params):

        with tf.device('/cpu:0'):

            total_example_count = params["total_example_count"]
            iteration_count = params["iteration_count"]
            label_count = params["label_count"]
            batch_size = tf.cast(tf.gather(tf.shape(examples), 0), tf.float32)

            is_training = tf.constant(mode == tf.estimator.ModeKeys.TRAIN)

            processed_example_counter = MatrixCapsNetEstimator.counter(batch_size, is_training)

            optimizer = tf.train.AdamOptimizer()

            #@TODO: parallelize using: https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L165
            train_op, loss, predicted_classes, activations, poses = self.make_parallel(examples, labels, optimizer, label_count, iteration_count, processed_example_counter,
                        tf.train.get_global_step())

            tf.summary.histogram('predicted_classes', predicted_classes)

            # Compute evaluation metrics.
            accuracy = tf.metrics.accuracy(labels=labels,
                                           predictions=predicted_classes,
                                           name='acc_op')

            metrics = {'accuracy': accuracy}
            # tf.summary.scalar('accuracy', accuracy[1])

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'class_ids': predicted_classes[:, tf.newaxis],
                    'probabilities': tf.nn.softmax(activations),
                    'activations': activations,
                    'poses': poses
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)



            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics)

            # Create training op.
            assert mode == tf.estimator.ModeKeys.TRAIN


            # grads = optimizer.compute_gradients(loss)
            # # weights_by_layer = tf.get_collection('weights')
            #
            # for index, grad in enumerate(grads):
            #     tf.summary.histogram("{}-grad".format(grads[index][1].name), grads[index])

            # optimizer=tf.contrib.estimator.TowerOptimizer(optimizer)


            # for layer_id, weights in enumerate(weights_by_layer):
            #     tf.summary.histogram('weights_at_layer' + str(layer_id), tf.gradients(loss, weights))


            # class TrainingHooks(tf.train.SessionRunHook):
            #
            #     def __init__(self, reset_routing_configuration_op):
            #         self.reset_routing_configuration_op = reset_routing_configuration_op
            #
            #     def after_create_session(self, session, coord):
            #         self.reset_routing_configuration_op.eval(session)
            #
            # training_hooks = TrainingHooks(reset_routing_configuration_op)

            #@TODO figure out how reinitialization should work
            train_spec = tf.estimator.EstimatorSpec(
                mode,
                loss=loss,
                train_op=train_op # ,
                # training_hooks=[training_hooks]
            )

            return train_spec

    def make_parallel(self, examples, labels, optimizer, label_count, iteration_count, processed_example_counter, global_step):

        examples_sub_batches = tf.split(examples, config.GPU_COUNT, 0)
        labels_sub_batches = tf.split(labels, config.GPU_COUNT, 0)

        tower_grads = []
        loss_sub_batches = []
        predicted_classes_sub_batches = []
        activations_sub_batches = []
        poses_sub_batches = []

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(config.GPU_COUNT):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constructs the entire CIFAR model but shares the variables across
                        # all towers.
                        loss_sub_batch, predicted_classes_sub_batch, activations_sub_batch, poses_sub_batch = \
                            self.network_output(examples_sub_batches[i], labels_sub_batches[i], label_count, iteration_count, processed_example_counter)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = optimizer.compute_gradients(loss_sub_batch)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

                        loss_sub_batches.append(loss_sub_batch)
                        predicted_classes_sub_batches.append(predicted_classes_sub_batch)
                        activations_sub_batches.append(activations_sub_batch)
                        poses_sub_batches.append(poses_sub_batch)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = self.average_gradients(tower_grads)

        # Apply the gradients to adjust the shared variables.
        train_op = optimizer.apply_gradients(grads, global_step=global_step)

        loss = tf.reduce_mean(tf.stack(loss_sub_batches, 0))
        predicted_classes = tf.concat(predicted_classes_sub_batches, 0)
        activations = tf.concat(activations_sub_batches, 0)
        poses = tf.concat(poses_sub_batches, 0)

        return train_op, loss, predicted_classes, activations, poses

    def network_output(self, examples, labels, label_count, iteration_count, processed_example_counter):

        routing_state = None

        mcn = MatrixCapsNet()

        if self.inialization:
            mcn.set_init_options(self.inialization)

        if self.regularization:
            mcn.set_regularization(self.regularization)

        # GET ARCHITECTURE FROM MATRIX CAPSNET
        network_output, reset_routing_configuration_op, regularization_loss = \
            getattr(mcn, self.architecture)(examples, iteration_count, routing_state)

        (activations, poses) = network_output

        predicted_classes = tf.reshape(tf.argmax(activations, 1), [-1])

        spread_loss_margin = MatrixCapsNetEstimator.spread_loss_margin(processed_example_counter)

        tf.summary.scalar('spread_loss_margin', spread_loss_margin)

        loss = self.loss_fn(tf.one_hot(labels, label_count), tf.reshape(activations, [-1, label_count]), {
            "margin": spread_loss_margin})

        if regularization_loss is not None: loss = loss + regularization_loss * self.regularization['axial']

        return loss, predicted_classes, activations, poses



    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def train_and_test(self, small_norb, batch_size=64, epoch_count=600, max_steps=None, save_summary_steps=500, eval_steps=100, model_path=config.TF_MODEL_PATH):

        if max_steps is None:
            batch_count_per_epoch = small_norb.training_example_count() / batch_size
            max_steps = batch_count_per_epoch * epoch_count

        estimator = self.create_estimator(small_norb, model_path, epoch_count, save_summary_steps=self.save_summary_steps)

        train_fn = lambda: tf.data.Dataset.from_tensor_slices(small_norb.default_training_set())\
            .shuffle(100000)\
            .repeat(epoch_count)\
            .batch(batch_size)
        validation_fn = lambda: tf.data.Dataset.from_tensor_slices(small_norb.default_validation_set()).batch(batch_size)
        test_fn = lambda: tf.data.Dataset.from_tensor_slices(small_norb.default_test_set()).batch(batch_size)

        train_spec = tf.estimator.TrainSpec(input_fn=train_fn, max_steps=max_steps)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=validation_fn,
            steps=self.eval_steps,
            throttle_secs=1200
        )

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        print("training complete")
        validation_result = estimator.evaluate(input_fn=validation_fn)
        print("validation results computed")

        test_result = estimator.evaluate(input_fn=test_fn)
        print("test results computed")

        test_predictions = list(estimator.predict(input_fn=test_fn))
        print("test predictions computed")

        return test_result, validation_result, test_predictions

    def create_estimator(self, small_norb, model_path, epoch_count=1.0, save_summary_steps=500):
        total_example_count = small_norb.training_example_count() * epoch_count

        run_config = tf.estimator.RunConfig(
            # msave_checkpoints_secs=60 * 60,  # Save checkpoints every hour minutes.
            keep_checkpoint_max=10,  # Retain the 10 most recent checkpoints.
            save_summary_steps=self.save_summary_steps,  # default is 100, but we even compute gradients for the summary, so maybe not wise to do this step too often
            session_config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True
            )
        )

        estimator = tf.estimator.Estimator(
            lambda features, labels, mode, params: self.model_function(features, labels, mode, params),
            params={
                'total_example_count': total_example_count,
                'iteration_count': 3,
                'label_count': small_norb.label_count()
            },
            model_dir=model_path,
            config=run_config,
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
