import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


class CapsNet:
    def init(self, width, height, color_count, capsule_count_layer_0, capsule_dimension_count_layer_0, capsule_count_layer_1, capsule_dimension_count_layer_1):

        self.correct_label = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

        [input_data, input_capsules] = self.input_capsule_layer(width, height, color_count,
                                                  capsule_count_layer_0, capsule_dimension_count_layer_0)

        self.input_data = input_data

        batch_size = tf.shape(self.input_data)[0]

        self.capsule_output = self.capsule_layer(batch_size, input_capsules, capsule_count_layer_0, capsule_dimension_count_layer_0,
                                                 capsule_count_layer_1, capsule_dimension_count_layer_1)

        self.predicted_label = self._prediction_layer(self.capsule_output)

        [reconstruction_targets, mask_with_labels] = self.reconstruction_targets(capsule_count_layer_1, capsule_dimension_count_layer_1, self.capsule_output, self.correct_label, self.predicted_label)

        self.mask_with_labels = mask_with_labels

        self.decoder_output = self.decoder(reconstruction_targets)

        reconstruction_loss = self._reconstruction_loss(input_data, 28 * 28, self.decoder_output)

        margin_loss = self._margin_loss(capsule_count_layer_1, self.capsule_output, self.correct_label)

        self.loss = self._loss(margin_loss, reconstruction_loss)

        optimizer = tf.train.AdamOptimizer()
        self.training_op = optimizer.minimize(self.loss, name="training_op")

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        return self

    def image_batch_placeholder(self, width, height, color_count):
        X = tf.placeholder(shape=[None, width, height, color_count], dtype=tf.float32, name="input_layer")
        return X

    def _prediction_layer(self, capsule_output):
        y_probability = safe_norm(capsule_output, axis=-2, name="y_proba")

        highest_probability_index = tf.argmax(y_probability, axis=2, name="y_proba")
        highest_probability_index = tf.squeeze(highest_probability_index, axis=[1, 2], name="y_pred")
        return highest_probability_index

    def _margin_loss(self, caps2_n_caps, caps2_output, correct_class):
        m_plus = 0.9
        m_minus = 0.1
        lambda_ = 0.5

        T = tf.one_hot(correct_class, depth=caps2_n_caps, name="T")
        caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True,
                                      name="caps2_output_norm")

        present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),
                                      name="present_error_raw")
        present_error = tf.reshape(present_error_raw, shape=(-1, 10),
                                   name="present_error")

        absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),
                                     name="absent_error_raw")
        absent_error = tf.reshape(absent_error_raw, shape=(-1, 10),
                                  name="absent_error")

        L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
                   name="L")
        margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")
        return margin_loss

    def reconstruction_targets(self, caps2_n_caps, caps2_n_dims, caps2_output, y, y_pred):

        mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                                       name="mask_with_labels")

        reconstruction_targets = tf.cond(mask_with_labels,  # condition
                                         lambda: y,  # if True
                                         lambda: y_pred,  # if False
                                         name="reconstruction_targets")

        reconstruction_mask = tf.one_hot(reconstruction_targets,
                                         depth=caps2_n_caps,
                                         name="reconstruction_mask")

        reconstruction_mask_reshaped = tf.reshape(
            reconstruction_mask, [-1, 1, caps2_n_caps, 1, 1],
            name="reconstruction_mask_reshaped")

        caps2_output_masked = tf.multiply(
            caps2_output, reconstruction_mask_reshaped,
            name="caps2_output_masked")

        decoder_input = tf.reshape(caps2_output_masked,
                                   [-1, caps2_n_caps * caps2_n_dims],
                                   name="decoder_input")

        return decoder_input, mask_with_labels

    def decoder(self, decoder_input):

        n_hidden1 = 512
        n_hidden2 = 1024
        n_output = 28 * 28

        with tf.name_scope("decoder"):
            hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                                      activation=tf.nn.relu,
                                      name="hidden1")
            hidden2 = tf.layers.dense(hidden1, n_hidden2,
                                      activation=tf.nn.relu,
                                      name="hidden2")
            decoder_output = tf.layers.dense(hidden2, n_output,
                                             activation=tf.nn.sigmoid,
                                             name="decoder_output")

            return decoder_output

    def _reconstruction_loss(self, X, n_output, decoder_output):
        X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
        squared_difference = tf.square(X_flat - decoder_output,
                                       name="squared_difference")
        reconstruction_loss = tf.reduce_mean(squared_difference,
                                             name="reconstruction_loss")

        return reconstruction_loss


    def _loss(self, margin_loss, reconstruction_loss):
        alpha = 0.0005

        loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")
        return loss

    def accuracy(self, y, y_pred):
        correct = tf.equal(y, y_pred, name="correct")
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
        return accuracy

    def optimizer(self, loss):
        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(loss, name="training_op")
        return training_op

    def conv_net(self, input_layer, filter_count):
        conv1_params = {
            "filters": 32,
            "kernel_size": 5,
            "strides": 1,
            "padding": "valid",
            "activation": tf.nn.relu,
        }

        conv2_params = {
            "filters": filter_count,
            "kernel_size": 5,
            "strides": 2,
            "padding": "valid",
            "activation": tf.nn.relu
        }

        conv1 = tf.layers.conv2d(input_layer, name="conv1", **conv1_params)
        conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)

        return conv2

    def input_capsule_layer(self, width, height, color_count, capsule_count, dimension_count):

        input_data = self.image_batch_placeholder(width, height, color_count)

        texture_filter_output = self.conv_net(input_data, capsule_count * dimension_count)

        capsule_representation = self.normalization_layer(
                                    tf.reshape(texture_filter_output, [-1, capsule_count, dimension_count]), name="input_capsules")

        return input_data, capsule_representation

    def capsule_layer(self, batch_size, input, input_capsule_count, input_dimension_count, output_capsule_count, output_dimension_count):

        potential_parents = self.potential_parent_layer(input, input_capsule_count, input_dimension_count, output_capsule_count, output_dimension_count)

        assembled_parents = self.assembly_layer(batch_size, input_capsule_count, output_capsule_count, potential_parents)

        return assembled_parents

    def potential_parent_layer(self, input, input_capsule_count, input_dimension_count, output_capsule_count, output_dimension_count):
        batch_size = tf.shape(input)[0]

        input_expanded = tf.expand_dims(input, -1, name="input_expanded")
        input_for_tiling = tf.expand_dims(input_expanded, 2, name="input_for_tiling")
        tiled_output = tf.tile(input_for_tiling, [1, 1, output_capsule_count, 1, 1], name="input_tiled")

        weights = self.potential_parent_weights(
            input_capsule_count, output_capsule_count,
            input_dimension_count, output_dimension_count)

        tiled_weights = tf.tile(weights, [batch_size, 1, 1, 1, 1], name="W_tiled")

        return tf.matmul(tiled_weights, tiled_output)

    def potential_parent_weights(self, input_capsule_count, output_capsule_count,
                                 input_capsule_dimension_count, output_capsule_dimension_count, init_sigma = 0.1):

        randomized_initial_weights = tf.random_normal(
            shape=(1, input_capsule_count, output_capsule_count,
                   output_capsule_dimension_count, input_capsule_dimension_count),
            stddev=init_sigma, dtype=tf.float32, name="randomized_initial_weights")
        W = tf.Variable(randomized_initial_weights, name="W")

        return W

    def assembly_layer(self, batch_size, input_capsule_count, output_capsule_count, caps2_predicted):

        # initialize weights
        raw_weights = tf.zeros([batch_size, input_capsule_count, output_capsule_count, 1, 1],
                               dtype=np.float32, name="raw_weights")

        def condition(input, counter):
            return tf.less(counter, 3)

        def loop_body(raw_weights, counter):

            # scale down routing weights
            routing_weights = tf.nn.softmax(raw_weights,
                                            dim=2,
                                            name="routing_weights")

            # compute weighted average of  predictions, i.e. the centroid of the harmoneous predictions
            # compute weighted predictions
            weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                               name="weighted_predictions")


            # sum up all predictions
            weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                                         name="weighted_sum")

            # and scale them down using squash
            caps2_centroid = self.normalization_layer(weighted_sum, axis=-2,
                                                    name="caps2_centroid_round_1")

            # now compute distance of all predictions to the centroid
            caps2_centroid_tiled = tf.tile(
                caps2_centroid, [1, input_capsule_count, 1, 1, 1],
                name="caps2_centroid_round_1_tiled")

            agreement = tf.matmul(caps2_predicted, caps2_centroid_tiled,
                                  transpose_a=True, name="agreement")

            raw_weights = tf.add(raw_weights, agreement,
                                 name="raw_weights")

            return raw_weights, tf.add(counter, 1)

        with tf.name_scope("compute_sum_of_squares"):
            counter = tf.constant(0)

            (raw_weights, counter) = tf.while_loop(condition, loop_body, [raw_weights, counter])

        routing_weights = tf.nn.softmax(raw_weights,
                                                dim=2,
                                                name="routing_weights")
        weighted_predictions = tf.multiply(routing_weights,
                                                   caps2_predicted,
                                                   name="weighted_predictions")
        weighted_sum = tf.reduce_sum(weighted_predictions,
                                             axis=1, keep_dims=True,
                                             name="weighted_sum")
        assembled_centroid = self.normalization_layer(weighted_sum,
                                      axis=-2,
                                      name="caps2_output")

        return assembled_centroid

    def _normalize_centroids(self, agreement_weights, predicted_centroids):
        routing_weights = tf.nn.softmax(agreement_weights, dim=2, name="routing_weights")
        weighted_predictions = tf.multiply(routing_weights, predicted_centroids,
                                           name="weighted_predictions")
        weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                                     name="weighted_sum")

        normalized_centroids = self.normalization_layer(weighted_sum)

        return normalized_centroids

    def normalization_layer(self, s, axis=-1, epsilon=1e-7, name=None):
        # reduce all vectors to length one
        with tf.name_scope(name, default_name="squash"):
            squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                         keep_dims=True)
            safe_norm = tf.sqrt(squared_norm + epsilon)
            squash_factor = squared_norm / (1. + squared_norm)
            unit_vector = s / safe_norm
            return squash_factor * unit_vector

    def train(self, mnist):
        n_epochs = 2 # 10
        batch_size = 10 # 50
        restore_checkpoint = True

        n_iterations_per_epoch = mnist.train.num_examples // batch_size
        n_iterations_validation = mnist.validation.num_examples // batch_size
        best_loss_val = np.infty
        checkpoint_path = "./output"

        with tf.Session() as sess:
            if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
                self.saver.restore(sess, checkpoint_path)
            else:
                self.init.run()

            for epoch in range(n_epochs):
                for iteration in range(1, n_iterations_per_epoch + 1):
                    X_batch, y_batch = mnist.train.next_batch(batch_size)
                    # Run the training operation and measure the loss:
                    _, loss_train = sess.run(
                        [self.training_op, self.loss],
                        feed_dict={self.input_data: X_batch.reshape([-1, 28, 28, 1]),
                                   self.correct_label: y_batch,
                                   self.mask_with_labels: True})
                    print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                        iteration, n_iterations_per_epoch,
                        iteration * 100 / n_iterations_per_epoch,
                        loss_train),
                        end="")

                # At the end of each epoch,
                # measure the validation loss and accuracy:
                loss_vals = []
                acc_vals = []
                for iteration in range(1, n_iterations_validation + 1):
                    X_batch, y_batch = mnist.validation.next_batch(batch_size)
                    loss_val, acc_val = sess.run(
                        [self.loss, self.accuracy],
                        feed_dict={self.input_data: X_batch.reshape([-1, 28, 28, 1]),
                                   self.mask_with_labels: y_batch})
                    loss_vals.append(loss_val)
                    acc_vals.append(acc_val)
                    print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                        iteration, n_iterations_validation,
                        iteration * 100 / n_iterations_validation),
                        end=" " * 10)
                loss_val = np.mean(loss_vals)
                acc_val = np.mean(acc_vals)
                print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
                    epoch + 1, acc_val * 100, loss_val,
                    " (improved)" if loss_val < best_loss_val else ""))

                # And save the model if it improved:
                if loss_val < best_loss_val:
                    save_path = self.saver.save(sess, checkpoint_path)
                    best_loss_val = loss_val

    def evaluate(self, mnist):
        n_iterations_test = mnist.test.num_examples // batch_size

        with tf.Session() as sess:
            self.saver.restore(sess, checkpoint_path)

            loss_tests = []
            acc_tests = []
            for iteration in range(1, n_iterations_test + 1):
                X_batch, y_batch = mnist.test.next_batch(batch_size)
                loss_test, acc_test = sess.run(
                    [self.loss, self.accuracy],
                    feed_dict={self.input_data: X_batch.reshape([-1, 28, 28, 1]),
                               self.mask_with_labels: y_batch})
                loss_tests.append(loss_test)
                acc_tests.append(acc_test)
                print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                    iteration, n_iterations_test,
                    iteration * 100 / n_iterations_test),
                    end=" " * 10)
            loss_test = np.mean(loss_tests)
            acc_test = np.mean(acc_tests)
            print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(
                acc_test * 100, loss_test))

    def predict(self, sample_images):
        n_samples = 5

        sample_images = mnist.test.images[:n_samples].reshape([-1, 28, 28, 1])

        with tf.Session() as sess:
            self.saver.restore(sess, checkpoint_path)
            caps2_output_value, decoder_output_value, y_pred_value = sess.run(
                [self.capsule_output, self.decoder_output, self.predicted_label],
                feed_dict={self.input_data: sample_images,
                           self.correct_label: np.array([], dtype=np.int64)})

    @staticmethod
    def load_mnist():

        mnist = input_data.read_data_sets("./data/mnist/")

        return mnist

    @staticmethod
    def main():
        print("MAIN")

        mnist = CapsNet.load_mnist()

        caps1_n_maps = 32
        caps1_n_caps = caps1_n_maps * 6 * 6  # 1152 primary capsules
        caps1_n_dims = 8

        caps2_n_caps = 10
        caps2_n_dims = 16

        cn = CapsNet().init(28, 28, 1, caps1_n_caps, caps1_n_dims, caps2_n_caps, caps2_n_dims)
        cn.train(mnist)
        return

        cn.evaluate(mnist)
        cn.predict(mnist)
        cn.visualize()

def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)

if __name__ == "__main__":
    CapsNet.main()
