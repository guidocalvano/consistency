import tensorflow as tf
import time
import sys



# batch_count = 17 # sgd + sgd/nesterov
batch_count = int(sys.argv[1]) # adam
mock_output = tf.random_uniform((batch_count, 5, 1, 1))
from learning.MatrixCapsNet import MatrixCapsNet

with tf.device('/gpu:0'):

    mcn = MatrixCapsNet()
    self = mcn
    full_example_count = batch_count
    random_input = tf.truncated_normal([batch_count, 32, 32, 1])
    iteration_count = 3
    routing_state = None
    is_training = tf.constant(True)

    input_layer = random_input


    texture_patches_A = 32
    capsule_count_C = 32
    capsule_count_D = 32
    capsule_count_E = 5

    batch_size = tf.cast(tf.gather(tf.shape(input_layer), 0), tf.float32)

    progress_percentage_node = self.progress_percentage_node(batch_size, full_example_count, is_training)[0]

    steepness_lambda = self.increasing_value(.01, .01, is_training)
    spread_loss_margin = self.changing_value(.2, .9, progress_percentage_node)

    convolution_layer_A = self.build_encoding_convolution(input_layer, 5, texture_patches_A)

    # number of capsules is defined by number of texture patches
    primary_capsule_layer_B = self.build_primary_matrix_caps(convolution_layer_A)

    conv_caps_layer_C = self.build_convolutional_capsule_layer(
        primary_capsule_layer_B,
        3,
        2,
        capsule_count_C,
        steepness_lambda,
        iteration_count,
        routing_state
    )

    conv_caps_layer_D = self.build_convolutional_capsule_layer(
        conv_caps_layer_C,
        3,
        1,
        capsule_count_D,
        steepness_lambda,
        iteration_count,
        routing_state
    )

    use_coordinate_addition = True
    aggregating_capsule_layer = self.build_aggregating_capsule_layer(
        conv_caps_layer_D,
        use_coordinate_addition,
        capsule_count_E,
        steepness_lambda,
        iteration_count,
        routing_state
    )


    loss = tf.losses.mean_squared_error(mock_output, aggregating_capsule_layer[0])
    # optimizer = tf.train.MomentumOptimizer(.1, .9, use_nesterov=True)
    # optimizer = tf.train.GradientDescentOptimizer(.001)
    optimizer = tf.train.AdamOptimizer()

    train_op = optimizer.minimize(loss)
sess = tf.Session()
sess.__enter__()
sess.run(tf.global_variables_initializer())
t0 = time.time()
res = sess.run(train_op)
t1 = time.time()
print(t1-t0)
