import sys
import tensorflow as tf
import numpy as np
from learning import MatrixCapsNet
from learning.TopologyBuilder import TopologyBuilder


class TestMatrixCapsNet(tf.test.TestCase):

    def setUp(self):
        self.sess = tf.Session()

        self.guard = self.sess.as_default()
        self.guard.__enter__()

    def tearDown(self):
        tf.reset_default_graph()
        self.guard.__exit__(None, None, None)
        tf.reset_default_graph()

    def test_happy_flow_through_default_architecture(self):
        batch_count = 3
        output_count = 5
        random_input_images = np.random.normal(np.zeros([batch_count, 32, 32, 1]))
        full_example_count = 10
        iteration_count = 3
        routing_state = None
        is_training = tf.constant(True)

        input_layer = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])

        network_output, next_routing_state, _ = MatrixCapsNet(tf.float32).build_default_architecture(input_layer, iteration_count, routing_state)

        self.sess.run(tf.global_variables_initializer())

        output_data = self.sess.run([network_output, next_routing_state], feed_dict={
            input_layer: random_input_images
        })

        network_output_data, next_routing_state_data = output_data
        activations, poses = network_output_data

        self.assertFiniteAndShape(activations, [batch_count, output_count, 1], 'happy flow through default network; activations')
        self.assertFiniteAndShape(poses, [batch_count, output_count, 4, 4], 'happy flow through default network; poses')

        kernel_width_capsule_layer = 3
        kernel_element_count = kernel_width_capsule_layer**2
        capsule_filter_count_layer_B_C_and_D = 32
        flattened_convolutional_child_count_C_and_d = kernel_element_count * capsule_filter_count_layer_B_C_and_D

        patch_count_layer_B = 6**2
        patch_count_layer_C = 4**2
        output_patch_count_layer_D = patch_count_layer_C  # for each patch layer C produces an output
        output_count_layer_B = patch_count_layer_B * capsule_filter_count_layer_B_C_and_D
        output_count_layer_D = output_patch_count_layer_D * capsule_filter_count_layer_B_C_and_D  # each output patch has 32 capsule filters

        self.assertTrue(len(next_routing_state_data) == 3, "Routing takes place between 4 matrix capsule layers, so it there are 3 sets of routing weights")
        self.assertFiniteAndShape(next_routing_state_data[0],
                                  [batch_count, flattened_convolutional_child_count_C_and_d, output_count_layer_B, 1], "first routing layer weights must be stored correctly")
        self.assertFiniteAndShape(next_routing_state_data[1],
                                  [batch_count, flattened_convolutional_child_count_C_and_d, output_count_layer_D, 1], "second routing layer weights must be stored correctly")
        self.assertFiniteAndShape(next_routing_state_data[2],
                                  [batch_count, output_count_layer_D, output_count, 1], "final routing layer weights must be stored correctly")

    def test_progress_percentage_node(self):
        batch_size = 5
        full_example_count = 10
        is_training = tf.placeholder(tf.bool, shape=[])

        p = MatrixCapsNet(tf.float32).progress_percentage_node(batch_size, full_example_count, is_training)

        self.sess.run(tf.global_variables_initializer())

        [percentage_predicting, example_count_predicting] = self.sess.run(list(p), feed_dict={is_training: False})

        [percentage0, example_count0] = self.sess.run(list(p), feed_dict={is_training: True})

        [percentage1, example_count1] = self.sess.run(list(p), feed_dict={is_training: True})

        [percentage2, example_count2] = self.sess.run(list(p), feed_dict={is_training: True})

        [percentage3, example_count3] = self.sess.run(list(p), feed_dict={is_training: False})

        self.assertTrue(percentage_predicting == 1., "progress should be 100% when predicting")

        self.assertTrue(percentage0 == .5, "progress should increase")
        self.assertTrue(percentage1 == 1., "progress should reach 100%")
        self.assertTrue(percentage2 == 1., "progress should to 100% but not beyond")
        self.assertTrue(percentage3 == 1., "progress should be 100% when doing predictions")

        self.assertTrue(example_count_predicting == 0, "examples should not change when predicting")

        self.assertTrue(example_count0 == 5, "examples should start correctly")
        self.assertTrue(example_count1 == 10, "examples should increase")
        self.assertTrue(example_count2 == 15., "examples should increase till beyond full example count")

        self.assertTrue(example_count3 == 15., "examples not should increase when not training")

    def test_increasing_value(self):

        is_training = tf.placeholder(tf.bool, shape=[])

        i = MatrixCapsNet(tf.float32).increasing_value(.1, .2, is_training)

        self.sess.run(tf.global_variables_initializer())

        [value0] = self.sess.run([i], feed_dict={is_training: False})

        [value1] = self.sess.run([i], feed_dict={is_training: True})

        [value2] = self.sess.run([i], feed_dict={is_training: False})

        [value3] = self.sess.run([i], feed_dict={is_training: True})

        self.assertAlmostEqual(value0, .1, 4, "value should not have changed if predicting without any training")

        self.assertAlmostEqual(value1, .3, 4, "progress should increase")
        self.assertAlmostEqual(value2, .3, 4, "value should not have changed if predicting after any training")
        self.assertAlmostEqual(value3, .5, 4, "progress should increase when training further")

    def test_changing_value(self):
        batch_size = 5
        full_example_count = 15
        is_training = tf.placeholder(tf.bool, shape=[])

        start = .4
        finish = .7

        mcn = MatrixCapsNet(tf.float32)
        p = mcn.progress_percentage_node(batch_size, full_example_count, is_training)[0]
        cv = mcn.changing_value(start, finish, p)

        self.sess.run(tf.global_variables_initializer())

        [value0] = self.sess.run([cv], feed_dict={is_training: False})

        [value1] = self.sess.run([cv], feed_dict={is_training: True})

        [value2] = self.sess.run([cv], feed_dict={is_training: False})

        [value3] = self.sess.run([cv], feed_dict={is_training: True})

        [value4] = self.sess.run([cv], feed_dict={is_training: True})

        self.assertAlmostEqual(value0, .7, 4, "value should assume completion when not training")

        self.assertAlmostEqual(value1, .5, 4, "progress should increase")
        self.assertAlmostEqual(value2, .7, 4, "value should assume completion when not training")
        self.assertAlmostEqual(value3, .6, 4, "progress should increase when training further")

        self.assertAlmostEqual(value4, .7, 4, "progress should increase when training further")

        self.assertAlmostEqual(value4, .7, 4, "progress should not increase beyond max despite training further")


    def test_conv_layer(self):
        random_input_images = np.random.normal(np.zeros([3, 32, 32, 1]))

        input_layer = tf.placeholder('float', shape=[None, 32, 32, 1])

        convolution_layer_A = MatrixCapsNet(tf.float32).build_encoding_convolution(input_layer, 5, 32)

        self.sess.run(tf.global_variables_initializer())

        output_data = convolution_layer_A.eval(feed_dict={
            input_layer: random_input_images
        })

        self.assertTrue(np.isfinite(output_data).all(), "convolutional layer must produce finite data")
        self.assertTrue((np.array(output_data.shape) == np.array([3, 14, 14, 32])).all(), "convolutional layer must produce finite data")

    def test_primary_capsule_layer(self):
        input_layer = tf.placeholder('float', shape=[None, 32, 32, 1])

        mcn = MatrixCapsNet(tf.float32)

        convolution_layer_A = mcn.build_encoding_convolution(input_layer, 5, 32)
        primary_capsules_B = mcn.build_primary_matrix_caps(convolution_layer_A)

        random_input_images = np.random.normal(np.zeros([3, 32, 32, 1]))

        self.sess.run(tf.global_variables_initializer())
        primary_capsule_output = self.sess.run([primary_capsules_B[0], primary_capsules_B[1]], feed_dict={
            input_layer: random_input_images
        })

        activations, poses = primary_capsule_output

        self.assertTrue(np.isfinite(activations).all(), "primary caps layer activations must produce finite data")
        self.assertTrue(np.isfinite(poses).all(), "primary caps layer poses must produce finite data")

        self.assertTrue((np.array(activations.shape) == np.array([3, 14, 14, 32, 1])).all(), "primary caps layer activation must have right shape")
        self.assertTrue((np.array(poses.shape) == np.array([3, 14, 14, 32, 4, 4])).all(), "primary caps layer poses must have right shape")

    def test_primary_capsule_layer_axial_system(self):
        input_layer = tf.placeholder('float', shape=[None, 32, 32, 1])

        mcn = MatrixCapsNet(tf.float32)

        convolution_layer_A = mcn.build_encoding_convolution(input_layer, 5, 32)
        primary_capsules_B = mcn.build_primary_matrix_caps(convolution_layer_A, is_axial_system=True)

        random_input_images = np.random.normal(np.zeros([3, 32, 32, 1]))

        self.sess.run(tf.global_variables_initializer())
        primary_capsule_output = self.sess.run([primary_capsules_B[0], primary_capsules_B[1]], feed_dict={
            input_layer: random_input_images
        })

        _, poses = primary_capsule_output

        self.assertTrue(np.all(poses[:, :, :, :, :, 3] == np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)), "primary caps layer that is axial system must produce poses with correct type elements")



    def test_aggregating_capsule_layer(self):
        final_steepness_lambda, iteration_count, routing_state = tf.constant(1.0), 3, None

        random_input_images = np.random.normal(np.zeros([3, 32, 32, 1]))

        input_layer = tf.placeholder(shape=[None, 32, 32, 1], dtype=tf.float32)

        mcn = MatrixCapsNet(dtype=tf.float32)

        convolution_layer_A = mcn.build_encoding_convolution(input_layer, 5, 32)
        primary_capsules_B = mcn.build_primary_matrix_caps(convolution_layer_A)

        aggregating_topology = TopologyBuilder().init()
        aggregating_topology.add_aggregation(primary_capsules_B[0].get_shape().as_list()[1], [[3, 0], [3, 1]])
        aggregating_topology.add_dense_connection(primary_capsules_B[0].get_shape().as_list()[3], 5)
        aggregating_topology.finish({
            "type": "identity",
            "uniform": .03
        })

        aggregating_capsule_layer = mcn.build_matrix_caps(
            primary_capsules_B,
            aggregating_topology,
            final_steepness_lambda,
            iteration_count,
            None
        )

        final_activations = aggregating_topology.reshape_parent_map_to_linear(aggregating_capsule_layer[0])
        final_poses = aggregating_topology.reshape_parent_map_to_linear(aggregating_capsule_layer[1])

        self.sess.run(tf.global_variables_initializer())
        aggregated_capsule_output = self.sess.run([final_activations, final_poses], feed_dict={
            input_layer: random_input_images
        })

        activations, poses = aggregated_capsule_output

        self.assertFiniteAndShape(activations, [3, 5, 1], "aggregated caps activations: ")
        self.assertFiniteAndShape(poses, [3, 5, 4, 4], "aggregated caps poses:")

    def test_estimate_parents_layer(self):

        batch_size = 3
        child_count = 4
        parent_count = 5
        pose_element_count = 16

        child_parent_assignment_weights = tf.constant(np.random.random([batch_size, child_count, parent_count, 1]))
        child_activations  = tf.constant(np.random.random([batch_size, child_count, 1, 1]))
        potential_pose_vectors  = tf.constant(np.random.normal(np.zeros([batch_size, child_count, parent_count, pose_element_count])))
        beta_u = tf.constant(np.random.normal(np.zeros([1, 1, parent_count, 1])))
        beta_a = tf.constant(np.random.normal(np.zeros([1, 1, parent_count, 1])))
        steepness_lambda = tf.constant(1.0, dtype=tf.float64)

        mcn = MatrixCapsNet(tf.float32)

        active_child_parent_assignment_weights = child_parent_assignment_weights * child_activations

        estimate_parent_layer = mcn.estimate_parents_layer(
            active_child_parent_assignment_weights,
            potential_pose_vectors,
            beta_u,
            beta_a,
            steepness_lambda
        )

        self.sess.run(tf.global_variables_initializer())
        estimate_parent_output = self.sess.run(list(estimate_parent_layer))

        parent_activations, likely_parent_pose, likely_parent_pose_deviation, likely_parent_pose_variance = estimate_parent_output

        self.assertFiniteAndShape(parent_activations, [batch_size, 1, parent_count, 1], "estimate parents, parent_activations ")
        self.assertFiniteAndShape(likely_parent_pose, [batch_size, 1, parent_count, pose_element_count], "estimate parents, likely_parent_pose ")
        self.assertFiniteAndShape(likely_parent_pose_deviation, [batch_size, 1, parent_count, pose_element_count], "estimate parents, likely_parent_pose_deviation")
        self.assertFiniteAndShape(likely_parent_pose_variance, [batch_size, 1, parent_count, pose_element_count], "estimate parents, likely_parent_pose_variance")

    def test_estimate_children_layer(self):

        batch_size = 3
        child_count = 4
        parent_count = 5
        pose_element_count = 16

        topology = TopologyBuilder().init()
        topology.add_dense_connection(child_count, parent_count)
        topology.finish({
            "type": "normal",
            "deviation": [1.0]
        })

        data = np.random.random([batch_size, child_count, parent_count, 1])
        data[0,0,0,0] = 1.0
        child_parent_assignment_weights = tf.constant(data, dtype=tf.float32)
        child_activations = tf.constant(np.random.random([batch_size, child_count, 1, 1]), dtype=tf.float32)
        potential_pose_vectors = tf.constant(
            np.random.normal(np.zeros([batch_size, child_count, parent_count, pose_element_count])), dtype=tf.float32)
        beta_u = tf.constant(np.random.normal(np.zeros([1, 1, parent_count, 1])), dtype=tf.float32)
        beta_a = tf.constant(np.random.normal(np.zeros([1, 1, parent_count, 1])), dtype=tf.float32)
        steepness_lambda = tf.constant(1.0, dtype=tf.float32)

        mcn = MatrixCapsNet(tf.float32)

        active_child_parent_assignment_weights = child_parent_assignment_weights * child_activations

        estimate_parent_layer = mcn.estimate_parents_layer(
            active_child_parent_assignment_weights,
            potential_pose_vectors,
            beta_u,
            beta_a,
            steepness_lambda
        )
        parent_activations, likely_parent_pose, likely_parent_pose_deviation, likely_parent_pose_variance = estimate_parent_layer

        estimate_children_layer = mcn.estimate_children_layer(parent_activations, likely_parent_pose, likely_parent_pose_variance, potential_pose_vectors, topology)

        self.sess.run(tf.global_variables_initializer())
        child_parent_assignment_weights = self.sess.run([estimate_children_layer])[0]

        self.assertFiniteAndShape(child_parent_assignment_weights, [batch_size, child_count, parent_count, 1], "estimate_children_layer, child_parent_assignment_weights")
        self.assertTrue((child_parent_assignment_weights >= 0).all(), "all child parent assignment weights must be >= 0")

    def test_estimate_parents_layer_learning(self):

        batch_size = 1
        child_count = 4
        parent_count = 5
        pose_element_count = 16

        child_parent_assignment_weights = tf.Variable(np.random.random([batch_size, child_count, parent_count, 1]), dtype=tf.float32)
        child_activations  = tf.Variable(np.random.random([batch_size, child_count, 1, 1]), dtype=tf.float32)
        potential_pose_vectors  = tf.Variable(np.random.normal(np.zeros([batch_size, child_count, parent_count, pose_element_count])), dtype=tf.float32)
        beta_u = tf.Variable(np.random.normal(np.zeros([1, 1, parent_count, 1])), dtype=tf.float32)
        beta_a = tf.Variable(np.random.normal(np.zeros([1, 1, parent_count, 1])), dtype=tf.float32)
        steepness_lambda = tf.constant(1.0, dtype=tf.float32)

        mcn = MatrixCapsNet(tf.float32)
        active_child_parent_assignment_weights = child_parent_assignment_weights * child_activations

        parent_activations, likely_parent_pose, likely_parent_pose_deviation, likely_parent_pose_variance = mcn.estimate_parents_layer(
            active_child_parent_assignment_weights,
            potential_pose_vectors,
            beta_u,
            beta_a,
            steepness_lambda
        )

        correct_parent_activations = tf.random_uniform([batch_size, 1, parent_count, 1])
        correct_likely_parent_pose = tf.random_uniform([batch_size, 1, parent_count, pose_element_count])

        loss = tf.losses.mean_squared_error(correct_parent_activations, parent_activations) + tf.losses.mean_squared_error(likely_parent_pose, correct_likely_parent_pose)

        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(train_op)

        parent_activation_values, likely_parent_pose_values = self.sess.run([parent_activations, likely_parent_pose])

        self.assertFiniteAndShape(parent_activation_values, [batch_size, 1, parent_count, 1],
                                  "estimate parents, parent_activations after learning")

        self.assertFiniteAndShape(likely_parent_pose_values, [batch_size, 1, parent_count, pose_element_count], "estimate parents, likely_parent_pose after learning")

    def test_estimate_children_layer_learning(self):

        batch_size = 1
        child_count = 4
        parent_count = 5
        pose_element_count = 16

        topology = TopologyBuilder().init()
        topology.add_dense_connection(child_count, parent_count)
        topology.finish({
            "type": "identity",
            "uniform": 0.3
        })

        parent_activations = tf.Variable(np.random.random([batch_size, 1, parent_count, 1]), dtype=tf.float32)
        potential_pose_vectors = tf.Variable(
            np.random.normal(np.zeros([batch_size, child_count, parent_count, pose_element_count])), dtype=tf.float32)
        likely_parent_pose = tf.Variable(np.random.normal(np.zeros([batch_size, 1, parent_count, pose_element_count])), dtype=tf.float32)
        likely_parent_pose_variance = tf.Variable(np.random.random([batch_size, 1, parent_count, pose_element_count]), dtype=tf.float32)

        mcn = MatrixCapsNet(tf.float32)

        child_parent_assignment_weights = mcn.estimate_children_layer(parent_activations, likely_parent_pose, likely_parent_pose_variance, potential_pose_vectors, topology)

        correct_child_parent_assignment_weights = tf.constant(np.random.random([batch_size, child_count, parent_count, 1]))

        loss = tf.losses.mean_squared_error(correct_child_parent_assignment_weights, child_parent_assignment_weights)

        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(train_op)

        child_parent_assignment_weights_values = self.sess.run([child_parent_assignment_weights])[0]

        self.assertFiniteAndShape(child_parent_assignment_weights_values, [batch_size, child_count, parent_count, 1], "estimate_children_layer, child_parent_assignment_weights")
        self.assertTrue((child_parent_assignment_weights_values >= 0).all(), "all child parent assignment weights must be >= 0")

    def test_estimate_children_parents_communication_learning(self):
        batch_size = 1
        child_count = 4
        parent_count = 5
        pose_element_count = 16

        topology = TopologyBuilder().init()
        topology.add_dense_connection(child_count, parent_count)
        topology.finish({
            "type": "xavier",
            "kernel": True
        })

        child_parent_assignment_weights = tf.Variable(np.random.random([batch_size, child_count, parent_count, 1]), dtype=tf.float32)
        child_activations  = tf.Variable(np.random.random([batch_size, child_count, 1, 1]), dtype=tf.float32)
        potential_pose_vectors  = tf.Variable(np.random.normal(np.zeros([batch_size, child_count, parent_count, pose_element_count])), dtype=tf.float32)
        beta_u = tf.Variable(np.random.normal(np.zeros([1, 1, parent_count, 1])), dtype=tf.float32)
        beta_a = tf.Variable(np.random.normal(np.zeros([1, 1, parent_count, 1])), dtype=tf.float32)
        steepness_lambda = tf.constant(1.0, dtype=tf.float32)

        mcn = MatrixCapsNet(tf.float32)

        active_child_parent_assignment_weights = child_parent_assignment_weights * child_activations

        parent_activations, likely_parent_pose, likely_parent_pose_deviation, likely_parent_pose_variance = mcn.estimate_parents_layer(
            active_child_parent_assignment_weights,
            potential_pose_vectors,
            beta_u,
            beta_a,
            steepness_lambda
        )

        child_parent_assignment_weights = mcn.estimate_children_layer(parent_activations, likely_parent_pose, likely_parent_pose_variance, potential_pose_vectors, topology)

        correct_child_parent_assignment_weights = tf.constant(np.random.random([batch_size, child_count, parent_count, 1]))

        loss = tf.losses.mean_squared_error(correct_child_parent_assignment_weights, child_parent_assignment_weights)

        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(train_op)
        child_parent_assignment_weights_values = self.sess.run(child_parent_assignment_weights)

        self.assertFiniteAndShape(
            child_parent_assignment_weights_values,
            [batch_size, child_count, parent_count, 1],
            "one loop of learning over children parent estimation")

    def test_build_parent_assembly_layer(self):

        iteration_count = 3
        routing_state = None
        final_steepness_lambda = tf.constant(.5)

        batch_size = 3
        spatial_shape = [6, 6]
        input_feature_count = 32
        output_feature_count = 32
        pose_element_count = 16

        topology = TopologyBuilder().init()
        topology.add_spatial_convolution(spatial_shape, 3, 1)
        topology.add_dense_connection(input_feature_count, output_feature_count)
        topology.finish({
            "type": "xavier",
            "kernel": True
        })

        linear_kernel_linear_parent_shape = [batch_size, topology.linear_kernel_shape(), topology.linear_parent_shape()]
        child_activations_lklp   = np.random.random( linear_kernel_linear_parent_shape + [1])
        potential_pose_vectors_lklp = np.random.random(linear_kernel_linear_parent_shape + [16])

        batch_size = 3

        child_activations_placeholder = tf.placeholder(tf.float32, shape=[None] + linear_kernel_linear_parent_shape[1:] + [1])
        potential_pose_vectors_placeholder = tf.placeholder(tf.float32, shape=[None] + linear_kernel_linear_parent_shape[1:] + [pose_element_count])

        mcn = MatrixCapsNet(tf.float32)

        capsule_output = mcn.build_parent_assembly_layer(child_activations_placeholder, potential_pose_vectors_placeholder, final_steepness_lambda, iteration_count, routing_state, topology)

        self.sess.run(tf.global_variables_initializer())
        capsule_output_data = self.sess.run(list(capsule_output), feed_dict={
            child_activations_placeholder: child_activations_lklp,
            potential_pose_vectors_placeholder: potential_pose_vectors_lklp
        })
        activations, pose_vectors = capsule_output_data

        self.assertFiniteAndShape(activations, [batch_size, 1, topology.linear_parent_shape(), 1], "parent assembly layer, activations:")
        self.assertTrue((activations >= 0).all(), "activations must always be positive")
        self.assertFiniteAndShape(pose_vectors, [batch_size, 1, topology.linear_parent_shape(), pose_element_count], "parent assembly layer, pose_vectors:")

    def test_build_parent_assembly_layer_learning(self):

        iteration_count = 3
        routing_state = None
        final_steepness_lambda = tf.constant(.5)

        batch_size = 3
        spatial_shape = [6, 6]
        input_feature_count = 32
        output_feature_count = 32
        pose_element_count = 16

        topology = TopologyBuilder().init()
        topology.add_spatial_convolution(spatial_shape, 3, 1)
        topology.add_dense_connection(input_feature_count, output_feature_count)
        topology.finish({
            "type": "xavier",
            "matrix": True
        })

        linear_kernel_linear_parent_shape = [batch_size, topology.linear_kernel_shape(), topology.linear_parent_shape()]
        child_activations_lklp   = np.random.random( linear_kernel_linear_parent_shape + [1])
        potential_pose_vectors_lklp = np.random.random(linear_kernel_linear_parent_shape + [16])

        batch_size = 3

        child_activations_placeholder = tf.Variable(child_activations_lklp, dtype=tf.float32)
        potential_pose_vectors_placeholder = tf.Variable(potential_pose_vectors_lklp, dtype=tf.float32)

        mcn = MatrixCapsNet(tf.float32)

        capsule_output = mcn.build_parent_assembly_layer(child_activations_placeholder, potential_pose_vectors_placeholder, final_steepness_lambda, iteration_count, routing_state, topology)
        activations, pose_vectors = capsule_output

        correct_activations = tf.constant(np.random.random([batch_size, 1, topology.linear_parent_shape(), 1]), dtype=tf.float32)
        correct_poses = tf.constant(np.random.normal(np.zeros([batch_size, 1, topology.linear_parent_shape(), pose_element_count])), dtype=tf.float32)

        loss = tf.losses.mean_squared_error(correct_activations, activations) + tf.losses.mean_squared_error(correct_poses, pose_vectors)

        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(train_op)
        activation_data, pose_data = self.sess.run([activations, pose_vectors])

        self.assertFiniteAndShape(activation_data, [batch_size, 1, topology.linear_parent_shape(), 1], 'learning parent assembly layer')
        self.assertFiniteAndShape(pose_data, [batch_size, 1, topology.linear_parent_shape(), pose_element_count], 'learning parent assembly layer')

    def test_build_matrix_caps_learning(self):

        iteration_count = 3
        routing_state = None
        final_steepness_lambda = tf.constant(.5)

        batch_size = 3
        spatial_shape = [6, 6]
        input_feature_count = 32
        output_feature_count = 32

        topology = TopologyBuilder().init()
        topology.add_spatial_convolution(spatial_shape, 3, 1)
        topology.add_dense_connection(input_feature_count, output_feature_count)

        topology.finish({
            "type": "xavier",
            "kernel": True
        })
        mock_activations   = tf.get_variable(name='mock_activations', initializer=tf.random_uniform([batch_size] + spatial_shape + [input_feature_count, 1], dtype=tf.float32), dtype=tf.float32, trainable=False)
        mock_pose_matrices = tf.get_variable(name='mock_pose_matrices', initializer=tf.truncated_normal([batch_size] + spatial_shape + [input_feature_count, 4, 4], 0, np.sqrt(1.0 / 20.0),
                                                             dtype=tf.float32), dtype=tf.float32, trainable=False)

        mcn = MatrixCapsNet(tf.float32)

        output = mcn.build_matrix_caps(
            [mock_activations, mock_pose_matrices],
            topology,
            final_steepness_lambda,
            iteration_count,
            routing_state
        )

        output_activations = output[0]
        output_poses = output[1]

        correct_activations = tf.random_uniform([batch_size] + topology.parent_shape() + [1])

        loss = tf.losses.mean_squared_error(correct_activations, output_activations)

        optimizer = tf.train.AdamOptimizer(epsilon=1e-4)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        self.sess.run(tf.global_variables_initializer())
        self.sess.run([train_op])
        activation_values = self.sess.run([output_activations])[0]

        self.assertFiniteAndShape(activation_values, [batch_size] + topology.parent_shape() + [1], "activation values should be finite after training")

    def assertFiniteAndShape(self, tensor_array, tensor_shape, message):
        self.assertTrue(np.isfinite(tensor_array).all(), message + ": does not have finite data")
        self.assertTrue((np.array(tensor_array.shape) ==
                        np.array(tensor_shape)).all(), message + ": does not have correct shape")
