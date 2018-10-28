import tensorflow as tf
import numpy as np
import math
import sys
from learning.TopologyBuilder import TopologyBuilder


class MatrixCapsNet:
    '''
    IMPORTANT!, READ THIS FIRST:
    Matrix capsules are connected through complex network topologies. These complex topologies require complex matrices
    which must be combined with complex interactions. However, these interactions are always done on batches, and
    always concern the relation ship between child capsules and parent capsules through their state and/or connections.

    In practice I found that every single one of these operations between parent and child could be solved through the
    semantics of broadcasting. That is why throughout the entire code any tensors related to matrix capsules are
    defined according to a strict schema: [batch, child, parent, state dimensions]. Every tensor related to matrix capsules
    MUST always have all these dimensions.

    For instance, a tensor containing data on child state must be defined as
    [batch, child, 1, state dimensions]. The 1 assures that the tensor will be broadcasted correctly in any operations
    that relate it to its (potential) parents. Though some tensors as a consequence have weird columns with a 1 in them,
    as a whole the code requires almost no tile and repeat operations and is thus a lot cleaner and more readable.

    For convolutional code this rule must still be applied, but to do so it must be reformulated in a more general sense:
    [
    batch, # e.g. which example image or which image patch
    child indexing dimensions, # e.g. width, height, feature or kernel width, kernel height, feature
    parent indexing dimensions,
    state dimensions # e.g. activation or a pose matrix, or a vector of pose elements
    ]

    '''

    def build_default_architecture(self, input_layer, iteration_count, routing_state):

        with tf.name_scope('default_matrix_capsule_architecture') as scope0:

            texture_patches_A = 32
            capsule_count_C = 32
            capsule_count_D = 32
            capsule_count_E = 5

            final_steepness_lambda = tf.constant(.01)

            with tf.name_scope('layerA') as scope1:
                convolution_layer_A = self.build_encoding_convolution(input_layer, 5, texture_patches_A)

            with tf.name_scope('layerB') as scope2:
                # number of capsules is defined by number of texture patches
                primary_capsule_layer_B = self.build_primary_matrix_caps(convolution_layer_A)

            with tf.name_scope('layerC') as scope3:
                c_topology = TopologyBuilder().init()
                c_topology.add_spatial_convolution(primary_capsule_layer_B[0].get_shape().as_list()[1:3], 3, 2)
                c_topology.add_dense_connection(primary_capsule_layer_B[0].get_shape().as_list()[3], capsule_count_C)
                c_topology.finish()

                conv_caps_layer_C = self.build_matrix_caps(
                    primary_capsule_layer_B,
                    c_topology,
                    final_steepness_lambda,
                    iteration_count,
                    routing_state
                )

            with tf.name_scope('layerD') as scope3:
                d_topology = TopologyBuilder().init()
                d_topology.add_spatial_convolution(conv_caps_layer_C[0].get_shape().as_list()[1:3], 3, 1)
                d_topology.add_dense_connection(conv_caps_layer_C[0].get_shape().as_list()[3], capsule_count_D)
                d_topology.finish()

                conv_caps_layer_D = self.build_matrix_caps(
                    conv_caps_layer_C,
                    d_topology,
                    final_steepness_lambda,
                    iteration_count,
                    routing_state
                )

            with tf.name_scope('layerAggregation') as scope3:

                aggregating_topology = TopologyBuilder().init()
                aggregating_topology.add_aggregation(conv_caps_layer_D[0].get_shape().as_list()[1], [[3, 0], [3, 1]])
                aggregating_topology.add_dense_connection(conv_caps_layer_D[0].get_shape().as_list()[3], capsule_count_E)
                aggregating_topology.finish()

                aggregating_capsule_layer = self.build_matrix_caps(
                    conv_caps_layer_D,
                    aggregating_topology,
                    final_steepness_lambda,
                    iteration_count,
                    routing_state
                )

            with tf.name_scope('outputFormatting') as scope3:

                final_activations = aggregating_topology.reshape_parent_map_to_linear(aggregating_capsule_layer[0])
                final_poses = aggregating_topology.reshape_parent_map_to_linear(aggregating_capsule_layer[1])

                next_routing_state = tf.get_collection("next_routing_state")

        return [final_activations, final_poses], next_routing_state, None

    def build_simple_architecture(self, input_layer, iteration_count, routing_state):

        routing_state = None

        iteration_count = 3

        texture_patches_A = 10
        capsule_count_C = 4
        capsule_count_D = 5

        final_steepness_lambda = tf.constant(.01)

        convolution_layer_A = self.build_encoding_convolution(input_layer, 5, texture_patches_A)

        # number of capsules is defined by number of texture patches
        primary_capsule_layer_B = self.build_primary_matrix_caps(convolution_layer_A)

        c_topology = TopologyBuilder().init()
        c_topology.add_spatial_convolution(primary_capsule_layer_B[0].get_shape().as_list()[1:3], 3, 2)
        c_topology.add_dense_connection(primary_capsule_layer_B[0].get_shape().as_list()[3], capsule_count_C)
        c_topology.finish()

        conv_caps_layer_C = self.build_matrix_caps(
            primary_capsule_layer_B,
            c_topology,
            final_steepness_lambda,
            iteration_count,
            routing_state
        )

        aggregating_topology = TopologyBuilder().init()
        aggregating_topology.add_aggregation(conv_caps_layer_C[0].get_shape().as_list()[1], [[3, 0], [3, 1]])
        aggregating_topology.add_dense_connection(conv_caps_layer_C[0].get_shape().as_list()[3], capsule_count_D)
        aggregating_topology.finish()

        aggregating_capsule_layer = self.build_matrix_caps(
            conv_caps_layer_C,
            aggregating_topology,
            final_steepness_lambda,
            iteration_count,
            routing_state
        )

        final_activations = aggregating_topology.reshape_parent_map_to_linear(aggregating_capsule_layer[0])
        final_poses = aggregating_topology.reshape_parent_map_to_linear(aggregating_capsule_layer[1])

        next_routing_state = tf.get_collection("next_routing_state")

        return [final_activations, final_poses], next_routing_state, None

    def build_axial_system_architecture(self, input_layer, iteration_count, routing_state):

        with tf.name_scope('axial_system_matrix_capsule_architecture') as scope0:

            texture_patches_A = 32
            capsule_count_C = 32
            capsule_count_D = 32
            capsule_count_E = 5

            final_steepness_lambda = tf.constant(.01)

            convolution_layer_A = self.build_encoding_convolution(input_layer, 5, texture_patches_A)

            # number of capsules is defined by number of texture patches
            primary_capsule_layer_B = self.build_primary_matrix_caps(convolution_layer_A, is_axial_system=True)

            c_topology = TopologyBuilder().init()
            c_topology.set_is_axial_system(True)
            c_topology.add_spatial_convolution(primary_capsule_layer_B[0].get_shape().as_list()[1:3], 3, 2)
            c_topology.add_dense_connection(primary_capsule_layer_B[0].get_shape().as_list()[3], capsule_count_C)
            c_topology.finish()

            conv_caps_layer_C = self.build_matrix_caps(
                primary_capsule_layer_B,
                c_topology,
                final_steepness_lambda,
                iteration_count,
                routing_state
            )

            d_topology = TopologyBuilder().init()
            d_topology.set_is_axial_system(True)
            d_topology.add_spatial_convolution(conv_caps_layer_C[0].get_shape().as_list()[1:3], 3, 1)
            d_topology.add_dense_connection(conv_caps_layer_C[0].get_shape().as_list()[3], capsule_count_D)
            d_topology.finish()

            conv_caps_layer_D = self.build_matrix_caps(
                conv_caps_layer_C,
                d_topology,
                final_steepness_lambda,
                iteration_count,
                routing_state
            )

            aggregating_topology = TopologyBuilder().init()
            aggregating_topology.set_is_axial_system(True)
            aggregating_topology.add_aggregation(conv_caps_layer_D[0].get_shape().as_list()[1], [[3, 0], [3, 1]])
            aggregating_topology.add_dense_connection(conv_caps_layer_D[0].get_shape().as_list()[3], capsule_count_E)
            aggregating_topology.finish()

            aggregating_capsule_layer = self.build_matrix_caps(
                conv_caps_layer_D,
                aggregating_topology,
                final_steepness_lambda,
                iteration_count,
                routing_state
            )

            final_activations = aggregating_topology.reshape_parent_map_to_linear(aggregating_capsule_layer[0])
            final_poses = aggregating_topology.reshape_parent_map_to_linear(aggregating_capsule_layer[1])

            next_routing_state = tf.get_collection("next_routing_state")

            axial_system_loss = tf.reduce_sum(tf.get_collection("unit_scale_regularization")) + tf.reduce_sum(tf.get_collection("orthogonal_regularization"))

        return [final_activations, final_poses], next_routing_state, axial_system_loss

    def build_semantic_convolution_architecture(self, input_layer, iteration_count, routing_state):

        with tf.name_scope('semantic_convolution_matrix_capsule_architecture') as scope0:

            texture_patches_A = 64
            capsule_count_C = 32
            capsule_count_D = 32
            capsule_count_E = 5

            final_steepness_lambda = tf.constant(.01)

            convolution_layer_A = self.build_encoding_convolution(input_layer, 5, texture_patches_A)

            # number of capsules is defined by number of texture patches
            primary_capsule_layer_B = self.build_primary_matrix_caps(convolution_layer_A)

            semantically_convolved_input_shape = primary_capsule_layer_B[0].get_shape().as_list()[1:3] + [8, 8]

            semantically_convolved_activation_shape = semantically_convolved_input_shape + [1]
            semantically_convolved_pose_shape = semantically_convolved_input_shape + [4, 4]

            semantically_convolved_primary_capsule_layer_B = [
                tf.reshape(primary_capsule_layer_B[0], semantically_convolved_activation_shape),
                tf.reshape(primary_capsule_layer_B[1], semantically_convolved_pose_shape)
            ]

            c_topology = TopologyBuilder().init()
            c_topology.add_spatial_convolution(semantically_convolved_primary_capsule_layer_B[0].get_shape().as_list()[1:3], 3, 2)
            c_topology.add_semantic_convolution(semantically_convolved_primary_capsule_layer_B[0].get_shape().as_list()[3:5], 3, 1)
            c_topology.finish()

            conv_caps_layer_C = self.build_matrix_caps(
                primary_capsule_layer_B,
                c_topology,
                final_steepness_lambda,
                iteration_count,
                routing_state
            )

            d_topology = TopologyBuilder().init()
            d_topology.add_spatial_convolution(conv_caps_layer_C[0].get_shape().as_list()[1:3], 3, 1)
            d_topology.add_semantic_convolution(conv_caps_layer_C[0].get_shape().as_list()[3:5], 3, 1)
            d_topology.finish()

            conv_caps_layer_D = self.build_matrix_caps(
                conv_caps_layer_C,
                d_topology,
                final_steepness_lambda,
                iteration_count,
                routing_state
            )

            semantically_collapsed_layer_D_shape = conv_caps_layer_D[0].get_shape().as_list()[0:3] + \
                                                   [np.prod(conv_caps_layer_C[0].get_shape().as_list()[3:5])]

            semantically_collapsed_layer_D_activation_shape = semantically_collapsed_layer_D_shape + [1]
            semantically_collapsed_layer_D_pose_shape = semantically_collapsed_layer_D_shape + [4, 4]

            semantically_collapsed_layer_D = [
                tf.reshape(conv_caps_layer_D[0], semantically_convolved_activation_shape),
                tf.reshape(conv_caps_layer_D[1], semantically_convolved_pose_shape)
            ]

            aggregating_topology = TopologyBuilder().init()
            aggregating_topology.add_aggregation(semantically_collapsed_layer_D[0].get_shape().as_list()[1], [[3, 0], [3, 1]])
            aggregating_topology.add_dense_connection(semantically_collapsed_layer_D[0].get_shape().as_list()[3], capsule_count_E)
            aggregating_topology.finish()

            aggregating_capsule_layer = self.build_matrix_caps(
                conv_caps_layer_D,
                aggregating_topology,
                final_steepness_lambda,
                iteration_count,
                routing_state
            )

            final_activations = aggregating_topology.reshape_parent_map_to_linear(aggregating_capsule_layer[0])
            final_poses = aggregating_topology.reshape_parent_map_to_linear(aggregating_capsule_layer[1])

            next_routing_state = tf.get_collection("next_routing_state")

        return [final_activations, final_poses], next_routing_state, None

    def build_semantic_convolution_axial_system_architecture(self, input_layer, iteration_count, routing_state):

        with tf.name_scope('semantic_convolved_axial_system_matrix_capsule_architecture') as scope0:

            texture_patches_A = 64
            capsule_count_C = 32
            capsule_count_D = 32
            capsule_count_E = 5

            final_steepness_lambda = tf.constant(.01)

            convolution_layer_A = self.build_encoding_convolution(input_layer, 5, texture_patches_A)

            # number of capsules is defined by number of texture patches
            primary_capsule_layer_B = self.build_primary_matrix_caps(convolution_layer_A, is_axial_system=True)

            semantically_convolved_input_shape = primary_capsule_layer_B[0].get_shape().as_list()[1:3] + [8, 8]

            semantically_convolved_activation_shape = semantically_convolved_input_shape + [1]
            semantically_convolved_pose_shape = semantically_convolved_input_shape + [4, 4]

            semantically_convolved_primary_capsule_layer_B = [
                tf.reshape(primary_capsule_layer_B[0], semantically_convolved_activation_shape),
                tf.reshape(primary_capsule_layer_B[1], semantically_convolved_pose_shape)
            ]

            c_topology = TopologyBuilder().init()
            c_topology.set_is_axial_system(True)
            c_topology.add_spatial_convolution(semantically_convolved_primary_capsule_layer_B[0].get_shape().as_list()[1:3], 3, 2)
            c_topology.add_semantic_convolution(semantically_convolved_primary_capsule_layer_B[0].get_shape().as_list()[3:5], 3, 1)
            c_topology.finish()

            conv_caps_layer_C = self.build_matrix_caps(
                primary_capsule_layer_B,
                c_topology,
                final_steepness_lambda,
                iteration_count,
                routing_state
            )

            d_topology = TopologyBuilder().init()
            d_topology.set_is_axial_system(True)
            d_topology.add_spatial_convolution(conv_caps_layer_C[0].get_shape().as_list()[1:3], 3, 1)
            d_topology.add_semantic_convolution(conv_caps_layer_C[0].get_shape().as_list()[3:5], 3, 1)
            d_topology.finish()

            conv_caps_layer_D = self.build_matrix_caps(
                conv_caps_layer_C,
                d_topology,
                final_steepness_lambda,
                iteration_count,
                routing_state
            )

            semantically_collapsed_layer_D_shape = conv_caps_layer_D[0].get_shape().as_list()[0:3] + \
                                                   [np.prod(conv_caps_layer_C[0].get_shape().as_list()[3:5])]

            semantically_collapsed_layer_D_activation_shape = semantically_collapsed_layer_D_shape + [1]
            semantically_collapsed_layer_D_pose_shape = semantically_collapsed_layer_D_shape + [4, 4]

            semantically_collapsed_layer_D = [
                tf.reshape(conv_caps_layer_D[0], semantically_convolved_activation_shape),
                tf.reshape(conv_caps_layer_D[1], semantically_convolved_pose_shape)
            ]

            aggregating_topology = TopologyBuilder().init()
            aggregating_topology.set_is_axial_system(True)
            aggregating_topology.add_aggregation(semantically_collapsed_layer_D[0].get_shape().as_list()[1], [[3, 0], [3, 1]])
            aggregating_topology.add_dense_connection(semantically_collapsed_layer_D[0].get_shape().as_list()[3], capsule_count_E)
            aggregating_topology.finish()

            aggregating_capsule_layer = self.build_matrix_caps(
                conv_caps_layer_D,
                aggregating_topology,
                final_steepness_lambda,
                iteration_count,
                routing_state
            )

            final_activations = aggregating_topology.reshape_parent_map_to_linear(aggregating_capsule_layer[0])
            final_poses = aggregating_topology.reshape_parent_map_to_linear(aggregating_capsule_layer[1])

            next_routing_state = tf.get_collection("next_routing_state")

            axial_system_loss = tf.reduce_sum(tf.get_collection("unit_scale_regularization")) + tf.reduce_sum(tf.get_collection("orthogonal_regularization"))

        return [final_activations, final_poses], next_routing_state, axial_system_loss

    def build_aggregating_capsule_layer(self,
                                        input_layer_list,
                                        use_coordinate_addition,
                                        parent_count,
                                        final_steepness_lambda,
                                        iteration_count,
                                        routing_state=None):

        with tf.name_scope('aggregating_capsule_layer') as scope0:

            input_activations = input_layer_list[0]
            input_poses = input_layer_list[1]
            [batch_size, width, height, capsule_count, activation_dim] = numpy_shape_ct(
                input_activations)

            normal_layer_coordinate_addition = None
            if use_coordinate_addition:
                with tf.name_scope('coordinate_addition') as scope1:


                    width_addition = tf.constant((np.arange(width) / width).astype('float32'), shape=[1, width, 1, 1, 1, 1])
                    height_addition = tf.constant((np.arange(height) / height).astype('float32'), shape=[1, 1, height, 1, 1, 1])

                    tiled_width_addition = tf.tile(width_addition, [1, 1, height, 1, 1, 1])
                    tiled_height_addition = tf.tile(height_addition, [1, width, 1, 1, 1, 1])

                    element_axis = 5
                    # raise Exception("width_addition and height_addition must be repeated and tiled correctly instead of relying on broadcasting")
                    coordinate_addition = tf.concat([tiled_width_addition, tiled_height_addition], axis=element_axis)
                    coordinate_addition_per_capsule = tf.tile(coordinate_addition,[1, 1, 1, capsule_count, 1, 1])
                    normal_layer_coordinate_addition = self.build_cast_conv_to_normal_layer(coordinate_addition_per_capsule)
                    # coordinate_addition = tf.reshape(coordinate_addition, shape=[1, width * height * capsule_count, 1, 2])

            normal_input_activations = self.build_cast_conv_to_normal_layer(input_activations)
            normal_input_poses = self.build_cast_conv_to_normal_layer(input_poses)

            unaggregated_child_count = tf.shape(normal_input_activations)[1]

            aggregation_topology = TopologyBuilder().init()

            aggregation_topology.add_spatial_convolution([width, height], 1, 1)
            aggregation_topology.flatten_to_features

            aggregation_topology.add_dense_connection(unaggregated_child_count, parent_count)

            aggregated_capsule_layer = self.build_matrix_caps(
                [normal_input_activations, normal_input_poses],
                aggregation_topology,
                final_steepness_lambda,
                iteration_count,
                routing_state,
                normal_layer_coordinate_addition
            )

        return aggregated_capsule_layer

    def build_cast_conv_to_normal_layer(self, input_layer):
        input_shape = numpy_shape_ct(input_layer)
        [batch_size, width, height, filter_count] = input_shape[:4]
        output_dimensions = input_shape[4:]
        node_count = width * height * filter_count

        normal_shape = np.concatenate([[batch_size, node_count], output_dimensions])

        normal_layer = tf.reshape(input_layer, shape=normal_shape, name='convolutional_to_normal_layer')
        return normal_layer

    def build_encoding_convolution(self, input_layer, kernel_size, filter_count):
        #@TODO: Make the standard deviation take into account the number of outputs per node
        xavier_stddev = np.sqrt(1.0 / (kernel_size * kernel_size))
        output_layer = tf.layers.conv2d(
            input_layer,
            kernel_size=kernel_size,
            kernel_initializer=tf.initializers.truncated_normal(mean=0.0, stddev=xavier_stddev),
            filters=filter_count,
            strides=[2, 2],
            activation=tf.nn.relu,  #@TODO: test leaky relu
            name='convolutional_layer'
        )
        return output_layer

    # def build_decoding_convolution(self, decoded_input, encoded_output, parent_decoder, kernel_size, filter_count):
    #
    #     weights_shape = np.concatenate([kernel_size, [filter_count]])
    #     weights = tf.Variable(tf.truncated_normal(weights_shape))
    #
    #     reconstruction_training_layer = tf.layers.conv2d_transpose(
    #         encoded_output,
    #         weights,
    #         kernel_size=kernel_size,
    #         filters=decoded_input.shape[3].value,
    #         activation=tf.nn.relu  #@TODO: test leaky relu
    #     )
    #     reconstruction_loss = tf.abs(tf.stop_gradient(decoded_input) - reconstruction_training_layer)
    #
    #     reconstruction_prediction_layer = tf.layers.conv2d_transpose(
    #         parent_decoder,
    #         weights,
    #         kernel_size=kernel_size,
    #         filters=decoded_input.shape[3].value,
    #         activation=tf.nn.relu  #@TODO test leaky relu
    #     )
    #
    #     return reconstruction_prediction_layer, reconstruction_loss

    def build_cast_conv_to_pose_layer(self, input_layer, input_filter_count,
                                  simplify_to_axial_system=False):

        elements_per_pose = 16
        pose_axis_element_count = 4

        if simplify_to_axial_system:
            elements_per_pose = 12
            pose_axis_element_count = 3

        pose_element_layer = tf.layers.conv2d(
            input_layer,
            kernel_size=[1, 1],
            filters=input_filter_count * elements_per_pose,
            activation=None,
            name='extract_poses'
        )

        pose_shape = pose_element_layer.get_shape()

        pose_layer = tf.reshape(
            pose_element_layer,
            shape=[-1, pose_shape[1], pose_shape[2], input_filter_count, 4, pose_axis_element_count],
            name='reshape_poses'
        )

        pose_axis_vector_axis = -2
        pose_axis_element_axis = -1
        if simplify_to_axial_system:

            pose_layer = self.append_axial_system_row(pose_layer)

            axial_system_list = tf.reshape(pose_layer, shape=[-1, 4, 4])

            TopologyBuilder.add_orthogonal_unit_axis_loss(axial_system_list)

        return pose_layer

    def append_axial_system_row(self, axial_system_tensor):
        pose_axis_vector_axis = -2
        pose_axis_element_axis = -1

        axial_system_tensor_shape = tf.shape(axial_system_tensor)
        tileable_axial_system_shape = np.ones([axial_system_tensor_shape.get_shape()[0].value])
        bottom_row_axial_system = [0.0, 0.0, 0.0, 1.0]
        tileable_axial_system_shape[pose_axis_vector_axis] = len(bottom_row_axial_system)
        tileable_bottom_row_axial_system = tf.reshape(tf.constant(bottom_row_axial_system),
                                                     shape=tileable_axial_system_shape)
        tiling_shape = tf.concat([axial_system_tensor_shape[0:(pose_axis_vector_axis)], tf.constant([1, 1])], axis=0)
        tiled_bottom_row_axial_system = tf.tile(tileable_bottom_row_axial_system, tiling_shape)

        axial_system_tensor = tf.concat([axial_system_tensor, tiled_bottom_row_axial_system], axis=pose_axis_element_axis)

        return axial_system_tensor

    def build_cast_conv_to_activation_layer(self, input_layer, input_filter_count):

        raw_activation_layer = tf.layers.conv2d(
            input_layer,
            kernel_size=[1, 1],
            filters=input_filter_count,
            activation=tf.nn.sigmoid,
            name='extract_activations'
        )

        # by adding two extra dimensions both em routing and the linking of layers becomes a lot easier
        target_shape = np.concatenate([raw_activation_layer.get_shape(), [tf.Dimension(1)]])
        for i in range(target_shape.shape[0]):
            target_shape[i] = target_shape[i] if target_shape[i].value is not None else -1

        activation_layer = tf.reshape(raw_activation_layer, shape=target_shape, name='reshape_activations')

        return activation_layer

    def build_primary_matrix_caps(self, input_layer, is_axial_system=False):

        with tf.name_scope('primary_matrix_capsules') as scope0:

            filter_axis = 3

            input_filter_count = input_layer.get_shape()[filter_axis]

            activation_layer = self.build_cast_conv_to_activation_layer(input_layer, input_filter_count)
            pose_layer = self.build_cast_conv_to_pose_layer(input_layer, input_filter_count, is_axial_system)

        return [activation_layer, pose_layer]

    def build_convolutional_capsule_layer(self,
                                          input_layer_list,
                                          kernel_size,
                                          stride,
                                          parent_count,
                                          final_steepness_lambda,
                                          iteration_count,
                                          routing_state
                                          ):

        return self.build_convolution_of(
            input_layer_list,
            kernel_size,
            stride,
            lambda activations, poses:
                self.build_matrix_caps(
                    parent_count,
                    final_steepness_lambda,
                    iteration_count,
                    activations,
                    poses,
                    routing_state
                )
        )

    def build_convolution_of(self, input_feature_image_layer_list, kernel_size: int, stride: int, filter_layer_constructor):

        with tf.name_scope('convolution_layer') as scope0:

            # first collapse the nasty complicated feature dimensions into 1 dimension with reshape: [batch, width, height, feature_dimensions*]
            # use tf.extract_image_patches to create [batch, patch_x, patch_y, kernel_width * kernel_height * collapsed_feature_dimensions]
            # collapse batch patch_x and patch_y into one dimension batch_patches:
            # [batch * patch_x * patch_y, kernel_width * kernel_height * collapsed_feature_dimensions]
            # decollapse feature dimensions to [batch * patch_x * patch_y, kernel_width * kernel_height, feature_dimensions*]
            # apply filter constructor
            # decollapse batch_patches back into [batch, patch_x, patch_y, filter, output_dimensions]

            # @TODO: fix ugly hack of getting patch_row_width, and patch_column_height from loop
            batch_size, patch_row_width, patch_column_height = None, None, None
            argument_list = []

            with tf.name_scope('image_patch_construction') as scope1:

                for i in range(len(input_feature_image_layer_list)):
                    input_feature_image_layer = input_feature_image_layer_list[i]

                    input_feature_image_layer_shape = numpy_shape_ct(input_feature_image_layer)

                    batch_size = input_feature_image_layer_shape[0]
                    width = input_feature_image_layer_shape[1]
                    height = input_feature_image_layer_shape[2]
                    feature_count = input_feature_image_layer_shape[3]
                    feature_dimensions = input_feature_image_layer_shape[4:] # feature dimensions are the EXTRA structure beyond the filter dimension; i.e. 1 for activation and 4x4 for pose matrices
                    collapsed_feature_dimensions = feature_count * np.product(feature_dimensions)

                    collapsed_feature_image = tf.reshape(input_feature_image_layer, shape=[batch_size, width, height, collapsed_feature_dimensions])

                    # IMAGE PATCH EXTRACTION
                    collapsed_feature_image_patches = tf.extract_image_patches(collapsed_feature_image,
                                                                               ksizes=[1, kernel_size, kernel_size, 1],
                                                                               strides=[1, stride, stride, 1],
                                                                               rates=[1, 1, 1, 1],
                                                                               padding='VALID')

                    [not_used, patch_row, patch_column, collapsed_image_patch_dimensions] = numpy_shape_ct(collapsed_feature_image_patches)

                    assert (numpy_shape_ct(collapsed_feature_image_patches) == np.array([batch_size,
                                                                           (width - (kernel_size - 1)) / stride,
                                                                           (height - (kernel_size - 1)) / stride,
                                                                           collapsed_image_patch_dimensions])).all()

                    # PREPARE IMAGE PATCH PROCESSING

                    # Each patch must become a row, and all features must be expanded again into their appropriate dimensions.
                    # Dimensions can be seen as a tree structure; for each dimension a value selects a branch.
                    # Each level of the tree can be seen as an interface; both the tree structure before the level and after
                    # the level can be restructured by mapping to a different number of dimensions, as long as the number of
                    # elements at that specific level remains the same.
                    # The processing layer requires data in its original feature dimensions, without the structure imposed
                    # by the kernels. Restructuring the input payload into kernel_size * kernel_size * feature_count, feature_dimensions
                    # will therefore only restructure the input from a list of values into a tree structure, but the roots of
                    # that tree structure still form the same list as the original data. This is the list indexed by batch,
                    # patch_row and patch_column.
                    # This means that each root describes a single input for the processing layer. I.e. by passing -1 to reshape
                    # the dimensions batch, patch_row and patch_column collapse into a new dimension (patch_batch) corresponding
                    # to the records that must be processed.
                    # The following commented lines show this process step by step in reverse order (though this is not possible in tensor flow
                    # because batch_size is not yet known):
                    #
                    # merge all patches by converting batches and patches to patch_batches
                    # patch_batch_size = batch_size * patch_row * patch_column # WON'T WORK BECAUSE WE DON'T KNOW batch_size, but -1 can be passed to shape
                    # collapsed_feature_image_patch_batches = tf.reshape(collapsed_feature_image_patches, shape=[-1, collapsed_image_patch_dimensions])
                    # feature_image_patch_batches = tf.reshape(collapsed_feature_image_patch_batches,
                    #     np.concatenate([[patch_batch_size, kernel_size * kernel_size * feature_count], feature_dimensions])
                    # )
                    # this line does the above all in one go, as -1 makes reshape compute batch_size * patch_row * patch_column
                    # implicitly.
                    feature_image_patch_batches = tf.reshape(collapsed_feature_image_patches,
                        np.concatenate([[-1, kernel_size * kernel_size * feature_count], feature_dimensions])
                    )

                    argument_list.append(feature_image_patch_batches)

            # PROCESS IMAGE PATCHES

            with tf.name_scope('image_patch_to_feature') as scope3:

                patch_batch_results = filter_layer_constructor(*argument_list)
                results = []

            with tf.name_scope('feature_image_construction') as scope3:

                for filtered_image_patch_batches in patch_batch_results:
                    # RESHAPE TO PATCH X AND PATCH Y

                    filtered_output_shape = numpy_shape_ct(filtered_image_patch_batches)[1:]

                    filtered_batch_x_y_shape = [batch_size, patch_row, patch_column]

                    filtered_image_batch_shape = np.concatenate([filtered_batch_x_y_shape, filtered_output_shape])

                    filtered_image_batch = tf.reshape(filtered_image_patch_batches, shape=filtered_image_batch_shape)
                    results.append(filtered_image_batch)

        return results

    # def build_children_of_potential_parent_pose_layer(self,
    #                                                   child_poses,  # [batch, x, y, f0 ... fn, m0, m1]
    #                                                   parent_capsule_feature_count,
    #                                                   spatial_kernel_size,
    #                                                   spatial_stride):
    #     with tf.name_scope('potential_parent_pose_layer') as scope:
    #         # retrieve relevant dimensions
    #         child_poses_shape = numpy_shape_ct(child_poses)
    #         batch_size = tf.shape(child_poses)[0]
    #         child_row_count = child_poses_shape[1]
    #         child_column_count = child_poses_shape[2]
    #         child_feature_count = child_poses_shape[3]
    #
    #         # UNFORTUNATELY tf.matmul DOES NOT IMPLEMENT BROADCASTING, SO THE CODE BELOW DOES SO MANUALLY
    #
    #         # A pose transform matrix exists from each child to each parent.
    #         # Weights with bigger stddev improve numerical stability
    #         pose_transform_weights = tf.Variable(
    #             tf.truncated_normal([1, child_capsule_count, parent_capsule_feature_count, 4, 4], mean=0.0, stddev=1.0),
    #             dtype=tf.float32, name='pose_transform_weights')
    #
    #         # Potential parent poses must be predicted for each batch row. So weights must be copied for each batch row.
    #         pose_transform_weights_copied_for_batch = tf.tile(pose_transform_weights, [batch_size, 1, 1, 1, 1], name='pose_transform_tiled')
    #
    #         ## Because the child poses are used to produce the pose of every potential parent, a copy is necessary for
    #         ## predicting the potential parent poses
    #         ## A column after the child capsule column is added,
    #         # child_poses_with_copy_column = tf.reshape(child_poses, shape=[batch_size, child_capsule_count, 1, 4, 4])
    #
    #         # above code redundant now that broadcasting dims are part of specs
    #
    #         # so output can be copied for each parent
    #         child_poses_copied_for_parents = tf.tile(child_poses, [1, 1, parent_capsule_feature_count, 1, 1], name='child_pose_tiled')
    #
    #         # child poses are now copied for each potential parent, and child to parent tranforms are copied for each batch
    #         # row, resulting in two tensors; a tensor containing pose matrices in its last two indices, and one
    #         # containing pose transformations from each child to each potential parent pose
    #         #
    #         # matmul will now iterate over batch, child capsule, and parent capsule in both tensors and multiply the child
    #         # pose with the pose transform to output the pose of a potential parent
    #         potential_parent_poses = tf.matmul(child_poses_copied_for_parents, pose_transform_weights_copied_for_batch, name='child_transformation')
    #
    #     return potential_parent_poses

    # def cast_pose_matrices_to_vectors(self, pose_matrices):  # takes the last two matrix indices and reshapes into vector indices
    #     current_shape = pose_matrices.get_shape()
    #     new_shape = current_shape[:-1]
    #     new_shape[-1] = current_shape[-1] * current_shape[-2]
    #
    #     return tf.reshape(pose_matrices, shape=new_shape)
    #
    # def cast_pose_vectors_to_matrices(self, pose_vectors):
    #     current_shape = pose_vectors.get_shape()
    #     matrix_diagonal_length = math.sqrt(current_shape[-1])
    #     new_shape = np.concatenate([current_shape[:-1], [matrix_diagonal_length], [matrix_diagonal_length]])
    #     new_shape[-1] = current_shape[-1] * current_shape[-2]

    def build_parent_assembly_layer(self,
                                    child_activations,
                                    children_per_potential_parent_pose_vectors,  # [batch, child, parent, pose_vector]
                                    final_steepness_lambda,
                                    iteration_count, # em routing
                                    routing_state,
                                    topology):

        with tf.name_scope('parent_assembly_layer') as scope0:

            parent_axis = 2
            [_, child_count, parent_count, pose_element_count] = numpy_shape_ct(children_per_potential_parent_pose_vectors)
            batch_size = tf.shape(children_per_potential_parent_pose_vectors)[0]

            beta_u = tf.Variable(tf.truncated_normal([1, 1, parent_count, 1]), name='beta_u')
            beta_a = tf.Variable(tf.truncated_normal([1, 1, parent_count, 1]), name='beta_a')

            initial_child_parent_assignment_weights = tf.ones([batch_size, child_count, parent_count, 1], tf.float32) / float(parent_count)

            if routing_state is not None:
                initial_child_parent_assignment_weights = tf.constant(next(routing_state), name='initial_routing_state')

            child_parent_assignment_weights = initial_child_parent_assignment_weights

            with tf.name_scope('expectation_maximization') as scope1:
                for i in range(iteration_count + 1):
                    steepness_lambda = final_steepness_lambda * (1.0 - tf.pow(0.95, tf.cast(i + 1, tf.float32)))

                    # select child parent assignments based on child activity
                    # [batch, child, parent, 1]
                    active_child_parent_assignment_weights = child_parent_assignment_weights * child_activations
                    active_child_parent_assignment_weights = tf.identity(active_child_parent_assignment_weights,
                                                                         name='active_child_parent_assignment_weights')

                    #@TODO make arguments match convolution topology
                    parent_activations, likely_parent_pose, likely_parent_pose_deviation, likely_parent_pose_variance =\
                        self.estimate_parents_layer(
                            active_child_parent_assignment_weights, #@TODO limit to only children of parent
                            children_per_potential_parent_pose_vectors, #@TODO limit to only children of parent
                            beta_u,
                            beta_a,
                            steepness_lambda
                        )

                    # the final iteration of this loop will needlessly produce an extra set of nodes using below code,
                    # BUT, this doesn't matter because what is actually computed is determined by dependency analysis of the
                    # graph and the nodes produced below in the final loop are not required for anything and thus ignored
                    #@TODO make arguments match convolution topology
                    child_parent_assignment_weights = self.estimate_children_layer(
                        parent_activations,
                        likely_parent_pose,
                        likely_parent_pose_variance,
                        children_per_potential_parent_pose_vectors, #@TODO limit to only parents of children
                        topology
                    )
                    #@TODO invert child parent assignment weights to contain children of parents
                    # rather than parents of children

            tf.summary.histogram('final_routing', child_parent_assignment_weights)

            tf.add_to_collection('next_routing_state', child_parent_assignment_weights)

            output_parent_activations = parent_activations
            output_parent_poses = likely_parent_pose

        return output_parent_activations, output_parent_poses

    def estimate_parents_layer(self, # m step
                               active_child_parent_assignment_weights,  # [batch, child, parent, 1]
                               potential_parent_pose_vectors,  # [batch, child, parent, pose_vector]
                               beta_u,  # [1, 1, parent, 1] @TODO double check correct dimensions!
                               beta_a,  # [1, 1, parent, 1] @TODO double check correct dimensions!
                               steepness_lambda  # scalar
                               ):

        with tf.name_scope('estimate_parents_layer') as scope:
            [batch_count, child_count, parent_count, pose_element_count] = numpy_shape_ct(potential_parent_pose_vectors)

            assert (numpy_shape_ct(beta_u) == np.array([1, 1, parent_count, 1])).all()
            assert (numpy_shape_ct(beta_a) == np.array([1, 1, parent_count, 1])).all()

            batch_axis = 0
            child_axis = 1
            parent_axis = 2
            pose_element_axis = 3  # @TODO: double check if this is the correct axis
            # select child parent assignments based on child activity

            assert (numpy_shape_ct(active_child_parent_assignment_weights)[1:] == np.array([batch_count, child_count, parent_count, 1])[1:]).all()

            # [batch, 1, parent, 1]
            total_active_child_parent_assignment_per_parent = tf.reduce_sum(active_child_parent_assignment_weights,
                                                                     axis=child_axis,
                                                                     keepdims=True,
                                                                     name='total_active_child_parent_assignment_per_parent')
            assert (numpy_shape_ct(total_active_child_parent_assignment_per_parent)[1:] == np.array([batch_count, 1, parent_count, 1])[1:]).all()

            # scale down child parent assignment weights to proportions
            # [batch, child, parent, 1]
            child_proportion_of_parent = active_child_parent_assignment_weights \
                                         / (total_active_child_parent_assignment_per_parent + sys.float_info.epsilon)
            assert (numpy_shape_ct(child_proportion_of_parent)[1:] == np.array([batch_count, child_count, parent_count, 1])[1:]).all()

            # Assemble the most probable parent pose
            # by summing the potential_parent_pose_vectors weighted by proportional contributions of the children;
            # This is the mu of the parent in the paper
            # [batch, 1, parent, likely_pose_element]
            likely_parent_pose = tf.reduce_sum(
                potential_parent_pose_vectors * child_proportion_of_parent,
                axis=child_axis, keepdims=True, name='likely_parent_pose'
            )
            assert (numpy_shape_ct(likely_parent_pose)[1:] == np.array([batch_count, 1, parent_count, pose_element_count])[1:]).all()

            # @TODO DOUBLE CHECK IF tf.square DOES WHAT THE MATRIX CAPS PAPER DEMANDS!!!
            # this is the sigma of the parent in the paper
            # [batch, 1, parent, likely_pose_element_variance]
            likely_parent_pose_variance = tf.reduce_sum(
                tf.square(potential_parent_pose_vectors - likely_parent_pose) * child_proportion_of_parent,
                axis=child_axis, keepdims=True, name='likely_parent_pose_variance'
            )  + sys.float_info.epsilon
            assert (numpy_shape_ct(likely_parent_pose_variance)[1:] == np.array([batch_count, 1, parent_count, pose_element_count])[1:]).all()

            # [batch, 1, parent, likely_pose_element_standard_deviation]
            likely_parent_pose_deviation = tf.sqrt(likely_parent_pose_variance) + sys.float_info.epsilon
            assert (numpy_shape_ct(likely_parent_pose_deviation)[1:] == np.array([batch_count, 1, parent_count, pose_element_count])[1:]).all()

            # inactivity parent per for child capsules
            # [batch, 1, parent, 1]
            sum_cost_h = tf.reduce_sum(
                (beta_u + tf.log(likely_parent_pose_deviation + sys.float_info.epsilon)) * total_active_child_parent_assignment_per_parent,
                axis=pose_element_axis,
                keepdims=True,
                name='inactive_cost'
            )
            assert (numpy_shape_ct(sum_cost_h)[1:] == np.array([batch_count, 1, parent_count, 1])[1:]).all()

            cost_a = beta_a
            # [batch, 1, parent, 1]
            cost_balance_activity = cost_a - sum_cost_h
            cost_balance_activity = tf.identity(cost_balance_activity, name='cost_balance_activity')
            assert (numpy_shape_ct(cost_balance_activity)[1:] == np.array([batch_count, 1, parent_count, 1])[1:]).all()

            # [batch, 1, parent, 1]
            parent_activations = tf.nn.sigmoid(steepness_lambda * cost_balance_activity)
            assert (numpy_shape_ct(parent_activations)[1:] == np.array([batch_count, 1, parent_count, 1])[1:]).all()

        return parent_activations, likely_parent_pose, likely_parent_pose_deviation, likely_parent_pose_variance

    def estimate_children_layer(self,  # e step
                                parent_activations,  # [batch, 1, parent, 1]
                                likely_parent_pose,  # [batch, 1, parent, likely_pose_element]
                                likely_parent_pose_variance,  # [batch, 1, parent, likely_pose_element_variance]
                                potential_parent_pose_vectors, # [batch, child, parent, pose_element]
                                topology):

        with tf.name_scope('estimate_children_layer') as scope:

            [batch_count, child_count, parent_count, pose_element_count] = numpy_shape_ct(potential_parent_pose_vectors)

            assert (numpy_shape_ct(parent_activations)[1:] == np.array([batch_count, 1, parent_count, 1])[1:]).all()
            assert (numpy_shape_ct(likely_parent_pose)[1:] == np.array([batch_count, 1, parent_count, pose_element_count])[1:]).all()
            assert (numpy_shape_ct(likely_parent_pose_variance)[1:] == np.array([batch_count, 1, parent_count, pose_element_count])[1:]).all()

            batch_axis = 0
            child_axis = 1
            parent_axis = 2
            pose_element_axis = 3

            # [batch, 1, parent, 1]
            # factor = tf.reciprocal(
            #     tf.sqrt(tf.reduce_prod(likely_parent_pose_variance, axis=pose_element_axis, keepdims=True) * 2.0 * math.pi) + sys.float_info.epsilon
            # )

            divisor = tf.sqrt(tf.reduce_prod(likely_parent_pose_variance, axis=pose_element_axis, keepdims=True) * 2.0 * math.pi + sys.float_info.epsilon) + sys.float_info.epsilon
            divisor = tf.identity(divisor, name='divisor')
            # log_factor = (tf.reduce_sum(-tf.log(likely_parent_pose_variance * 2.0 * np.pi), axis=pose_element_axis, keepdims=True)) / 2.0

            # assert (numpy_shape_ct(factor)[1:] == np.array([batch_count, 1, parent_count, 1])[1:]).all()

            # [batch, child, parent, pose_element] @TODO is likely_parent_pose  broadcasted correctly
            potential_parent_pose_variance = tf.square(potential_parent_pose_vectors - likely_parent_pose)
            potential_parent_pose_variance = tf.identity(potential_parent_pose_variance, name='potential_parent_pose_variance')

            assert (numpy_shape_ct(potential_parent_pose_variance)[1:] == np.array([batch_count, child_count, parent_count, pose_element_count])[1:]).all()

            # [batch, child, parent, 1]
            power = -tf.reduce_sum(
                potential_parent_pose_variance / (likely_parent_pose_variance * 2 + sys.float_info.epsilon),
                axis=pose_element_axis, keepdims=True)

            power = tf.identity(power, name='power')

            assert (numpy_shape_ct(power)[1:] == np.array([batch_count, child_count, parent_count, 1])[1:]).all()

            # [batch, child, parent, 1]
            # parent_probability_per_child = factor * tf.exp(power)
            # assert (numpy_shape_ct(parent_probability_per_child)[1:] == np.array([batch_count, child_count, parent_count, 1])[1:]).all()

            parent_probability_per_child = tf.exp(power) / (divisor + sys.float_info.epsilon)
            parent_probability_per_child = tf.identity(parent_probability_per_child, name='parent_probability_per_child')

            # [batch, child, parent, 1]
            active_parent_probability_per_child = parent_probability_per_child * parent_activations
            active_parent_probability_per_child = tf.identity(active_parent_probability_per_child, name='active_parent_probability_per_child')

            assert (numpy_shape_ct(active_parent_probability_per_child)[1:] == np.array([batch_count, child_count, parent_count, 1])[1:]).all()

            # [batch, child, parent, 1]
            child_parent_assignment_weights = active_parent_probability_per_child \
                / (topology.replace_kernel_elements_with_sum_of_child(active_parent_probability_per_child) + sys.float_info.epsilon)

            child_parent_assignment_weights = tf.identity(child_parent_assignment_weights, name='child_parent_assignment_weights')

            assert (numpy_shape_ct(child_parent_assignment_weights)[1:] == np.array([batch_count, child_count, parent_count, 1])[1:]).all()

        return child_parent_assignment_weights

    def build_matrix_caps(
            self,
            input_layer,
            topology,
            final_steepness_lambda,
            iteration_count,
            routing_state,
            convolution_coordinates=None):

        with tf.name_scope('matrix_capsule_layer') as scope:
            child_activation_vector, child_pose_layer = topology.map_children_to_linear_kernel_linear_parent(input_layer[0]), topology.map_children_to_linear_kernel_linear_parent(input_layer[1])

            potential_parent_poses = topology.linearized_potential_parent_poses(input_layer[1])

            [batch_size, child_count, parent_count, pose_matrix_width, pose_matrix_height] = numpy_shape_ct(potential_parent_poses)

            pose_element_count = pose_matrix_width * pose_matrix_height
            potential_parent_pose_vectors = tf.reshape(potential_parent_poses,
                [batch_size, child_count, parent_count, pose_element_count]
            )
            pose_element_axis = 3

            # @TODO the code below is now part of topology, double check if it can be removed
            # if convolution_coordinates is not None:
            #     dynamic_batch_size = tf.shape(potential_parent_poses)[0]
            #     convolution_coordinates_tiled_for_parents = tf.tile(convolution_coordinates, [dynamic_batch_size, 1, parent_count, 1])
            #
            #     potential_parent_pose_vectors = tf.concat(
            #         [potential_parent_pose_vectors, convolution_coordinates_tiled_for_parents],
            #         axis=pose_element_axis)

            parent_activations, parent_pose_vectors = self.build_parent_assembly_layer(
                child_activation_vector,
                potential_parent_pose_vectors,
                final_steepness_lambda,
                iteration_count,
                routing_state,
                topology
            )

            linear_parent_as_child_activations = tf.reshape(parent_activations, shape=[batch_size, parent_count, 1])
            linear_parent_as_child_pose_matrices = tf.reshape(parent_pose_vectors, shape=[batch_size, parent_count, pose_matrix_width, pose_matrix_height])

            mapped_parent_as_child_activations = topology.reshape_parents_to_map(linear_parent_as_child_activations)
            mapped_parent_as_child_pose_matrices = topology.reshape_parents_to_map(linear_parent_as_child_pose_matrices)

            tf.summary.histogram('activations', mapped_parent_as_child_activations)
            tf.summary.histogram('poses', mapped_parent_as_child_pose_matrices)
            tf.summary.histogram('poses_determinant', tf.linalg.det(mapped_parent_as_child_pose_matrices))

        return mapped_parent_as_child_activations, mapped_parent_as_child_pose_matrices

    def progress_percentage_node(self, batch_size, full_example_count, is_training):
        example_counter = tf.Variable(tf.constant(float(0)), trainable=False, dtype=tf.float32)

        example_counter = tf.cond(is_training,
                                  lambda: example_counter.assign(example_counter + batch_size, use_locking=True),
                                  lambda: example_counter)

        progress_percentage = example_counter / tf.cast(full_example_count, tf.float32)
        progress_percentage = tf.cond(tf.logical_and(is_training, progress_percentage <= 1), lambda: progress_percentage, lambda: tf.constant(1.0))

        progress_percentage = tf.identity(progress_percentage, name='progress_percentage')

        return progress_percentage, example_counter

    def changing_value(self, start, finish, progress_node):
        total_change = finish - start

        current_value = tf.constant(start) + tf.constant(total_change) * progress_node

        return current_value

    def increasing_value(self, start, increase, is_training):
        value = tf.Variable(tf.constant(float(start)), trainable=False, dtype=tf.float32)
        increase = tf.constant(float(increase), dtype=tf.float32)
        value = tf.cond(is_training, lambda: value.assign(value + increase), lambda: value)

        return value




    class TopologyStrategy:
        def potential_parent_pose_input(self, input):
            raise NotImplemented()

        def potential_parent_pose_weights(self, input, parent_feature_count):
            raise NotImplemented()

        def children_of_parents(self, input):
            raise NotImplemented()

        def parents_of_children(self, input):
            raise NotImplemented()

    class TopologyBuilder:

        def __init__(self):
            self.input_to_parent_kernel_indices = np.array([])

            self.is_input_dimension = []  # python LIST, True (1) for 1nput en False (0) for 0utput

            self.weight_shape = []  # must be python LIST and definitely NOT numpy array. We want + for concatenation.

            self.weight_tiling = []  # must be python LIST and definitely NOT numpy array. We want + for concatenation.

            self.input_tiling = []  # must be python LIST and definitely NOT numpy array. We want + for concatenation.

        def append_spatial_convolution(self, spatial_shape, kernel_size, stride):

            kernel_patch_indices = self.create_spatial_convolution(spatial_shape, kernel_size, stride)

            self.expand_by_convolution(kernel_patch_indices)

            self.weight_shape += (kernel_patch_indices.shape[[0, 1]] + [1, 1])

            self.weight_tiling += ([1, 1] + kernel_patch_indices.shape[[2, 3]])

            self.input_tiling += [1, 1, 1, 1]

            self.is_input_dimension += [True, True, False, False]

        def append_semantic_convolution(self, semantic_shape, kernel_size, stride):

            kernel_patch_indices = self.create_spatial_convolution(semantic_shape, kernel_size, stride, make_toroid=True)

            self.expand_by_convolution(kernel_patch_indices)

            self.weight_shape += kernel_patch_indices.shape[:-1]  # [kernel_size, kernel_size, semantic_row, semantic_column]

            self.weight_tiling += [1, 1, 1, 1]

            self.input_tiling += [1, 1, 1, 1]

            self.is_input_dimension += [True, True, False, False]

        def append_fully_connected(self, input_features, output_features):

            #@TODO implement how input is tiled as a consequence, and how

            self.weight_shape += [input_features, output_features]

            self.weight_tiling += [1, 1]

            self.input_tiling += [1, output_features]  # input features are taken directly from the input

            self.is_input_dimension += [True, False]


        def create_spatial_convolution(self, children_shape, kernel_size, stride, make_toroid=False):

            [child_row_count, child_column_count] = children_shape

            # create index array with shape of [kernel_row, kernel_column, parent_row, parent_column]

            # For each kernel row/column index and parent row/column index the correct child row/column index must
            # be in the last dimension of the index array passed to gather_nd. The correct row/column index can be
            # seen as a 2D axial system with an origin defined by patch row/column offset, and a further translation
            # along the kernel's axes defined by the relative kernel row/column position. Note the practical fact
            # that the row axes and column axes are entirely orthogonal and thus independent.

            # Therefore a practical way to accomplish all these translations is to define the translation for each
            # dimension (row or column) independently, and then add all coordinates for each dimension using broadcasting;
            #
            # Due to column and row translation independence; patch row offset + kernel row position = child row
            # position, and the same for column positions.
            #
            # By indexing the n -1 indices of the indices argument to gather_nd by [kernel_row, kernel_column, parent_row, parent_column]
            # and  putting all the translation data into arrays as described below broadcasting applies each translation
            # for every dimension correctly;

            # [kernel_row, 1, 1, 1, [0,1]] = [kernel_row_translation, 0] +  # kernel_row translation
            # [1, 1, parent_row, 1, [0,1]] = [parent_row_offset, 0]         # parent_row_offset translation
            # [1, kernel_column, 1, [0,1]] = [0, kernel_column_translation] + # kernel_column translation
            # [1, 1, 1, parent_column_translation, [0, 1]] = [0, parent_column_offset]

            # compute relevant data
            valid_kernel_space = (kernel_size - 1)

            if make_toroid:
                # NOTE: any index exceeding child_row/column_count will be wrapped round by subtracting child_row/column_count
                # using a modulo division later on
                valid_kernel_space = 0


            parent_row_count = int((child_row_count - valid_kernel_space) / stride)
            parent_column_count = int((child_column_count - valid_kernel_space) / stride)

            # translation data
            patch_row_offsets    = np.arange(0, parent_row_count) * stride
            patch_column_offsets =  np.arange(0, parent_column_count) * stride

            kernel_row_deltas = np.arange(kernel_size)
            kernel_column_deltas = np.arange(kernel_size)

            # data translatable through broadcasting

            broadcastable_patch_row_offsets = np.reshape(patch_row_offsets, [1, 1, parent_row_count, 1, 1])
            broadcastable_patch_row_offsets = np.concatenate([broadcastable_patch_row_offsets, np.zeros(broadcastable_patch_row_offsets.shape)], axis=-1)

            broadcastable_patch_column_offsets = np.reshape(patch_column_offsets, [1, 1, 1, parent_column_count, 1])
            broadcastable_patch_column_offsets = np.concatenate([np.zeros(broadcastable_patch_column_offsets.shape), broadcastable_patch_column_offsets], axis=-1)

            broadcastable_kernel_row_deltas = np.reshape(kernel_row_deltas, [kernel_size, 1, 1, 1, 1])
            broadcastable_kernel_row_deltas = np.concatenate([broadcastable_kernel_row_deltas, np.zeros(broadcastable_kernel_row_deltas.shape)], axis=-1)

            broadcastable_kernel_column_deltas = np.reshape(kernel_column_deltas, [1, kernel_size, 1, 1, 1])
            broadcastable_kernel_column_deltas = np.concatenate([np.zeros(broadcastable_kernel_column_deltas.shape), broadcastable_kernel_column_deltas], axis=-1)

            kernel_patch_indices = broadcastable_patch_row_offsets + broadcastable_kernel_row_deltas + \
                broadcastable_patch_column_offsets + broadcastable_kernel_column_deltas

            if make_toroid:
                kernel_patch_indices[:, :, :, :, 0] = kernel_patch_indices[:, :, :, :, 0] % child_row_count
                kernel_patch_indices[:, :, :, :, 1] = kernel_patch_indices[:, :, :, :, 0] % child_column_count

            return kernel_patch_indices


        def expand_by_convolution(self, kernel_patch_indices):
            new_patch_address_dimensions = len(kernel_patch_indices.shape) - 1
            new_index_dimensions = kernel_patch_indices.shape[-1]

            current_patch_address_dimensions = len(self.input_to_parent_kernel_indices.shape) - 1
            current_patch_index_dimensions = self.input_to_parent_kernel_indices.shape[-1]

            zero_padded_kernel_patch_indices = np.concatenate(  # prepend zeros to allow addition of kernel_patch_indices to current_patch_indices
                [
                    np.zeros(new_patch_address_dimensions + [current_patch_index_dimensions]),
                    kernel_patch_indices
                ] , axis=-1)

            zero_padded_current_patch_indices = np.concatenate(  # prepend zeros to allow addition of kernel_patch_indices to current_patch_indices
                [
                    self.input_to_parent_kernel_indices,
                    np.zeros(current_patch_address_dimensions + [new_index_dimensions])
                ] , axis=-1)

            broadcastable_kernel_patch_shape = np.concatenate([np.ones([current_patch_address_dimensions]), zero_padded_kernel_patch_indices.shape])
            broadcastable_current_patch_shape = \
                np.concatenate([self.input_to_parent_kernel_indices.shape[:-1], np.ones([new_patch_address_dimensions], zero_padded_current_patch_indices.shape[-1])])

            broadcastable_kernel_patch = np.reshape(zero_padded_kernel_patch_indices, broadcastable_kernel_patch_shape)
            broadcastable_current_patch = np.reshape(zero_padded_current_patch_indices, broadcastable_current_patch_shape)

            concatentated_convolution = broadcastable_current_patch + broadcastable_kernel_patch  # new convolution is broadcast to each existing convolution

            self.input_to_parent_kernel_indices = concatentated_convolution

        def _invert_input_to_parent_kernel_indices(self, input_shape):
            #@TODO remove numpy zeros dense array and just create tf.SparseTensor directly
            parent_kernel_to_input = np.zeros(self.input_to_parent_kernel_indices.shape + input_shape)

            parent_kernel_index_list = np.indices(self.input_to_parent_kernel_indices.shape).reshape([2, -1]).transpose()

            connection_index_list = np.concatenate([parent_kernel_index_list, self.input_to_parent_kernel_indices[parent_kernel_index_list.transpose()]])

            parent_kernel_to_input[connection_index_list.transpose()] = 1

            return parent_kernel_to_input



        # def _gather_nd_ignore_batch(self, input_layer, indices):
        #     dimension_count = len(input_layer.get_shape())
        #
        #     input_payload_dimensions = range(1, dimension_count)
        #     input_batch_dimension = [0]
        #     to_gather_transpose_ = input_payload_dimensions + input_batch_dimension
        #
        #     input_layer_transposed_
        #
        #     tf.gather_nd()

        def project_input_to_kernels(self, input_layer):

            pass

        def reduce_sum_kernels_to_input(self, kernels_to_parent):
            pass

        def flatten_kernel_parent_map(self, kernel_to_parent):
            pass

        def unflatten_kernel_parent_map(self, flattened_kernel_to_parent):
            pass

        def project_sum_per_child_from_kernel_map_to_kernel_map(self, kernel_to_parent):
            input_sum = self.reduce_sum_kernels_to_input(kernel_to_parent)
            child_sum_in_kernel_to_parent = self.project_input_to_kernels(input_sum)
            return child_sum_in_kernel_to_parent


    class Convolution:
        def __init__(self, shape, kernel, stride):
            mapping = self.generate_2D_convolution_indices(shape, kernel, stride)
            reverse_mapping = self.reverse_2D_convolution_indices(mapping, shape)

            super(self, mapping, reverse_mapping)

        def generate_2D_convolution_indices(self, shape, kernel_size, stride):

            # generate indices for kernel at initial offset

            kernel_x = np.repeat(np.arange(kernel_size), kernel_size).reshape([1, 1, -1, 1])
            kernel_y = np.tile(np.arange(kernel_size), kernel_size).reshape([1, 1, 1, -1])

            # for each patch generate translations of kernel

            patch_row_ids = np.arange(0, shape[0] - kernel_size + 1, stride).reshape([-1, 1, 1, 1])
            patch_column_ids = np.arange(0, shape[1] - kernel_size + 1, stride).reshape([1, -1, 1, 1])

            patch_row_translation = np.tile(patch_row_ids, [1, patch_column_ids.shape[1], 1, 1])
            patch_column_translation = np.tile(patch_column_ids, [patch_row_ids.shape[0], 1, 1, 1])

            # for each translation create a kernel translation. Broadcasting should combine all coordinates correctly
            patch_row_x_kernel = patch_row_translation + kernel_x
            patch_column_y_kernel = patch_column_translation + kernel_y

            patch_to_kernel_index = np.concatenate([patch_row_x_kernel, patch_column_y_kernel], axis=3)

            return patch_to_kernel_index

        def reverse_2D_convolution_indices(self, indices, parent_shape):

            reverse_indices = np.zeros(np.concatenate([indices.shape[[0, 1]], parent_shape])).astype('bool')

            for x in range(indices.shape[0]):
                for y in range(indices.shape[1]):
                    reverse_indices[x, y][indices[:, 0], indices[:, 1]] = True

            return reverse_indices


    class BaseConvolutionTS(TopologyStrategy):
        def __init__(self, convolution):
            self.convolution = convolution


    class SpatialConvolutionTS(BaseConvolutionTS):
        def potential_parent_pose_input(self,
                                        input,  # [batch, child_row, child_column, child_feature, pose_row, pose_column]
                                        parent_feature_count
                                        ):

            s = tf.shape(input)

            [batch_size, child_row_count, child_column_count, child_feature_count, *pose_dimensions] = [s[0], s[1], s[2], s[3], s[4], s[5]]

            # reshape input to shift batch dimension to end of array

            input_shape_length = tf.shape(s)[0]

            transpose_perm_for_gather = tf.concat([tf.range(1, input_shape_length), tf.constant([0])])


            # [child_row, child_column, child_feature, pose_row, pose_column, batch]
            gatherable_input = tf.transpose(input, transpose_perm_for_gather)

            # create index array with shape of [kernel_row, kernel_column, parent_row, parent_column]

            # For each kernel row/column index and parent row/column index the correct child row/column index must
            # be in the last dimension of the index array passed to gather_nd. The correct row/column index can be
            # seen as a 2D axial system with an origin defined by patch row/column offset, and a further translation
            # along the kernel's axes defined by the relative kernel row/column position. Note the practical fact
            # that the row axes and column axes are entirely orthogonal and thus independent.

            # Therefore a practical way to accomplish all these translations is to define the translation for each
            # dimension (row or column) independently, and then add all coordinates for each dimension using broadcasting;
            #
            # Due to column and row translation independence; patch row offset + kernel row position = child row
            # position, and the same for column positions.
            #
            # By indexing the n -1 indices of the indices argument to gather_nd by [kernel_row, kernel_column, parent_row, parent_column]
            # and  putting all the translation data into arrays as described below broadcasting applies each translation
            # for every dimension correctly;

            # [kernel_row, 1, 1, 1, [0,1]] = [kernel_row_translation, 0] +  # kernel_row translation
            # [1, 1, parent_row, 1, [0,1]] = [parent_row_offset, 0]         # parent_row_offset translation
            # [1, kernel_column, 1, [0,1]] = [0, kernel_column_translation] + # kernel_column translation
            # [1, 1, 1, parent_column_translation, [0, 1]] = [0, parent_column_offset]

            # compute relevant data
            valid_kernel_space = (self.kernel_size - 1)

            parent_row_count = int((child_row_count - valid_kernel_space) / self.stride)
            parent_column_count = int((child_column_count - valid_kernel_space) / self.stride)

            # translation data
            patch_row_offsets    = tf.range(0, parent_row_count) * self.stride
            patch_column_offsets =  tf.range(0, parent_column_count) * self.stride

            kernel_row_deltas = tf.range(self.kernel_size)
            kernel_column_deltas = tf.range(self.kernel_size)

            # data translatable through broadcasting

            broadcastable_patch_row_offsets = tf.reshape(patch_row_offsets, [1, 1, parent_row_count, 1, 1])
            broadcastable_patch_row_offsets = tf.concat([broadcastable_patch_row_offsets, tf.zeros(tf.shape(broadcastable_patch_row_offsets))], axis=-1)

            broadcastable_patch_column_offsets = tf.reshape(patch_column_offsets, [1, 1, 1, parent_column_count, 1])
            broadcastable_patch_column_offsets = tf.concat([tf.zeros(tf.shape(broadcastable_patch_column_offsets)), broadcastable_patch_column_offsets], axis=-1)

            broadcastable_kernel_row_deltas = tf.reshape(kernel_row_deltas, [self.kernel_size, 1, 1, 1, 1])
            broadcastable_kernel_row_deltas = tf.concat([broadcastable_kernel_row_deltas, tf.zeros(tf.shape(broadcastable_kernel_row_deltas))], axis=-1)

            broadcastable_kernel_column_deltas = tf.reshape(kernel_column_deltas, [1, self.kernel_size, 1, 1, 1])
            broadcastable_kernel_column_deltas = tf.concat([tf.zeros(tf.shape(broadcastable_kernel_column_deltas)), broadcastable_kernel_column_deltas], axis=-1)

            kernel_patch_indices = broadcastable_patch_row_offsets + broadcastable_kernel_row_deltas + \
                broadcastable_patch_column_offsets + broadcastable_kernel_column_deltas

            gathered_output = tf.gather_nd(gatherable_input, kernel_patch_indices)

            # indices are [kernel_row_index + parent_row_index * stride, kernel_column_index + parent_column_index

            # reshape output to have batch at the beginning again
            transpose_perm_after_gather = tf.concat([[input_shape_length - 1], tf.range(input_shape_length - 1)])

            output = tf.tranpose(gathered_output, transpose_perm_after_gather)  # [batch, kernel_row, kernel_column, parent_row, parent_column, child_feature, pose_row, pose_column]

            # move child_feature to target position
            output = tf.transpose(output, [0, 1, 2, 5, 3, 4, 6, 7])

            # insert a dimension for parent feature at the target position
            output = tf.expand_dims(output, axis=5)

            # tile output for parent feature dimension
            output = tf.tile(output, [1, 1, 1, 1, 1, parent_feature_count, 1, 1, 1])

            return output  # [batch, kernel_row, kernel_column, child_feature, parent_row, parent_column, parent_feature, pose_row, pose_column]

        def potential_parent_pose_weights(self,
                                          input,   # [batch, child_row, child_column, child_feature, pose_row, pose_column]
                                          parent_feature_count
                                          ):

            # retrieve all relevant data from input
            s = tf.shape(input)
            [batch_size, child_row_count, child_column_count, child_feature_count, *pose_dimensions] = [s[0], s[1], s[2], s[3], s[4], s[5]]


            # construct the filter set
            weights = tf.Variable(
                tf.truncated_normal([1, self.kernel_size, self.kernel_size, child_feature_count, 1, 1, parent_feature_count, *pose_dimensions], mean=0.0, stddev=1.0, dtype=tf.float32),
                dtype=tf.float32, name='pose_transform_weights')

            # tile the filter set for each parent_row and parent_column

            valid_kernel_space = (self.kernel_size - 1)

            parent_row_count = int((child_row_count - valid_kernel_space) / self.stride)
            parent_column_count = int((child_column_count - valid_kernel_space) / self.stride)

            tiled_weights = tf.tile(weights, [batch_size, 1, 1, 1, parent_row_count, parent_column_count, 1, 1, 1])

            return tiled_weights  # [batch, kernel_row, kernel_column, child_feature, parent_row, parent_column, parent_feature, pose_row, pose_column]

        def children_of_parents(self, input):
            pass

        def parents_of_children(self, input):
            pass

    class SemanticConvolutionTS(BaseConvolutionTS):
        def potential_parent_pose_input(self, input):
            pass

        def potential_parent_pose_weights(self,
                                          input,  # [batch, child_row, child_column, child_feature_row, child_feature_column, pose_row, pose_column]
                                          parent_feature_dimensions  # [height, widths]
                                          ):
            # retrieve all relevant data from input
            s = tf.shape(input)
            [batch_size, child_row_count, child_column_count, *child_feature_dimensions, pose_row_count, pose_column_count] = [s[0], s[1], s[2], s[3], s[4], s[5], s[6]]


            # construct the filter set
            weights = tf.Variable(
                tf.truncated_normal(
                    [1, self.kernel_size, self.kernel_size, *child_feature_dimensions, 1, 1, *parent_feature_dimensions, pose_row_count, pose_column_count], mean=0.0, stddev=1.0, dtype=tf.float32),
                dtype=tf.float32, name='pose_transform_weights')

            # tile the filter set for each parent_row and parent_column

            valid_kernel_space = (self.kernel_size - 1)

            parent_row_count = int((child_row_count - valid_kernel_space) / self.stride)
            parent_column_count = int((child_column_count - valid_kernel_space) / self.stride)

            tiled_weights = tf.tile(weights, [batch_size, 1, 1, 1, 1, parent_row_count, parent_column_count, 1, 1, 1, 1])

            return tiled_weights

        def children_of_parents(self, input):
            pass

        def parents_of_children(self, input):
            pass

# get tensor's shape at graph construction time and standardizes it into a numpy array, such that reshape and all the
# other tensor flow functions can work with it
def numpy_shape_ct(tensor):
    shape = np.array(tensor.get_shape())

    numpy_shape = []
    for i in range(shape.shape[0]):

        v = shape[i].value
        v = -1 if v is None else v

        numpy_shape.append(v)

    return np.array(numpy_shape)
