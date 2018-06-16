import tensorflow as tf
import numpy as np
import math
import sys

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

    def build_default_architecture(self, input_layer, full_example_count, iteration_count, routing_state, is_training):

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

        next_routing_state = tf.get_collection("next_routing_state")

        return aggregating_capsule_layer, spread_loss_margin, next_routing_state

    def build_aggregating_capsule_layer(self,
                                        input_layer_list,
                                        use_coordinate_addition,
                                        parent_count,
                                        steepness_lambda,
                                        iteration_count,
                                        routing_state=None):
        input_activations = input_layer_list[0]
        input_poses = input_layer_list[1]

        normal_layer_coordinate_addition = None
        if use_coordinate_addition:

            [batch_size, width, height, capsule_count, parent_broadcast_dim, activation_dim] = numpy_shape_ct(input_activations)

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

        aggregated_capsule_layer = self.build_matrix_caps(
            parent_count,
            steepness_lambda,
            iteration_count,
            normal_input_activations,
            normal_input_poses,
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

        normal_layer = tf.reshape(input_layer, shape=normal_shape)
        return normal_layer

    def build_encoding_convolution(self, input_layer, kernel_size, filter_count):
        output_layer = tf.layers.conv2d(
            input_layer,
            kernel_size=kernel_size,
            filters=filter_count,
            strides=[2, 2],
            activation=tf.nn.relu  #@TODO: test leaky relu
        )
        return output_layer

    def build_decoding_convolution(self, decoded_input, encoded_output, parent_decoder, kernel_size, filter_count):

        weights_shape = np.concatenate([kernel_size, [filter_count]])
        weights = tf.Variable(tf.truncated_normal(weights_shape))

        reconstruction_training_layer = tf.layers.conv2d_transpose(
            encoded_output,
            weights,
            kernel_size=kernel_size,
            filters=decoded_input.shape[3].value,
            activation=tf.nn.relu  #@TODO: test leaky relu
        )
        reconstruction_loss = tf.abs(tf.stop_gradient(decoded_input) - reconstruction_training_layer)

        reconstruction_prediction_layer = tf.layers.conv2d_transpose(
            parent_decoder,
            weights,
            kernel_size=kernel_size,
            filters=decoded_input.shape[3].value,
            activation=tf.nn.relu  #@TODO test leaky relu
        )

        return reconstruction_prediction_layer, reconstruction_loss

    def build_cast_conv_to_pose_layer(self, input_layer, input_filter_count):

        pose_element_layer = tf.layers.conv2d(
            input_layer,
            kernel_size=[1, 1],
            filters=input_filter_count * 16,
            activation=None
        )
        pose_shape = pose_element_layer.get_shape()

        parent_broadcast_dim = 1
        pose_layer = tf.reshape(pose_element_layer, shape=[-1, pose_shape[1], pose_shape[2], input_filter_count, parent_broadcast_dim, 4, 4])

        return pose_layer

    def build_cast_conv_to_activation_layer(self, input_layer, input_filter_count):

        raw_activation_layer = tf.layers.conv2d(
            input_layer,
            kernel_size=[1, 1],
            filters=input_filter_count,
            activation=tf.nn.sigmoid
        )

        parent_broadcast_dim = tf.Dimension(1)
        # by adding two extra dimensions both em routing and the linking of layers becomes a lot easier
        target_shape = np.concatenate([raw_activation_layer.get_shape(), [parent_broadcast_dim, tf.Dimension(1)]])
        for i in range(target_shape.shape[0]):
            target_shape[i] = target_shape[i] if target_shape[i].value is not None else -1

        activation_layer = tf.reshape(raw_activation_layer, shape=target_shape)

        return activation_layer

    def build_primary_matrix_caps(self, input_layer):

        filter_axis = 3

        input_filter_count = input_layer.get_shape()[filter_axis]

        activation_layer = self.build_cast_conv_to_activation_layer(input_layer, input_filter_count)
        pose_layer = self.build_cast_conv_to_pose_layer(input_layer, input_filter_count)

        return [activation_layer, pose_layer]

    def build_convolutional_capsule_layer(self,
                                          input_layer_list,
                                          kernel_size,
                                          stride,
                                          parent_count,
                                          steepness_lambda,
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
                    steepness_lambda,
                    iteration_count,
                    activations,
                    poses,
                    routing_state
                )
        )

    def build_convolution_of(self, input_feature_image_layer_list, kernel_size: int, stride: int, filter_layer_constructor):
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
        patch_batch_results = filter_layer_constructor(*argument_list)
        results = []

        for filtered_image_patch_batches in patch_batch_results:
            # RESHAPE TO PATCH X AND PATCH Y

            filtered_output_shape = numpy_shape_ct(filtered_image_patch_batches)[1:]

            filtered_batch_x_y_shape = [batch_size, patch_row, patch_column]

            filtered_image_batch_shape = np.concatenate([filtered_batch_x_y_shape, filtered_output_shape])

            filtered_image_batch = tf.reshape(filtered_image_patch_batches, shape=filtered_image_batch_shape)
            results.append(filtered_image_batch)

        return results


    def build_potential_parent_pose_layer(self, child_poses, parent_capsule_count):
        # retrieve relevant dimensions
        child_poses_shape = numpy_shape_ct(child_poses)
        batch_size = tf.shape(child_poses)[0]
        child_capsule_count = child_poses_shape[1]

        # UNFORTUNATELY tf.matmul DOES NOT IMPLEMENT BROADCASTING, SO THE CODE BELOW DOES SO MANUALLY

        # A pose transform matrix exists from each child to each parent.
        # Weights with bigger stddev improve numerical stability
        pose_transform_weights = tf.Variable(
            tf.truncated_normal([1, child_capsule_count, parent_capsule_count, 4, 4], mean=0.0, stddev=1.0),
            dtype=tf.float32)

        # Potential parent poses must be predicted for each batch row. So weights must be copied for each batch row.
        pose_transform_weights_copied_for_batch = tf.tile(pose_transform_weights, [batch_size, 1, 1, 1, 1])

        ## Because the child poses are used to produce the pose of every potential parent, a copy is necessary for
        ## predicting the potential parent poses
        ## A column after the child capsule column is added,
        # child_poses_with_copy_column = tf.reshape(child_poses, shape=[batch_size, child_capsule_count, 1, 4, 4])

        # above code redundant now that broadcasting dims are part of specs

        # so output can be copied for each parent
        child_poses_copied_for_parents = tf.tile(child_poses, [1, 1, parent_capsule_count, 1, 1])

        # child poses are now copied for each potential parent, and child to parent tranforms are copied for each batch
        # row, resulting in two tensors; a tensor containing pose matrices in its last two indices, and one
        # containing pose transformations from each child to each potential parent pose
        #
        # matmul will now iterate over batch, child capsule, and parent capsule in both tensors and multiply the child
        # pose with the pose transform to output the pose of a potential parent
        potential_parent_poses = tf.matmul(child_poses_copied_for_parents, pose_transform_weights_copied_for_batch)

        return potential_parent_poses

    def cast_pose_matrices_to_vectors(self, pose_matrices):  # takes the last two matrix indices and reshapes into vector indices
        current_shape = pose_matrices.get_shape()
        new_shape = current_shape[:-1]
        new_shape[-1] = current_shape[-1] * current_shape[-2]

        return tf.reshape(pose_matrices, shape=new_shape)

    def cast_pose_vectors_to_matrices(self, pose_vectors):
        current_shape = pose_vectors.get_shape()
        matrix_diagonal_length = math.sqrt(current_shape[-1])
        new_shape = np.concatenate([current_shape[:-1], [matrix_diagonal_length], [matrix_diagonal_length]])
        new_shape[-1] = current_shape[-1] * current_shape[-2]

    def build_parent_assembly_layer(self,
                                    child_activations,
                                    potential_parent_pose_vectors,  # [batch, child, parent, pose_vector]
                                    steepness_lambda,
                                    iteration_count, # em routing
                                    routing_state):
        parent_axis = 2
        [_, child_count, parent_count, pose_element_count] = numpy_shape_ct(potential_parent_pose_vectors)
        batch_size = tf.shape(potential_parent_pose_vectors)[0]

        beta_u = tf.Variable(tf.truncated_normal([1, 1, parent_count, 1]))
        beta_a = tf.Variable(tf.truncated_normal([1, 1, parent_count, 1]))

        initial_child_parent_assignment_weights = tf.ones([batch_size, child_count, parent_count, 1], tf.float32) / float(parent_count)

        if routing_state is not None:
            initial_child_parent_assignment_weights = tf.constant(next(routing_state))

        child_parent_assignment_weights = initial_child_parent_assignment_weights

        for i in range(iteration_count + 1):
            parent_activations, likely_parent_pose, likely_parent_pose_deviation, likely_parent_pose_variance =\
                self.estimate_parents_layer(
                    child_parent_assignment_weights,
                    child_activations,
                    potential_parent_pose_vectors,
                    beta_u,
                    beta_a,
                    steepness_lambda
                )

            # the final iteration of this loop will needlessly produce an extra set of nodes using below code,
            # BUT, this doesn't matter because what is actually computed is determined by dependency analysis of the
            # graph and the nodes produced below in the final loop are not required for anything and thus ignored
            child_parent_assignment_weights = self.estimate_children_layer(
                parent_activations,
                likely_parent_pose,
                likely_parent_pose_variance,
                potential_parent_pose_vectors
            )

        tf.add_to_collection('next_routing_state', child_parent_assignment_weights)

        output_parent_activations = parent_activations
        output_parent_poses = likely_parent_pose

        return output_parent_activations, output_parent_poses


    def estimate_parents_layer(self, # m step
                               child_parent_assignment_weights,  # [batch, child, parent, 1]
                               child_activations,  # [batch, child, 1, 1]
                               potential_parent_pose_vectors,  # [batch, child, parent, pose_vector]
                               beta_u,  # [1, 1, parent, 1] @TODO double check correct dimensions!
                               beta_a,  # [1, 1, parent, 1] @TODO double check correct dimensions!
                               steepness_lambda  # scalar
                               ):

        [batch_count, child_count, parent_count, pose_element_count] = numpy_shape_ct(potential_parent_pose_vectors)

        assert (numpy_shape_ct(child_parent_assignment_weights)[1:] == np.array([batch_count, child_count, parent_count, 1])[1:]).all()
        assert (numpy_shape_ct(child_activations) == np.array([batch_count, child_count, 1, 1])).all()
        assert (numpy_shape_ct(beta_u) == np.array([1, 1, parent_count, 1])).all()
        assert (numpy_shape_ct(beta_a) == np.array([1, 1, parent_count, 1])).all()

        batch_axis = 0
        child_axis = 1
        parent_axis = 2
        pose_element_axis = 3  # @TODO: double check if this is the correct axis
        # select child parent assignments based on child activity
        # [batch, child, parent, 1]
        #@TODO: check if broadcasting works correctly
        active_child_parent_assignment_weights = child_parent_assignment_weights * child_activations
        assert (numpy_shape_ct(active_child_parent_assignment_weights)[1:] == np.array([batch_count, child_count, parent_count, 1])[1:]).all()

        # [batch, 1, parent, 1]
        total_active_child_parent_assignment_per_parent = tf.reduce_sum(active_child_parent_assignment_weights,
                                                                 axis=child_axis,
                                                                 keepdims=True)
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
            axis=child_axis, keepdims=True
        )
        assert (numpy_shape_ct(likely_parent_pose)[1:] == np.array([batch_count, 1, parent_count, pose_element_count])[1:]).all()

        # @TODO DOUBLE CHECK IF tf.square DOES WHAT THE MATRIX CAPS PAPER DEMANDS!!!
        # this is the sigma of the parent in the paper
        # [batch, 1, parent, likely_pose_element_variance]
        likely_parent_pose_variance = tf.reduce_sum(
            tf.square(potential_parent_pose_vectors - likely_parent_pose) * child_proportion_of_parent,
            axis=child_axis, keepdims=True
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
            keepdims=True
        )
        assert (numpy_shape_ct(sum_cost_h)[1:] == np.array([batch_count, 1, parent_count, 1])[1:]).all()

        cost_a = beta_a
        # [batch, 1, parent, 1]
        cost_balance_activity = cost_a - sum_cost_h
        assert (numpy_shape_ct(cost_balance_activity)[1:] == np.array([batch_count, 1, parent_count, 1])[1:]).all()

        # [batch, 1, parent, 1]
        parent_activations = tf.nn.sigmoid(steepness_lambda * cost_balance_activity)
        assert (numpy_shape_ct(parent_activations)[1:] == np.array([batch_count, 1, parent_count, 1])[1:]).all()

        return parent_activations, likely_parent_pose, likely_parent_pose_deviation, likely_parent_pose_variance

    def estimate_children_layer(self,  # e step
                                parent_activations,  # [batch, 1, parent, 1]
                                likely_parent_pose,  # [batch, 1, parent, likely_pose_element]
                                likely_parent_pose_variance,  # [batch, 1, parent, likely_pose_element_variance]
                                potential_parent_pose_vectors):  # [batch, child, parent, pose_element]

        [batch_count, child_count, parent_count, pose_element_count] = numpy_shape_ct(potential_parent_pose_vectors)

        assert (numpy_shape_ct(parent_activations)[1:] == np.array([batch_count, 1, parent_count, 1])[1:]).all()
        assert (numpy_shape_ct(likely_parent_pose)[1:] == np.array([batch_count, 1, parent_count, pose_element_count])[1:]).all()
        assert (numpy_shape_ct(likely_parent_pose_variance)[1:] == np.array([batch_count, 1, parent_count, pose_element_count])[1:]).all()

        batch_axis = 0
        child_axis = 1
        parent_axis = 2
        pose_element_axis = 3

        # [batch, 1, parent, 1]
        factor = tf.reciprocal(
            tf.sqrt(tf.reduce_prod(likely_parent_pose_variance, axis=pose_element_axis, keepdims=True) * 2.0 * math.pi) + sys.float_info.epsilon
        )
        assert (numpy_shape_ct(factor)[1:] == np.array([batch_count, 1, parent_count, 1])[1:]).all()


        # [batch, child, parent, pose_element] @TODO is likely_parent_pose  broadcasted correctly
        potential_parent_pose_variance = tf.square(potential_parent_pose_vectors - likely_parent_pose)
        assert (numpy_shape_ct(potential_parent_pose_variance)[1:] == np.array([batch_count, child_count, parent_count, pose_element_count])[1:]).all()

        # [batch, child, parent, 1]
        power = -tf.reduce_sum(
            potential_parent_pose_variance /
            (likely_parent_pose_variance * 2 + sys.float_info.epsilon), axis=pose_element_axis, keepdims=True)
        assert (numpy_shape_ct(power)[1:] == np.array([batch_count, child_count, parent_count, 1])[1:]).all()

        # [batch, child, parent, 1]
        parent_probability_per_child = factor * tf.exp(power)
        assert (numpy_shape_ct(parent_probability_per_child)[1:] == np.array([batch_count, child_count, parent_count, 1])[1:]).all()

        # [batch, child, parent, 1]
        active_parent_probability_per_child = parent_probability_per_child * parent_activations
        assert (numpy_shape_ct(active_parent_probability_per_child)[1:] == np.array([batch_count, child_count, parent_count, 1])[1:]).all()

        # [batch, child, parent, 1]
        child_parent_assignment_weights = active_parent_probability_per_child \
            / (tf.reduce_sum(active_parent_probability_per_child, axis=parent_axis, keepdims=True) + sys.float_info.epsilon)
        assert (numpy_shape_ct(child_parent_assignment_weights)[1:] == np.array([batch_count, child_count, parent_count, 1])[1:]).all()

        return child_parent_assignment_weights

    def build_matrix_caps(
            self,
            parent_count,
            steepness_lambda,
            iteration_count,
            child_activation_vector,
            child_pose_layer,
            routing_state,
            convolution_coordinates=None):

        potential_parent_poses = self.build_potential_parent_pose_layer(child_pose_layer, parent_count)

        [batch_size, child_count, _, pose_matrix_width, pose_matrix_height] = numpy_shape_ct(potential_parent_poses)

        pose_element_count = pose_matrix_width * pose_matrix_height
        potential_parent_pose_vectors = tf.reshape(potential_parent_poses,
            [batch_size, child_count, parent_count, pose_element_count]
        )
        pose_element_axis = 3
        if convolution_coordinates is not None:
            dynamic_batch_size = tf.shape(potential_parent_poses)[0]
            convolution_coordinates_tiled_for_parents = tf.tile(convolution_coordinates, [dynamic_batch_size, 1, parent_count, 1])

            potential_parent_pose_vectors = tf.concat(
                [potential_parent_pose_vectors, convolution_coordinates_tiled_for_parents],
                axis=pose_element_axis)

        parent_activations, parent_pose_vectors = self.build_parent_assembly_layer(
            child_activation_vector,
            potential_parent_pose_vectors,
            steepness_lambda,
            iteration_count,
            routing_state
        )

        new_child_count = parent_count
        new_parent_broadcast_dim = 1

        parent_as_child_activations = tf.reshape(parent_activations, shape=[batch_size, new_child_count, new_parent_broadcast_dim, 1])
        parent_pose_vectors_without_coordinate_addition = parent_pose_vectors[:, :, :, :16]
        parent_as_child_pose_matrices = tf.reshape(parent_pose_vectors_without_coordinate_addition, shape=[batch_size, new_child_count, new_parent_broadcast_dim, pose_matrix_width, pose_matrix_height])

        return parent_as_child_activations, parent_as_child_pose_matrices

    def progress_percentage_node(self, batch_size, full_example_count, is_training):
        example_counter = tf.Variable(tf.constant(float(0)), trainable=False, dtype=tf.float32)

        example_counter = tf.cond(is_training,
                                  lambda: example_counter.assign(example_counter + batch_size, use_locking=True),
                                  lambda: example_counter)

        progress_percentage = example_counter / tf.cast(full_example_count, tf.float32)
        progress_percentage = tf.cond(tf.logical_and(is_training, progress_percentage <= 1), lambda: progress_percentage, lambda: tf.constant(1.0))

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
