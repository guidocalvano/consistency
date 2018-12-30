import numpy as np
import tensorflow as tf
import sys


class UnknownInitializerException(Exception):
    pass

class TopologyBuilder:

    def init(self, pose_width=4, pose_height=4):

        self.children_shape = []
        self.child_index_dimension_count = 0
        self.child_index_translations_for_kernel = []
        self.child_index_translations_for_parent = []

        self.added_coordinates = []

        self.shared_weights_parent = []
        self.shared_weights_kernel = []

        self.is_axial_system = False

        self.pose_width = pose_width
        self.pose_height = pose_height

        return self

    def set_is_axial_system(self, is_true):
        self.is_axial_system = is_true

    # interface requirement
    def add_spatial_convolution(self, spatial_input_dimensions, kernel_size, stride):
        self.children_shape += spatial_input_dimensions

        valid_kernel_space = (kernel_size - 1)
        parents_share_weights = True
        kernels_share_weights = False

        parent_row_count = int((spatial_input_dimensions[0] - valid_kernel_space) / stride)
        parent_column_count = int((spatial_input_dimensions[1] - valid_kernel_space) / stride)

        self.add_convolution(np.arange(kernel_size), np.arange(parent_row_count) * stride, kernels_share_weights, parents_share_weights)
        self.add_convolution(np.arange(kernel_size), np.arange(parent_column_count) * stride, kernels_share_weights, parents_share_weights)

    # interface requirements
    def add_semantic_convolution(self, semantic_input_dimensions, kernel_size, stride):
        self.children_shape += semantic_input_dimensions

        valid_kernel_space = 0
        parents_share_weights = False
        kernels_share_weights = False

        parent_row_count = int((semantic_input_dimensions[0] - valid_kernel_space) / stride)
        parent_column_count = int((semantic_input_dimensions[1] - valid_kernel_space) / stride)

        self.add_convolution(np.arange(kernel_size), np.arange(parent_row_count) * stride, kernels_share_weights, parents_share_weights)
        self.add_convolution(np.arange(kernel_size), np.arange(parent_column_count) * stride, kernels_share_weights, parents_share_weights)

    # interface requirement
    def add_dense_connection(self, input_feature_count, output_feature_count):
        self.children_shape += [input_feature_count]

        parents_share_weights = False
        kernels_share_weights = False

        self.add_convolution(np.arange(input_feature_count), np.zeros([output_feature_count]), kernels_share_weights, parents_share_weights)

    # interface requirement
    def add_aggregation(self, kernel_size, coordinate_addition_targets):
        self.children_shape += [kernel_size, kernel_size]

        parents_share_weights = False  # irrelevant parameter
        kernels_share_weights = True

        self.add_convolution(np.arange(kernel_size), np.array([0]), kernels_share_weights, parents_share_weights, coordinate_addition_targets[0])
        self.add_convolution(np.arange(kernel_size), np.array([0]), kernels_share_weights, parents_share_weights, coordinate_addition_targets[1])

    def finish(self, initialization_options):

        self.weight_tiling = self._construct_weight_tiling()
        self.weight_shape  = self._construct_weight_shape()

        self.kernel_mapping = self._construct_child_to_kernel_mapping()

        forward_edge_list, reverse_edge_list = self._convert_index_mapping_to_child_kernel_parent_edge_list(self.kernel_mapping)

        self.child_parent_kernel_mapping = forward_edge_list
        self.parent_kernel_child_mapping = reverse_edge_list

        self.add_initializer(initialization_options)


    # interface requirement
    def children_to_linear(self, children):
        batch_size = children.get_shape()[0]
        payload_dimensions = children.get_shape().as_list()[(self.child_index_dimension_count + 1):]
        linear_child_size = np.prod(self.children_shape)

        target_shape = [-1, linear_child_size]  + payload_dimensions

        linear_children = tf.reshape(children, target_shape)

        return linear_children

    def map_children_to_linear_kernel_linear_parent(self, input_layer):
        children_payload_shape = self.children_payload_shape(input_layer)
        children_to_kernel_parent_map = self.map_input_layer_to_parent_kernels(input_layer)
        target_shape = [-1] + [self.linear_kernel_shape()] + [self.linear_parent_shape()] + children_payload_shape
        linear_kernel_linear_parent_input = tf.reshape(children_to_kernel_parent_map, target_shape)

        return linear_kernel_linear_parent_input
        # linear_kernel_linear_parent = self._to_linear_kernel_linear_parent_from_map(children_to_kernel_parent_map)

    def _to_linear_kernel_linear_parent_from_map(self, kernel_parent_map):
        linear_kernel_size = np.prod(self.kernel_)

    def linearized_potential_parent_poses(self,
                                          input_layer_poses # [batch, *child_dimensions, 1, pose_width, pose_height]
                                          ):

        potential_parent_poses_map = self._compute_potential_parent_poses_map(input_layer_poses)

        tf.summary.histogram('potential_parent_poses_map', potential_parent_poses_map)
        tf.summary.histogram('potential_parent_poses_determinant', tf.linalg.det(potential_parent_poses_map))

        potential_parent_poses_map = self.add_coordinates(potential_parent_poses_map)

        tf.summary.histogram('potential_parent_poses_map_with_coordinates', potential_parent_poses_map)

        potential_parent_poses_linearized = self._linearize_potential_parent_poses_map(potential_parent_poses_map)

        return potential_parent_poses_linearized

    def _compute_potential_parent_poses_map(self, input_layer_poses):
        s = tf.shape(input_layer_poses)
        batch_size = s[0]
        pose_width, pose_height = input_layer_poses.get_shape().as_list()[-2:]

        kernel_mapped_input = self.map_input_layer_to_parent_kernels(input_layer_poses)
        kernel_mapped_weights = self.map_weights_to_parent_kernels(batch_size, pose_width, pose_height)

        # [*kernel_dimensions, *parent_dimensions, pose_width, pose_height]
        potential_parent_poses_map = tf.matmul(kernel_mapped_input, kernel_mapped_weights)

        return potential_parent_poses_map

    def _linearize_potential_parent_poses_map(self, potential_parent_poses_map):

        batch_size, pose_width, pose_height = [-1] + potential_parent_poses_map.get_shape().as_list()[-2:]

        children_per_kernel = np.prod([len(v) for v in self.child_index_translations_for_kernel])
        parent_count = np.prod([len(v) for v in self.child_index_translations_for_parent])

        linearized_shape = [batch_size, children_per_kernel, parent_count, pose_width, pose_height]

        potential_parent_poses_linearized = tf.reshape(potential_parent_poses_map, linearized_shape)

        return potential_parent_poses_linearized

    def add_coordinates(self, potential_parent_poses_map):
        potential_parent_poses_map_shape = [-1] + potential_parent_poses_map.get_shape().as_list()[1:]
        pose_width, pose_height = potential_parent_poses_map_shape[-2:]

        for added_coordinate in self.added_coordinates:
            # get info relevant to task
            coordinate_source_index = added_coordinate["source"] + 1  # add one to skip batch dimension
            coordinate_sink_index = added_coordinate["sink"]

            # compute the values that need to be inserted
            coordinate_count = potential_parent_poses_map_shape[coordinate_source_index]
            coordinate_range = np.arange(coordinate_count)
            scaled_coordinate_range = coordinate_range / (coordinate_count - 1)

            # create shape of template matrix for the coordinates that must be added (not broadcastable yet)
            pose_matrices_with_coordinates_shape = [coordinate_count, pose_width, pose_height]

            # compute the target indices that require coordinate addition
            coordinate_range_for_index_concatenation = np.reshape(coordinate_range, [coordinate_range.shape[0], 1])
            coordinate_sink_for_index_concatenation = np.tile(
                np.reshape(np.array(coordinate_sink_index), [1, len(coordinate_sink_index)]),
                [coordinate_range.shape[0], 1]);

            target_indices = np.concatenate([coordinate_range_for_index_concatenation, coordinate_sink_for_index_concatenation], axis=1)

            # added the coordinates to the template

            pose_matrices_with_coordinates = np.zeros(pose_matrices_with_coordinates_shape)
            pose_matrices_with_coordinates[list(np.transpose(target_indices))] = scaled_coordinate_range

            # prepare the template for broadcasting
            broadcastable_shape = np.ones([len(potential_parent_poses_map_shape)], dtype=np.int32)
            broadcastable_shape[[coordinate_source_index, -2, -1]] = pose_matrices_with_coordinates_shape

            broadcastable_pose_matrices_with_coordinates = np.reshape(pose_matrices_with_coordinates, broadcastable_shape)

            # do the broadcasting
            potential_parent_poses_map = potential_parent_poses_map + broadcastable_pose_matrices_with_coordinates

        return potential_parent_poses_map

    def map_input_layer_to_parent_kernels(self, input_layer_poses):

        dimension_count_input = len(input_layer_poses.get_shape())

        # move batch dimension to the end

        batch_at_end_permutation = list(range(1, dimension_count_input)) + [0]
        input_layer_poses_for_gather_nd = tf.transpose(input_layer_poses, batch_at_end_permutation)

        # gather input data
        gathered_input_slices = tf.gather_nd(input_layer_poses_for_gather_nd, self.kernel_mapping)

        # move batch dimension to start of input data
        dimension_count_slices = len(gathered_input_slices.get_shape())

        kernel_map_permutation = [dimension_count_slices - 1] + list(range(dimension_count_slices - 1))

        kernel_mapped_slices = tf.transpose(gathered_input_slices, kernel_map_permutation)

        return kernel_mapped_slices


    def add_initializer(self, options):
        if options["type"] == "xavier":
            self.weight_initializer = self.build_xavier_init(options)
            return

        if options["type"] == "identity":
            self.weight_initializer = self.build_identity_init(options)
            return

        if options["type"] == "normal":
            self.weight_initializer = self.build_truncated_normal_init(options)
            return

        raise UnknownInitializerException()

    def build_xavier_init(self, options):

        input_count_per_node = 1
        if "kernel" in options and options["kernel"]:
            input_count_per_node = input_count_per_node * np.product(self.kernel_shape())

        if "matrix" in options and options["matrix"]:  # xavier_init_matrix or xavier_init_kernel_and_matrix
            # @TODO: perhaps height should always be passed first to follow the row metaphor for index 0
            input_count_per_node = input_count_per_node * self.pose_width

        #@TODO: Make the standard deviation take into account the number of outputs per node
        xavierish_standard_deviation = np.sqrt(1.0 / input_count_per_node)  # -ish because technically xavier init is for tanh not sigmoid

        return tf.truncated_normal(self.complete_weight_shape(), stddev=xavierish_standard_deviation)

    def build_identity_init(self, options):
        noise = tf.random_uniform(self.complete_weight_shape(), -options["uniform"], options["uniform"])

        eye = tf.eye(self.pose_width, self.pose_height - int(self.is_axial_system))

        eye_shape = np.ones([len(noise.get_shape().as_list())])
        eye_shape[[-2, -1]] =  eye.get_shape().as_list()

        broadcastable_eye = tf.reshape(eye, eye_shape)

        return noise * (broadcastable_eye - 1.0) + broadcastable_eye

    def build_truncated_normal_init(self, options):

        next_std = options["deviation"].pop(0)

        return tf.truncated_normal(self.complete_weight_shape(), stddev=next_std)


    def complete_weight_shape(self):
        return [1] + self.weight_shape + [self.pose_width, self.pose_height - int(self.is_axial_system)]

    def map_weights_to_parent_kernels(self, batch_size, pose_width, pose_height):

        weights = tf.Variable(self.weight_initializer,
                              dtype=tf.float32, name='pose_transform_weights')

        tf.summary.histogram('pose_transform_weights', weights)

        tf.add_to_collection('weights', weights)

        if self.is_axial_system:

            tiling_vector_type_shape = [1] + self.weight_shape + [1, 1]

            tilable_vector_type_shape = [1] * len(tiling_vector_type_shape)
            tilable_vector_type_shape[-2] = pose_width


            vector_type_column = tf.constant([0.0, 0.0, 0.0, 1.0], dtype=tf.float32)

            concatenatable_vector_type_column = tf.tile(tf.reshape(vector_type_column, tilable_vector_type_shape), tiling_vector_type_shape)

            weights = tf.concat([weights, concatenatable_vector_type_column], axis=-1)

        kernel_tile_dimensions = [batch_size] + self.weight_tiling + [1, 1]

        tiled_weights = tf.tile(weights, kernel_tile_dimensions)

        # by using the tiled_weights the update stays proportional to gradient based updates
        TopologyBuilder.add_orthogonal_unit_axis_loss(tf.reshape(tiled_weights, [-1, pose_width, pose_height]))

        return tiled_weights

    # interface requirement
    def replace_kernel_elements_with_sum_of_child(self,
                                                  parent_kernel_values  # [batch, kernel_position_linearized, parent_position_linearized, 1]
                                                  ):
        values_summed_per_child = self._compute_sum_for_children(parent_kernel_values)

        parent_kernel_values_shape = [-1] + parent_kernel_values.get_shape().as_list()[1:]
        child_values_projected_to_parent_kernel_values = self._project_child_scalars_to_parent_kernels(values_summed_per_child, parent_kernel_values_shape)
        return child_values_projected_to_parent_kernel_values

    def _compute_sum_for_children(self, parent_kernel_values):
        s = tf.shape(parent_kernel_values)
        batch_size, linear_kernel_position_size, parent_position_size = s[0], s[1], s[2]

        # [batch, kernel_parent_position_linearized]
        fully_linearized_kernel_parent_position = tf.reshape(parent_kernel_values, [batch_size, -1])

        # [kernel_parent_position_linearized, batch]
        parent_kernel_prepared_for_sum_per_child = tf.transpose(fully_linearized_kernel_parent_position)

        linearized_child_to_parent_kernel = self.make_sparse_linearized_child_to_parent_kernel(self.child_parent_kernel_mapping)  # [edge_id, coordinate], coordinate = *child_dimensions + *kernel_dimensions + *parent_dimensions

        # multiply by linear_parent_to_linear_child matrix, thus summing the values of all parents of each child:
        # [linearized_child, batch] = [linearized_child, linearized_kernel_parent] sparse dense matmul [linearized_kernel_parent, batch]

        values_summed_per_child = tf.sparse_tensor_dense_matmul(linearized_child_to_parent_kernel, parent_kernel_prepared_for_sum_per_child)

        return tf.transpose(values_summed_per_child)  # reverse batch and sum dimensions back

    def _project_child_scalars_to_parent_kernels(self,
                                                 child_values,  # [batch, child_value]
                                                 parent_kernel_shape # [batch, kernel_position_linearized, parent_position_linearized, 1]
                                                ):
        batch_size, linear_kernel_position_size, parent_position_size = parent_kernel_shape[0], parent_kernel_shape[1], parent_kernel_shape[2]

        linearized_parent_kernel_to_child = self.make_sparse_linearized_parent_kernel_to_child(self.parent_kernel_child_mapping)

        # multiply by linear_child_to_linear_parent matrix, thus copying the summed values to all the appropriate locations
        # [linearized_parent_kernel, batch] = [linearized_parent_kernel, linearized_child] sparse_dense_matmul [linearized_child, batch]

        child_summed_values_per_kernel_parent_position_transposed = tf.sparse_tensor_dense_matmul(linearized_parent_kernel_to_child, tf.transpose(child_values))

        linearized_child_summed_values_per_kernel_parent_position = tf.transpose(child_summed_values_per_kernel_parent_position_transposed)

        child_summed_values_per_kernel_parent_position = tf.reshape(linearized_child_summed_values_per_kernel_parent_position, [batch_size, linear_kernel_position_size, parent_position_size, 1])

        return child_summed_values_per_kernel_parent_position

    def make_sparse_linearized_child_to_parent_kernel(self, child_parent_kernel_mapping):
        values = np.array([1.0] * child_parent_kernel_mapping.shape[0], dtype=np.float32)
        parent_kernel_shape = self.parent_kernel_shape()
        dense_shape = self.children_shape + parent_kernel_shape
        as_sparse_map = tf.sparse_reorder(tf.SparseTensor(indices=child_parent_kernel_mapping, values=values, dense_shape=dense_shape))

        linearized_shape = [np.product(self.children_shape), np.product(parent_kernel_shape)]

        linearized_child_parent_kernel_mapping = tf.sparse_reshape(as_sparse_map, linearized_shape)

        return linearized_child_parent_kernel_mapping

    def make_sparse_linearized_parent_kernel_to_child(self, parent_kernel_child_mapping):
        values = np.array([1.0] * parent_kernel_child_mapping.shape[0], dtype=np.float32)
        parent_kernel_shape = self.parent_kernel_shape()
        dense_shape = parent_kernel_shape + self.children_shape
        as_sparse_map = tf.sparse_reorder(tf.SparseTensor(indices=parent_kernel_child_mapping, values=values, dense_shape=dense_shape))

        linearized_shape = [np.product(parent_kernel_shape), np.product(self.children_shape)]

        linearized_parent_kernel_child_mapping = tf.sparse_reshape(as_sparse_map, linearized_shape)

        return linearized_parent_kernel_child_mapping

    def children_payload_shape(self, children_map):

        structure_dimension_count = len(self.children_shape) + 1  # one extra for the batch dimension

        return children_map.get_shape().as_list()[structure_dimension_count:]

    def linear_kernel_shape(self):
        return np.prod(self.kernel_shape())

    def linear_parent_shape(self):
        return np.prod(self.parent_shape())

    def kernel_shape(self):
        return [len(x) for x in self.child_index_translations_for_kernel]

    def parent_shape(self):
        return [len(y) for y in self.child_index_translations_for_parent]

    def parent_kernel_shape(self):
        return self.kernel_shape() + self.parent_shape()

    def parent_payload_shape(self, parent_map):
        parent_map_shape = parent_map.get_shape().as_list()
        parent_map_dimension_count = len(parent_map_shape)
        parent_shape_dimension_count = len(self.parent_shape())
        non_payload_dimension_count = parent_shape_dimension_count + 1  # one extra for the batch dimension

        payload_shape = parent_map_shape[non_payload_dimension_count:]

        return payload_shape

    # interface requirement
    def reshape_parents_to_map(self, linear_parents):

        [batch_size, parent_count, one, *data_dimensions] = [-1] + linear_parents.get_shape().as_list()[1:]

        parent_map_dimensions = [len(v) for v in self.child_index_translations_for_parent]

        map_shape = [batch_size, *parent_map_dimensions, one, *data_dimensions]

        map_parents = tf.reshape(linear_parents, map_shape)

        return map_parents

    def reshape_parent_map_to_linear(self, parent_map):
        payload_shape = self.parent_payload_shape(parent_map)
        parent_count = self.linear_parent_shape()
        linear_parent_shape = [-1] + [parent_count] + payload_shape

        linear_parents = tf.reshape(parent_map, linear_parent_shape)

        return linear_parents

    # interface requirement
    def add_convolution(self, kernel_values, parent_values, kernels_share_weights, parents_share_weights, coordinate_addition_target=None):

        if coordinate_addition_target is not None:
            self.added_coordinates += [{
                "source": self.child_index_dimension_count,
                "sink": coordinate_addition_target
            }]

        self.child_index_dimension_count += 1

        self.child_index_translations_for_kernel += [kernel_values]
        self.child_index_translations_for_parent += [parent_values]
        self.shared_weights_parent += [parents_share_weights]
        self.shared_weights_kernel += [kernels_share_weights]

    def kernel_map_dimension_count(self):
        return len(self.child_index_translations_for_kernel) + len(self.child_index_translations_for_parent) + 1

    def _construct_array_for_broadcast_translation(self, array, translated_dimension, broadcasted_dimension):

        kernel_map_dimensions = self.kernel_map_dimension_count()

        array_length = array.shape[0]

        broadcastable_array_shape = np.ones(kernel_map_dimensions, dtype=np.int32)
        broadcastable_array_shape[broadcasted_dimension] = array_length

        broadcastable_array_shape[-1] = self.child_index_dimension_count

        broadcastable_indices_shape = [array.shape[0], self.child_index_dimension_count]
        broadcastable_indices = np.zeros(broadcastable_indices_shape, dtype=np.int32)
        broadcastable_indices[:, translated_dimension] = array

        broadcastable_array = broadcastable_indices.copy().reshape(broadcastable_array_shape)

        return broadcastable_array

    # utility
    def _construct_child_to_kernel_mapping(self):
        kernel_map_shape = np.ones([self.kernel_map_dimension_count()], dtype=np.int32)
        kernel_map_shape[-1] = self.child_index_dimension_count

        child_index_for_kernel_position = np.zeros(kernel_map_shape, dtype=np.int32)

        kernel_dimension_count = len(self.child_index_translations_for_kernel)
        for i in range(kernel_dimension_count):
            array = self.child_index_translations_for_kernel[i]
            child_index_for_kernel_position = child_index_for_kernel_position + self._construct_array_for_broadcast_translation(array, i, i)

        parent_dimension_count = len(self.child_index_translations_for_parent)
        for j in range(parent_dimension_count):
            array = self.child_index_translations_for_parent[j]
            child_index_for_kernel_position = child_index_for_kernel_position + self._construct_array_for_broadcast_translation(array, j, kernel_dimension_count + j)

        # if the convolution is semantic it should wrap round, which can be accomplished by doing a modulo division
        # for each child index dimension
        broadcastable_shape = np.ones([len(child_index_for_kernel_position.shape)], dtype=np.int32)
        broadcastable_shape[-1] = len(self.children_shape)

        broadcastable_division = np.array(self.children_shape).reshape(broadcastable_shape)

        child_index_for_kernel_position = child_index_for_kernel_position % broadcastable_division

        return child_index_for_kernel_position

    def _convert_index_mapping_to_child_kernel_parent_edge_list(self, gather_nd_index_mapping):

        kernel_parent_map_dimensions = [len(x) for x in self.child_index_translations_for_kernel] + [len(x) for x in self.child_index_translations_for_parent]

        kernel_parent_map_index_list = np.array(list(np.ndindex(*kernel_parent_map_dimensions)))

        enumerated_child_indices = gather_nd_index_mapping[list(np.transpose(kernel_parent_map_index_list))]

        kernel_parent_map_index_array = np.array(kernel_parent_map_index_list)

        child_parent_kernel_mapping = np.concatenate([enumerated_child_indices, kernel_parent_map_index_array], axis=1)
        parent_kernel_child_mapping = np.concatenate([kernel_parent_map_index_array, enumerated_child_indices], axis=1)

        return child_parent_kernel_mapping, parent_kernel_child_mapping

    def _construct_weight_shape(self):

        weight_shape = []

        for i in range(len(self.child_index_translations_for_kernel)):
            weight_shape += [1] if self.shared_weights_kernel[i] else [len(self.child_index_translations_for_kernel[i])]

        for i in range(len(self.child_index_translations_for_parent)):
            weight_shape += [1] if self.shared_weights_parent[i] else [len(self.child_index_translations_for_parent[i])]

        return weight_shape

    def _construct_weight_tiling(self):
        tiling_shape = []

        for i in range(len(self.child_index_translations_for_kernel)):
            tiling_shape += [len(self.child_index_translations_for_kernel[i])] if self.shared_weights_kernel[i] else [1]

        for i in range(len(self.child_index_translations_for_parent)):
            tiling_shape += [len(self.child_index_translations_for_parent[i])] if self.shared_weights_parent[i] else [1]

        return tiling_shape

    @staticmethod
    def add_orthogonal_loss( # adds loss of dot product of components of a set of axial system matrices
                              axial_system_list  # [axial_system, vector, component]
        ):
        # the objective is to reduce orthogonal loss to 0 for every single axial system it is applied to, so
        # it should be based on l1 regularization

        vector_axis = 1
        component_axis = 2

        x_axis = tf.gather(axial_system_list, [0], axis=vector_axis)
        y_axis = tf.gather(axial_system_list, [1], axis=vector_axis)
        z_axis = tf.gather(axial_system_list, [2], axis=vector_axis)

        non_orthogonal_error = tf.reduce_sum(x_axis * y_axis + y_axis * z_axis + z_axis * x_axis, axis=component_axis)

        # this causes orthogonal loss to be regularized analogous to l1 regularization, pushing it down to 0
        non_orthogonal_loss = tf.reduce_sum(tf.abs(non_orthogonal_error))

        tf.add_to_collection("orthogonal_regularization", non_orthogonal_loss)

        return non_orthogonal_loss

    @staticmethod
    def add_unit_scale_loss( # adds loss of dot product of components of a set of axial system matrices
                              axial_system_list  # [axial_system, vector, component]
        ):
        # The objective is to make every axial system have unit scale [1, 1, 1]
        # To accomplish this all actual_scale - unit_scale should equal [0, 0, 0]
        # Minimizing the absolute difference (analogous to L1 regularization) has this effect

        vector_axis = 1
        component_axis = 2

        # drop translation
        axial_system_without_translation_list = tf.gather(axial_system_list, [0, 1, 2], axis=vector_axis)

        # derivative of sqrt for non zero input is problematic.
        non_zero_abs_axial_system_list = tf.abs(axial_system_without_translation_list) + sys.float_info.epsilon

        scale_list = tf.sqrt(tf.reduce_sum(non_zero_abs_axial_system_list * non_zero_abs_axial_system_list, axis=component_axis))

        scale_error = scale_list - 1.0  # the difference between the actual scale and 1 is the error

        # By using the absolute error as the loss behavior analogous to L1 regularization will push
        scale_loss = tf.reduce_sum(tf.abs(scale_error))

        tf.add_to_collection("unit_scale_regularization", scale_loss)

        return scale_loss


    @staticmethod
    def add_orthogonal_unit_axis_loss(axial_system_list):
        non_orthogonal_loss = TopologyBuilder.add_orthogonal_loss(axial_system_list)
        scale_loss = TopologyBuilder.add_unit_scale_loss(axial_system_list)

        return tf.reduce_sum(non_orthogonal_loss) + tf.reduce_sum(scale_loss)
