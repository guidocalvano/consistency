import numpy as np
import tensorflow as tf

class TopologyBuilder:

    def init(self):

        self.children_shape = []
        self.child_index_dimension_count = 0
        self.child_index_translations_for_kernel = []
        self.child_index_translations_for_parent = []

        self.added_coordinates = []

        self.shared_weights_parent = []
        self.shared_weights_kernel = []

    def finish(self):

        self.weight_tiling = self._construct_weight_tiling()
        self.weight_shape  = self._construct_weight_shape()

        self.kernel_mapping = self._construct_child_to_kernel_mapping()

        forward_edge_list, reverse_edge_list = self._convert_index_mapping_to_child_kernel_parent_edge_list(self.kernel_mapping)

        self.child_parent_kernel_mapping = forward_edge_list
        self.parent_kernel_child_mapping = reverse_edge_list

    # interface requirement
    def linearized_potential_parent_poses(self,
                                          input_layer_poses # [batch, *child_dimensions, 1, pose_width, pose_height]
                                          ):

        batch_size, pose_width, pose_height = tf.shape(input_layer_poses)[[0, -2, -1]]

        kernel_mapped_input =self.map_input_layer_to_parent_kernels(input_layer_poses)
        kernel_mapped_weights = self.map_weights_to_parent_kernels(batch_size, pose_width, pose_height)

        # [*kernel_dimensions, *parent_dimensions, pose_width, pose_height]
        potential_parent_poses_map = tf.matmul(kernel_mapped_input, kernel_mapped_weights)

        potential_parent_poses_map = self.add_coordinates(potential_parent_poses_map)

        children_per_kernel = np.prod([len(v) for v in self.child_index_translations_for_kernel])
        parent_count = np.prod([len(v) for v in self.child_index_translations_for_parent])

        linearized_shape = [batch_size, children_per_kernel, parent_count, pose_width, pose_height]

        potential_parent_poses_linearized = tf.reshape(potential_parent_poses_map, linearized_shape)

        return potential_parent_poses_linearized

    def add_coordinates(self, potential_parent_poses_map):
        potential_parent_poses_map_shape = tf.shape(potential_parent_poses_map)
        pose_width, pose_height = potential_parent_poses_map_shape[[-2, -1]]

        for added_coordinate in self.added_coordinates:
            coordinate_source_index = added_coordinate["source"]
            coordinate_sink_index = added_coordinate["sink"]

            coordinate_count = potential_parent_poses_map_shape[coordinate_source_index]
            coordinate_range = tf.range(coordinate_count)
            scaled_coordinate_range = coordinate_range / (coordinate_count - 1)

            base_pose_matrices_shape = [coordinate_count, pose_width, pose_height]
            base_pose_matrices = tf.zeros(base_pose_matrices_shape)

            target_indices = tf.concat([coordinate_range, coordinate_sink_index])

            pose_matrices_with_coordinates = tf.scatter_update(base_pose_matrices, target_indices, scaled_coordinate_range)

            broadcastable_shape = tf.ones([tf.shape(potential_parent_poses_map_shape)[0]])
            broadcastable_shape = tf.scatter_update(broadcastable_shape, [coordinate_source_index, -2, -1], base_pose_matrices_shape)

            broadcastable_pose_matrices_with_coordinates = tf.reshape(pose_matrices_with_coordinates, broadcastable_shape)

            potential_parent_poses_map = potential_parent_poses_map + broadcastable_pose_matrices_with_coordinates

        return potential_parent_poses_map

    def map_input_layer_to_parent_kernels(self, input_layer_poses):

        dimension_count_input = tf.shape(tf.shape(input_layer_poses))[0]

        # move batch dimension to the end

        batch_at_end_permutation = tf.concat([tf.range(1, dimension_count_input), [0]])
        input_layer_poses_for_gather_nd = tf.transpose(input_layer_poses, batch_at_end_permutation)

        # gather input data
        gathered_input_slices = tf.gather_nd(input_layer_poses_for_gather_nd, self.kernel_mapping)

        # move batch dimension to start of input data
        dimension_count_slices = tf.shape(tf.shape(gathered_input_slices))[0]

        kernel_map_permutation = tf.concat([[dimension_count_slices - 1], tf.range(dimension_count_slices - 2)])

        kernel_mapped_slices = tf.transpose(gathered_input_slices, kernel_map_permutation)

        return kernel_mapped_slices

    def map_weights_to_parent_kernels(self, batch_size, pose_width, pose_height):

        complete_weight_shape = [1] + self.weight_shape + [pose_width, pose_height]

        weights = tf.Variable(tf.truncated_normal(complete_weight_shape),
                              dtype=tf.float32, name='pose_transform_weights')

        kernel_tile_dimensions = [batch_size] + self.weight_tiling + [1, 1]

        tiled_weights = tf.tile(weights, kernel_tile_dimensions)

        return tiled_weights

    # interface requirement
    def replace_kernel_elements_with_sum_of_child(self,
                                                  parent_kernel_values  # [batch, kernel_position_linearized, parent_position_linearized, 1]
                                                  ):

        batch_size, linear_kernel_position_size, parent_position_size = tf.shape(parent_kernel_values)[[0, 1, 2]]

        # [batch, kernel_parent_position_linearized]
        fully_linearized_kernel_parent_position = tf.reshape(parent_kernel_values, [batch_size, -1])

        # [kernel_parent_position_linearized, batch]
        parent_kernel_prepared_for_sum_per_child = tf.transpose(fully_linearized_kernel_parent_position)

        linearized_child_to_parent_kernel = self.make_sparse_linearized_child_to_parent_kernel(self.child_parent_kernel_mapping)  # [edge_id, coordinate], coordinate = *child_dimensions + *kernel_dimensions + *parent_dimensions

        linearized_parent_kernel_to_child = self.make_sparse_linearized_parent_kernel_to_child(self.parent_kernel_child_mapping)  # [edge id, coordinate], coordinate = *kernel_dimensions + *parent_dimensions + *child_dimensions

        # multiply by linear_parent_to_linear_child matrix, thus summing the values of all parents of each child:
        # [linearized_child, batch] = [linearized_child, linearized_kernel_parent] sparse dense matmul [linearized_kernel_parent, batch]

        values_summed_per_child = tf.sparse_tensor_dense_matmul(linearized_child_to_parent_kernel, parent_kernel_prepared_for_sum_per_child)

        # multiply by linear_child_to_linear_parent matrix, thus copying the summed values to all the appropriate locations
        # [linearized_parent_kernel, batch] = [linearized_parent_kernel, linearized_child] sparse_dense_matmul [linearized_child, batch]

        child_summed_values_per_kernel_parent_position_transposed = tf.sparse_tensor_dense_matmul(linearized_parent_kernel_to_child, values_summed_per_child)

        linearized_child_summed_values_per_kernel_parent_position = tf.transpose(child_summed_values_per_kernel_parent_position_transposed)

        child_summed_values_per_kernel_parent_position = tf.reshape(linearized_child_summed_values_per_kernel_parent_position, [batch_size, linear_kernel_position_size, parent_position_size, 1])

        return child_summed_values_per_kernel_parent_position

    def make_sparse_linearized_child_to_parent_kernel(self, child_parent_kernel_mapping):
        values = np.array([True] * child_parent_kernel_mapping.shape[0])
        parent_kernel_shape = self.parent_kernel_shape()
        dense_shape = self.children_shape + parent_kernel_shape
        as_sparse_map = tf.SparseTensor(indices=child_parent_kernel_mapping, values=values, dense_shape=dense_shape)

        linearized_shape = [np.product(self.children_shape), np.product(parent_kernel_shape)]

        linearized_child_parent_kernel_mapping = tf.reshape(as_sparse_map, linearized_shape)

        return linearized_child_parent_kernel_mapping

    def make_sparse_linearized_parent_kernel_to_child(self, parent_kernel_child_mapping):
        values = np.array([True] * parent_kernel_child_mapping.shape[0])
        parent_kernel_shape = self.parent_kernel_shape()
        dense_shape = parent_kernel_shape + self.children_shape
        as_sparse_map = tf.SparseTensor(indices=parent_kernel_child_mapping, values=values, dense_shape=dense_shape)

        linearized_shape = [np.product(parent_kernel_shape), np.product(self.children_shape)]

        linearized_parent_kernel_child_mapping = tf.reshape(as_sparse_map, linearized_shape)

        return linearized_parent_kernel_child_mapping

    def parent_kernel_shape(self):
        return [len(x) for x in self.child_index_translations_for_kernel] + [len(y) for y in self.child_index_translations_for_parent]

    # interface requirement
    def reshape_parents_to_map(self, linear_parents):

        [batch_size, parent_count, one, *data_dimensions] = tf.shape(linear_parents)

        parent_map_dimensions = np.prod([len(v) for v in self.child_index_translations_for_parent])

        map_shape = [batch_size, parent_map_dimensions, one, *data_dimensions]

        map_parents = tf.reshape(linear_parents, map_shape)

        return map_parents

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

        broadcastable_array_shape = np.ones(kernel_map_dimensions)
        broadcastable_array_shape[broadcasted_dimension] = array_length

        broadcastable_array_shape[-1] = self.child_index_dimension_count

        broadcastable_indices_shape = [array.shape[0], self.child_index_dimension_count]
        broadcastable_indices = np.zeros(broadcastable_indices_shape)
        broadcastable_indices[:, translated_dimension] = array

        broadcastable_array = broadcastable_indices.copy().reshape(broadcastable_array_shape)

        return broadcastable_array

    # utility
    def _construct_child_to_kernel_mapping(self):
        kernel_map_shape = np.ones([self.kernel_map_dimension_count()])
        kernel_map_shape[-1] = self.child_index_dimension_count

        child_index_for_kernel_position = np.zeros(kernel_map_shape)

        kernel_dimension_count = len(self.child_index_translations_for_kernel)
        for i in range(kernel_dimension_count):
            array = self.child_index_translations_for_kernel[i]
            child_index_for_kernel_position += self._construct_array_for_broadcast_translation(array, i, i)

        parent_dimension_count = len(self.child_index_translations_for_parent)
        for j in range(parent_dimension_count):
            array = self.child_index_translations_for_parent[j]
            child_index_for_kernel_position += self._construct_array_for_broadcast_translation(array, j, kernel_dimension_count + j)

        return child_index_for_kernel_position

    def _convert_index_mapping_to_child_kernel_parent_edge_list(self, gather_nd_index_mapping):

        kernel_parent_map_dimensions = [len(x) for x in self.child_index_translations_for_kernel] + [len(x) for x in self.child_index_translations_for_parent]


        kernel_parent_map_index_list = list(np.ndindex(*kernel_parent_map_dimensions.shape))[:-1]

        enumerated_child_indices = gather_nd_index_mapping[kernel_parent_map_index_list]

        kernel_parent_map_index_array = np.array(kernel_parent_map_index_list)

        child_parent_kernel_mapping = np.concatenate([enumerated_child_indices, kernel_parent_map_index_array], axis=1)
        parent_kernel_child_mapping = np.concatenate([kernel_parent_map_index_array, enumerated_child_indices], axis=1)

        return child_parent_kernel_mapping, parent_kernel_child_mapping

    def _construct_weight_shape(self):

        weight_shape = []

        for i in range(len(self.child_index_translations_for_kernel)):
            weight_shape += 1 if self.shared_weights_kernel else len(self.child_index_translations_for_kernel[i])

        for i in range(len(self.child_index_translations_for_parent)):
            weight_shape += 1 if self.shared_weights_parent else len(self.child_index_translations_for_parent[i])

        return weight_shape

    def _construct_weight_tiling(self):
        tiling_shape = []

        for i in range(len(self.child_index_translations_for_kernel)):
            tiling_shape += len(self.child_index_translations_for_kernel[i]) if self.shared_weights_kernel else 1

        for i in range(len(self.child_index_translations_for_parent)):
            tiling_shape += len(self.child_index_translations_for_parent[i]) if self.shared_weights_parent else 1

        return tiling_shape

    # interface requirement
    def add_spatial_convolution(self, spatial_input_dimensions, kernel_size, stride):
        valid_kernel_space = (kernel_size - 1)
        parents_share_weights = True
        kernels_share_weights = False

        parent_row_count = int((spatial_input_dimensions[0] - valid_kernel_space) / stride)
        parent_column_count = int((spatial_input_dimensions[1] - valid_kernel_space) / stride)

        self.add_convolution(np.arange(kernel_size), np.arange(parent_row_count) * stride, kernels_share_weights, parents_share_weights)
        self.add_convolution(np.arange(kernel_size), np.arange(parent_column_count) * stride, kernels_share_weights, parents_share_weights)

    # interface requirements
    def add_semantic_convolution(self, semantic_input_dimensions, kernel_size, stride):

        valid_kernel_space = 0
        parents_share_weights = True
        kernels_share_weights = False

        parent_row_count = int((semantic_input_dimensions[0] - valid_kernel_space) / stride)
        parent_column_count = int((semantic_input_dimensions[1] - valid_kernel_space) / stride)

        self.add_convolution(np.arange(kernel_size), np.arange(parent_row_count) * stride, kernels_share_weights, parents_share_weights)
        self.add_convolution(np.arange(kernel_size), np.arange(parent_column_count) * stride, kernels_share_weights, parents_share_weights)

    # interface requirement
    def add_dense_connection(self, input_feature_count, output_feature_count):
        parents_share_weights = False
        self.add_convolution(np.arange(input_feature_count), np.zeros([output_feature_count]), parents_share_weights)

    # interface requirement
    def add_aggregation(self, kernel_size, input_feature_count, output_feature_count, coordinate_addition_targets):

        parents_share_weights = False  # irrelevant parameter
        kernels_share_weights = True

        self.add_convolution(np.arange(kernel_size), np.array([0]), kernels_share_weights, parents_share_weights, coordinate_addition_targets[0])
        self.add_convolution(np.arange(kernel_size), np.array([0]), kernels_share_weights, parents_share_weights, coordinate_addition_targets[1])

        self.add_convolution(np.arange(input_feature_count), np.zeros([output_feature_count]))

