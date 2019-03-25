import tensorflow as tf
import numpy as np
from learning.TopologyBuilder import TopologyBuilder
from learning.TopologyBuilder import UnknownInitializerException
import random


class TestTopologyBuilder(tf.test.TestCase):

    def setUp(self):

        self.topology = TopologyBuilder().init()

    def run(self, result=None):
        with tf.variable_scope('something' + str(random.random())) as resource:
            self.resource = resource
            super(tf.test.TestCase, self).run(result)

    def test_convolution_concatenation(self):
        # Possible ramifications: high
        # underlying ideas on how to concatenate convolutions might be flawed requiring complete redesign of the code
        pass

    def test_representation(self):
        # Possible ramifications: high
        # Topologies might be represented in a fundamentally flawed way, requiring redesign of the whole code
        pass

    def test_performance(self):
        # Possible ramifications: high
        # Code might have performance design flaws requiring massive redesigns
        pass

    def test_replace_kernel_elements_with_sum_of_child(self):
        # Possible ramifications: high
        # high risk of performance issues requiring difficult architectural change over many functions, semantic errors
        # could lead to algorithm failure
        # risks:
        #   sparse matrix multiplication is not even remotely fast enough to map input on gpu
        #   input is mapped incorrectly
        pass

    def test_map_input_layer_to_parent_kernels(self):
        # Possible ramifications: high
        # high risk of performance issues requiring difficult architectural change over many functions, semantic errors
        # could lead to algorithm failure
        # risks:
        #   gather_nd is not even remotely fast enough to extract input and is slower than matmul on gpu
        #   input is mapped incorrectly
        pass

    def test_linearized_potential_parent_poses(self):
        # Possible ramifications: medium
        # the risks are low but could potential cause the algorithm to fail
        # risks:
        #   erronous topologies
        #   erronous reshaping

        pass

    def test_add_coordinates(self):
        # Possible ramifications: low
        # low risk at minor performance impact
        # risks:
        #   data not extracted correctly from self.added_coordinates
        #   coordinates not broadcasted to the same cell, correct cell
        #   wrong coordinates broadcasted

        pass

    def test_init(self):
        # Possible ramifications: low
        # Are proper variables initialized?

        self.assertTrue(self.topology is not None, "init must result in TopologyBuilder")

    def test_finish(self):
        # Possible ramifications: low
        # Is builder phase finished properly?
        pass

    def test_make_sparse_linearized_child_to_parent_kernel(self):
        # Possible ramifications: low
        # sparse tensor is not constructed correctly, which could lead to a lot of problems, but this is not very likely
        pass

    def test_make_sparse_linearized_parent_kernel_to_child(self):
        # Possible ramifications: low
        # sparse tensor is not constructed correctly, which could lead to a lot of problems, but this is not very likely
        pass

    def test_parent_kernel_shape(self):
        # Possible ramifications: low
        # wrong shape is constructed, which is very unlikely and will become apparent quickly even without test
        pass

    def test_reshape_parents_to_map(self):
        # Possible ramifications: medium
        # there is a low risk of a fundamental mistake in how parents are converted between linear and mapped index
        pass

    def test_add_convolution(self):
        # Possible ramifications: high
        # There is a medium high risk of a fundamental problem with the representation with very bad consequences
        # throughout the code

        pass

    def test_kernel_map_dimension_count(self):
        # Possible ramifications: low
        pass

    def test__construct_array_for_broadcast_translation(self):
        # Possible ramifications: medium
        # risk of mistakes is reasonable

        pass

    def test__construct_child_to_kernel_mapping(self):
        # Possible ramifications: high
        # high risk of semantic error that leads to algorithm failure
        pass

    def test__convert_index_mapping_to_child_kernel_parent_edge_list(self):
        # Possible ramification: medium
        # medium risk of semantic algorithm that leads to algorithm failure

        pass

    def test__construct_weight_shape(self):
        # Possible ramifications: medium low
        # low risk of semantic error leading to algorithm failure
        pass

    def test__construct_weight_tiling(self):
        # Possible ramifications: medium low
        # low risk of semantic error leading to algorithm failure

        pass

    def test_spatial_convolution_weight_construction(self):
        # name numbers for legibility
        [child_row_count, child_column_count] = [14, 14]

        kernel_size = 3
        stride = 2

        [batch_size, kernel_row_count, kernel_column_count, parent_row_count, parent_column_count, pose_row_count, pose_column_count] =\
        [3,          kernel_size,      kernel_size,         6,                6,                4,              4]

        tiled_weight_shape = [batch_size, kernel_row_count, kernel_column_count, parent_row_count, parent_column_count, pose_row_count, pose_column_count]

        # setup tested topology
        self.topology.add_spatial_convolution([child_row_count, child_column_count], kernel_size, stride)

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": True
        })


        # assert internal state is correct
        self.assertTrue(self.topology.weight_shape == [kernel_row_count, kernel_column_count, 1, 1], "weights must have the correct shape")
        self.assertTrue(self.topology.weight_tiling == [1, 1, parent_row_count, parent_column_count], "weight tiling must be correct")

        # test if weights are constructed correctly, and mapped to kernels correctly
        tiled_weights = self.topology.map_weights_to_parent_kernels(3, 4, 4)

        #                                             b  k  k  p  p  m  m
        self.assertTrue(tiled_weights.get_shape() == [batch_size, kernel_row_count, kernel_column_count, parent_row_count, parent_column_count, pose_row_count, pose_column_count], "weight matrix must have the correct shape")

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # test if weights are initialized and constructed correctly
            w = tiled_weights.eval()
            self.assertTrue(np.all(np.array(w.shape) == np.array(tiled_weight_shape)))
            data_one_batch = w[0, :, :, :, :, :, :]
            data_other_batch = w[1, :, :, :, :, :]

            self.assertTrue(np.all(data_one_batch == data_other_batch), "weights must be tiled across batches")

            data_one_parent_on_row = w[:, :, :, 0, 0, :, :]
            data_other_parent_on_row = w[:, :, :, 1, 0, :, :]

            self.assertTrue(np.all(data_one_parent_on_row == data_other_parent_on_row), "weights must be tiled across parent row")

            data_one_parent_on_column = w[:, :, :, 1, 0, :, :]
            data_other_parent_on_column = w[:, :, :, 1, 1, :, :]

            self.assertTrue(np.all(data_one_parent_on_column == data_other_parent_on_column),
                            "weights must be tiled across parent column")

    def test_spatial_convolution_input_mapping_internal_state(self):
        [child_row_count, child_column_count] = [14, 14]

        kernel_size = 3
        stride = 2

        [batch_size, kernel_row_count, kernel_column_count, parent_row_count, parent_column_count, pose_row_count, pose_column_count] =\
        [3,          kernel_size,      kernel_size,         6,                6,                4,              4]

        # setup tested topology
        self.topology.add_spatial_convolution([child_row_count, child_column_count], kernel_size, stride)

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": True
        })

        # assert that each child has the correct number of parents
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[0, 1]]), axis=1)) == 1)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[1, 0]]), axis=1)) == 1)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[1, 1]]), axis=1)) == 1)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[2, 0]]), axis=1)) == 2)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[2, 2]]), axis=1)) == 4)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[2, 3]]), axis=1)) == 2)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[3, 3]]), axis=1)) == 1)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[5, 5]]), axis=1)) == 1)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[4, 4]]), axis=1)) == 4)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[5, 4]]), axis=1)) == 2)

    def test_spatial_convolution_input_mapping_pose_to_kernel_projection(self):
        [child_row_count, child_column_count] = [14, 14]

        kernel_size = 3
        stride = 2

        [batch_size, kernel_row_count, kernel_column_count, parent_row_count, parent_column_count, pose_row_count, pose_column_count] =\
        [3,          kernel_size,      kernel_size,         6,                6,                4,              4]

        # setup tested topology
        self.topology.add_spatial_convolution([child_row_count, child_column_count], kernel_size, stride)

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": True
        })

        # test if input is mapped to kernel parent space correctly
        input_template = np.zeros([batch_size, child_row_count, child_column_count, pose_row_count, pose_column_count])

        row_values = np.arange(child_row_count).reshape([1, child_row_count, 1, 1, 1])
        column_values = np.arange(child_column_count).reshape([1, 1, child_column_count, 1, 1]) * 100
        batch_values = np.arange(batch_size).reshape([batch_size, 1, 1, 1, 1]) * 100 * 100

        input_layer_traceable_values = input_template + row_values + column_values + batch_values
        # [batch, row, column, m, m]
        input_layer = tf.constant(input_layer_traceable_values.astype(np.float32))

        mapped_input = self.topology.map_input_layer_to_parent_kernels(input_layer)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # test if input is mapped to parent kernel space correctly
            # setup variables
            full_input_map = mapped_input.eval()
            i = full_input_map[:, :, :, :, :, 0, 0]
            correct_first_kernel_values = np.array([[0, 100, 200], [1, 101, 201], [2, 102, 202]])
            actual_first_kernel_values = i[0, :, :, 0, 0]

            self.assertTrue(np.all(actual_first_kernel_values == correct_first_kernel_values),
                            "first kernel must be mapped accurately")

            correct_next_row_kernel_values = correct_first_kernel_values + 2  # next kernel must be two further because stride 2
            actual_next_row_kernel_values = i[0, :, :, 1, 0]

            self.assertTrue(np.all(actual_next_row_kernel_values == correct_next_row_kernel_values),
                            "next row kernel must be mapped accurately")

            correct_next_column_kernel_values = correct_next_row_kernel_values + 200  # next kernel must be two further because stride 2
            actual_next_column_kernel_values = i[0, :, :, 1, 1]

            self.assertTrue(np.all(actual_next_column_kernel_values == correct_next_column_kernel_values),
                            "next column kernel must be mapped accurately")

            correct_next_batch_kernel_values = correct_next_column_kernel_values + 10000  # batch stride is 1
            actual_next_batch_kernel_values = i[1, :, :, 1, 1]

            self.assertTrue(np.all(actual_next_batch_kernel_values == correct_next_batch_kernel_values),
                            "next batch kernel must be mapped accurately")

    def test_spatial_convolution_child_summing_logic(self):
        [child_row_count, child_column_count] = [14, 14]

        kernel_size = 3
        stride = 2

        [batch_size, kernel_row_count, kernel_column_count, parent_row_count, parent_column_count, pose_row_count, pose_column_count] =\
        [3,          kernel_size,      kernel_size,         6,                6,                4,              4]

        # setup tested topology
        self.topology.add_spatial_convolution([child_row_count, child_column_count], kernel_size, stride)

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": True
        })

        # test if input is mapped to kernel parent space correctly
        input_template = np.zeros([batch_size, child_row_count, child_column_count, pose_row_count, pose_column_count])

        row_values = np.arange(child_row_count).reshape([1, child_row_count, 1, 1, 1])
        column_values = np.arange(child_column_count).reshape([1, 1, child_column_count, 1, 1]) * 100
        batch_values = np.arange(batch_size).reshape([batch_size, 1, 1, 1, 1]) * 100 * 100

        input_layer_traceable_values = input_template + row_values + column_values + batch_values
        # [batch, row, column, m, m]
        input_layer = tf.constant(input_layer_traceable_values.astype(np.float32))

        mapped_input = self.topology.map_input_layer_to_parent_kernels(input_layer)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # test if input is mapped to parent kernel space correctly
            # setup variables
            full_input_map = mapped_input.eval()

            m = self.topology.make_sparse_linearized_child_to_parent_kernel(self.topology.child_parent_kernel_mapping)

            dense_linearized_child_to_parent_kernel = tf.sparse_to_dense(m.indices, m.dense_shape, m.values).eval()

            self.assertTrue(np.sum(dense_linearized_child_to_parent_kernel) == self.topology.child_parent_kernel_mapping.shape[0])

            # mock assuming weights are all identity matrices
            linearized_kernel_parent_mock = self.topology._linearize_potential_parent_poses_map(tf.constant(full_input_map)).eval()[:, :, :, 0, 0]
            mapped_kernel_parent_mock = full_input_map[:, :, :, :, :, 0, 0]

            children_sum = self.topology._compute_sum_for_children(tf.constant(linearized_kernel_parent_mock)).eval()

            mapped_children_sum = np.reshape(children_sum, input_template.shape[:3])

            # a corner child is projected to only one parent
            self.assertTrue(np.all(mapped_children_sum[:, 0, 0] == mapped_kernel_parent_mock[:, 0, 0, 0, 0]))
            # with a stride of 2 the next value is projected to two parents
            self.assertTrue(np.all(mapped_children_sum[:, 1, 0] == mapped_kernel_parent_mock[:, 1, 0, 0, 0]))

            # the next node is projected to only one parent
            self.assertTrue(np.all(mapped_children_sum[:, 2, 0] == 2.0 * mapped_kernel_parent_mock[:, 2, 0, 0, 0]))

            # The same holds for columns
            self.assertTrue(np.all(mapped_children_sum[:, 0, 1] == mapped_kernel_parent_mock[:, 0, 1, 0, 0]))
            self.assertTrue(np.all(mapped_children_sum[:, 0, 2] == 2.0 * mapped_kernel_parent_mock[:, 0, 2, 0, 0]))

            # nodes at 1 1 exist in 4 kernels
            self.assertTrue(np.all(mapped_children_sum[:, 1, 1] == mapped_kernel_parent_mock[:, 1, 1, 0, 0]))

            # nodes at 2 2 exist in only one kernel surprisingly
            self.assertTrue(np.all(mapped_children_sum[:, 2, 2] == 4.0 * mapped_kernel_parent_mock[:, 2, 2, 0, 0]))

            # but nodes at 3 3 exist in 4 kernels again
            self.assertTrue(np.all(mapped_children_sum[:, 3, 3] == mapped_kernel_parent_mock[:, 1, 1, 1, 1]))

            projected_children_sums = self.topology._project_child_scalars_to_parent_kernels(children_sum, tf.shape(tf.constant(linearized_kernel_parent_mock))).eval()

            projected_children_sums_map = np.reshape(projected_children_sums, [batch_size, kernel_row_count, kernel_column_count, parent_row_count, parent_column_count])

            self.assertTrue(np.all(projected_children_sums_map[:, 0, 0, 0, 0] == mapped_children_sum[:, 0, 0]))

            self.assertTrue(np.all(projected_children_sums_map[:, 1, 0, 0, 0] == mapped_children_sum[:, 1, 0]))
            self.assertTrue(np.all(projected_children_sums_map[:, 0, 0, 1, 0] == mapped_children_sum[:, 2, 0]))

            self.assertTrue(np.all(projected_children_sums_map[:, 2, 2, 0, 0] == mapped_children_sum[:, 2, 2]))
            self.assertTrue(np.all(projected_children_sums_map[:, 0, 0, 1, 1] == mapped_children_sum[:, 2, 2]))
            self.assertTrue(np.all(projected_children_sums_map[:, 0, 2, 1, 0] == mapped_children_sum[:, 2, 2]))
            self.assertTrue(np.all(projected_children_sums_map[:, 2, 0, 0, 1] == mapped_children_sum[:, 2, 2]))


    def test_semantic_convolution_weight_construction(self):
        # Possible ramifications: medium low
        # medium low risk of some semantic mistake in how convolution is constructed
        [child_row_count, child_column_count] = [14, 14]

        kernel_size = 3
        stride = 2

        [batch_size, kernel_row_count, kernel_column_count, parent_row_count, parent_column_count, pose_row_count, pose_column_count] =\
        [3,          kernel_size,      kernel_size,         7,                7,                4,              4]

        tiled_weight_shape = [batch_size, kernel_row_count, kernel_column_count, parent_row_count, parent_column_count, pose_row_count, pose_column_count]

        # setup tested topology
        self.topology.add_semantic_convolution([child_row_count, child_column_count], kernel_size, stride)

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": True
        })

        # assert internal state is correct
        self.assertTrue(self.topology.weight_shape == [kernel_row_count, kernel_column_count, parent_row_count, parent_column_count], "weights must have the correct shape")
        self.assertTrue(self.topology.weight_tiling == [1, 1, 1, 1], "weight tiling must be correct")

        # test if weights are constructed correctly, and mapped to kernels correctly
        tiled_weights = self.topology.map_weights_to_parent_kernels(3, 4, 4)

        #                                             b  k  k  p  p  m  m
        self.assertTrue(
            tiled_weights.get_shape() == [batch_size, kernel_row_count, kernel_column_count, parent_row_count,
                                          parent_column_count, pose_row_count, pose_column_count],
            "weight matrix must have the correct shape")

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # test if weights are initialized and constructed correctly
            w = tiled_weights.eval()
            self.assertTrue(np.all(np.array(w.shape) == np.array(tiled_weight_shape)))
            data_one_batch = w[0, :, :, :, :, :, :]
            data_other_batch = w[1, :, :, :, :, :]

            self.assertTrue(np.all(data_one_batch == data_other_batch), "weights must be tiled across batches")

            data_one_parent_on_row = w[:, :, :, 0, 0, :, :]
            data_other_parent_on_row = w[:, :, :, 1, 0, :, :]

            self.assertTrue(np.sum(data_one_parent_on_row == data_other_parent_on_row) == 0, "weights must not be tiled across parent row")

            data_one_parent_on_column = w[:, :, :, 1, 0, :, :]
            data_other_parent_on_column = w[:, :, :, 1, 1, :, :]

            self.assertTrue(np.sum(data_one_parent_on_column == data_other_parent_on_column) == 0,
                            "weights must not be tiled across parent column")

    def test_semantic_convolution_input_mapping_internal_state(self):
        [child_row_count, child_column_count] = [14, 14]

        kernel_size = 3
        stride = 2

        [batch_size, kernel_row_count, kernel_column_count, parent_row_count, parent_column_count, pose_row_count, pose_column_count] =\
        [3,          kernel_size,      kernel_size,         7,                7,                4,              4]

        # setup tested topology
        self.topology.add_semantic_convolution([child_row_count, child_column_count], kernel_size, stride)

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": True
        })

        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[0, 0]]), axis=1)) == 4)  # due to wrapping 4
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[0, 1]]), axis=1)) == 2)  # due to wrapping 2
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[1, 0]]), axis=1)) == 2)  # due to wrapping 2
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[1, 1]]), axis=1)) == 1)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[2, 0]]), axis=1)) == 4)  # due to wrapping 4
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[2, 2]]), axis=1)) == 4)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[2, 3]]), axis=1)) == 2)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[3, 3]]), axis=1)) == 1)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[5, 5]]), axis=1)) == 1)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[4, 4]]), axis=1)) == 4)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[5, 4]]), axis=1)) == 2)

    def test_semantic_convolution_input_mapping_pose_to_kernel_projection(self):
        [child_row_count, child_column_count] = [14, 14]

        kernel_size = 3
        stride = 2

        [batch_size, kernel_row_count, kernel_column_count, parent_row_count, parent_column_count, pose_row_count, pose_column_count] =\
        [3,          kernel_size,      kernel_size,         7,                7,                4,              4]

        # setup tested topology
        self.topology.add_semantic_convolution([child_row_count, child_column_count], kernel_size, stride)

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": True
        })

        # test if input is mapped to kernel parent space correctly
        input_template = np.zeros([batch_size, child_row_count, child_column_count, pose_row_count, pose_column_count])

        row_values = np.arange(child_row_count).reshape([1, child_row_count, 1, 1, 1])
        column_values = np.arange(child_column_count).reshape([1, 1, child_column_count, 1, 1]) * 100
        batch_values = np.arange(batch_size).reshape([batch_size, 1, 1, 1, 1]) * 100 * 100

        input_layer_traceable_values = input_template + row_values + column_values + batch_values
        # [batch, row, column, m, m]
        input_layer = tf.constant(input_layer_traceable_values.astype(np.float32))

        mapped_input = self.topology.map_input_layer_to_parent_kernels(input_layer)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # assert that each child has the correct number of parents

            # test if input is mapped to parent kernel space correctly
            # setup variables
            full_input_map = mapped_input.eval()
            i = full_input_map[:, :, :, :, :, 0, 0]
            correct_first_kernel_values = np.array([[0, 100, 200], [1, 101, 201], [2, 102, 202]])
            actual_first_kernel_values = i[0, :, :, 0, 0]

            self.assertTrue(np.all(actual_first_kernel_values == correct_first_kernel_values), "first kernel must be mapped accurately")

            correct_next_row_kernel_values = correct_first_kernel_values + 2  # next kernel must be two further because stride 2
            actual_next_row_kernel_values = i[0, :, :, 1, 0]

            self.assertTrue(np.all(actual_next_row_kernel_values == correct_next_row_kernel_values), "next row kernel must be mapped accurately")

            correct_next_column_kernel_values = correct_next_row_kernel_values + 200  # next kernel must be two further because stride 2
            actual_next_column_kernel_values = i[0, :, :, 1, 1]

            self.assertTrue(np.all(actual_next_column_kernel_values == correct_next_column_kernel_values),
                            "next column kernel must be mapped accurately")

            correct_next_batch_kernel_values = correct_next_column_kernel_values + 10000  # batch stride is 1
            actual_next_batch_kernel_values = i[1, :, :, 1, 1]

            self.assertTrue(np.all(actual_next_batch_kernel_values == correct_next_batch_kernel_values),
                            "next batch kernel must be mapped accurately")

            correct_last_kernel_values = np.array([[1212, 1312, 12], [1213, 1313, 13], [1200, 1300, 0]])
            actual_last_kernel_values = i[0, :, :, 6, 6]

            self.assertTrue(np.all(actual_last_kernel_values == correct_last_kernel_values),
                            "last kernel must be wrap round correctly")

    def test_semantic_convolution_child_summing_logic(self):
        [child_row_count, child_column_count] = [14, 14]

        kernel_size = 3
        stride = 2

        [batch_size, kernel_row_count, kernel_column_count, parent_row_count, parent_column_count, pose_row_count,
         pose_column_count] = \
            [3, kernel_size, kernel_size, 7, 7, 4, 4]

        # setup tested topology
        self.topology.add_semantic_convolution([child_row_count, child_column_count], kernel_size, stride)

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": True
        })

        # test if input is mapped to kernel parent space correctly
        input_template = np.zeros([batch_size, child_row_count, child_column_count, pose_row_count, pose_column_count])

        row_values = np.arange(child_row_count).reshape([1, child_row_count, 1, 1, 1])
        column_values = np.arange(child_column_count).reshape([1, 1, child_column_count, 1, 1]) * 100
        batch_values = np.arange(batch_size).reshape([batch_size, 1, 1, 1, 1]) * 100 * 100

        input_layer_traceable_values = input_template + row_values + column_values + batch_values
        # [batch, row, column, m, m]
        input_layer = tf.constant(input_layer_traceable_values.astype(np.float32))

        mapped_input = self.topology.map_input_layer_to_parent_kernels(input_layer)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            full_input_map = mapped_input.eval()

            m = self.topology.make_sparse_linearized_child_to_parent_kernel(self.topology.child_parent_kernel_mapping)

            dense_linearized_child_to_parent_kernel = tf.sparse_to_dense(m.indices, m.dense_shape, m.values).eval()

            self.assertTrue(np.sum(dense_linearized_child_to_parent_kernel) == self.topology.child_parent_kernel_mapping.shape[0])

            # mock assuming weights are all identity matrices
            linearized_kernel_parent_mock = self.topology._linearize_potential_parent_poses_map(tf.constant(full_input_map)).eval()[:, :, :, 0, 0]
            mapped_kernel_parent_mock = full_input_map[:, :, :, :, :, 0, 0]

            children_sum = self.topology._compute_sum_for_children(tf.constant(linearized_kernel_parent_mock)).eval()

            mapped_children_sum = np.reshape(children_sum, input_template.shape[:3])

            # a corner child is projected to only one parent
            self.assertTrue(np.all(mapped_children_sum[:, 0, 0] == 4.0 * mapped_kernel_parent_mock[:, 0, 0, 0, 0]))
            # with a stride of 2 the next value is projected to two parents
            self.assertTrue(np.all(mapped_children_sum[:, 1, 0] == 2.0 * mapped_kernel_parent_mock[:, 1, 0, 0, 0]))

            # the next node is projected to only one parent
            self.assertTrue(np.all(mapped_children_sum[:, 2, 0] == 4.0 * mapped_kernel_parent_mock[:, 2, 0, 0, 0]))

            # The same holds for columns
            self.assertTrue(np.all(mapped_children_sum[:, 0, 1] == 2.0 * mapped_kernel_parent_mock[:, 0, 1, 0, 0]))
            self.assertTrue(np.all(mapped_children_sum[:, 0, 2] == 4.0 * mapped_kernel_parent_mock[:, 0, 2, 0, 0]))

            # nodes at 1 1 exist in 4 kernels
            self.assertTrue(np.all(mapped_children_sum[:, 1, 1] == mapped_kernel_parent_mock[:, 1, 1, 0, 0]))

            # nodes at 2 2 exist in only one kernel surprisingly
            self.assertTrue(np.all(mapped_children_sum[:, 2, 2] == 4.0 * mapped_kernel_parent_mock[:, 2, 2, 0, 0]))

            # but nodes at 3 3 exist in 4 kernels again
            self.assertTrue(np.all(mapped_children_sum[:, 3, 3] == mapped_kernel_parent_mock[:, 1, 1, 1, 1]))

            projected_children_sums = self.topology._project_child_scalars_to_parent_kernels(children_sum, tf.shape(tf.constant(linearized_kernel_parent_mock))).eval()

            projected_children_sums_map = np.reshape(projected_children_sums, [batch_size, kernel_row_count, kernel_column_count, parent_row_count, parent_column_count])

            self.assertTrue(np.all(projected_children_sums_map[:, 0, 0, 0, 0] == mapped_children_sum[:, 0, 0]))

            self.assertTrue(np.all(projected_children_sums_map[:, 1, 0, 0, 0] == mapped_children_sum[:, 1, 0]))
            self.assertTrue(np.all(projected_children_sums_map[:, 0, 0, 1, 0] == mapped_children_sum[:, 2, 0]))

            self.assertTrue(np.all(projected_children_sums_map[:, 2, 2, 0, 0] == mapped_children_sum[:, 2, 2]))
            self.assertTrue(np.all(projected_children_sums_map[:, 0, 0, 1, 1] == mapped_children_sum[:, 2, 2]))
            self.assertTrue(np.all(projected_children_sums_map[:, 0, 2, 1, 0] == mapped_children_sum[:, 2, 2]))
            self.assertTrue(np.all(projected_children_sums_map[:, 2, 0, 0, 1] == mapped_children_sum[:, 2, 2]))

            self.assertTrue(np.all(projected_children_sums_map[:, 1, 1, 6, 6] == mapped_children_sum[:, 13, 13]))
            self.assertTrue(np.all(projected_children_sums_map[:, 2, 2, 6, 6] == mapped_children_sum[:, 0, 0]))

    def test_aggregation_weight_construction(self):
        # name numbers for legibility
        [child_row_count, child_column_count] = [14, 14]

        kernel_size = 14

        [batch_size, kernel_row_count, kernel_column_count, parent_row_count, parent_column_count, pose_row_count, pose_column_count] =\
        [3,          kernel_size,      kernel_size,         1,                1,                4,              4]

        tiled_weight_shape = [batch_size, kernel_row_count, kernel_column_count, parent_row_count, parent_column_count, pose_row_count, pose_column_count]

        # setup tested topology
        self.topology.add_aggregation(kernel_size, [[3, 0], [3, 1]])

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": True
        })


        # assert internal state is correct
        self.assertTrue(self.topology.weight_shape == [1, 1, 1, 1], "weights must have the correct shape")
        self.assertTrue(self.topology.weight_tiling == [kernel_size, kernel_size, 1, 1], "weight tiling must be correct")

        # test if weights are constructed correctly, and mapped to kernels correctly
        tiled_weights = self.topology.map_weights_to_parent_kernels(3, 4, 4)

        #                                             b  k  k  p  p  m  m
        self.assertTrue(tiled_weights.get_shape() == [batch_size, kernel_row_count, kernel_column_count, 1, 1, pose_row_count, pose_column_count], "weight matrix must have the correct shape")

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # test if weights are initialized and constructed correctly
            w = tiled_weights.eval()
            self.assertTrue(np.all(np.array(w.shape) == np.array(tiled_weight_shape)))
            data_one_batch = w[0, :, :, :, :, :, :]
            data_other_batch = w[1, :, :, :, :, :]

            self.assertTrue(np.all(data_one_batch == data_other_batch), "weights must be tiled across batches")

            data_one_parent_on_row = w[:, 0, 0, :, :, :, :]
            data_other_parent_on_row = w[:, 1, 0, :, :, :, :]

            self.assertTrue(np.all(data_one_parent_on_row == data_other_parent_on_row), "weights must be tiled across kernel row")

            data_one_parent_on_column = w[:, 1, 0, :, :, :, :]
            data_other_parent_on_column = w[:, 1, 1, :, :, :, :]

            self.assertTrue(np.all(data_one_parent_on_column == data_other_parent_on_column),
                            "weights must be tiled across kernel column")

    def test_aggregation_input_mapping_internal_state(self):
        [child_row_count, child_column_count] = [14, 14]

        kernel_size = 14

        [batch_size, kernel_row_count, kernel_column_count, parent_row_count, parent_column_count, pose_row_count, pose_column_count] =\
        [3,          kernel_size,      kernel_size,         1,                1,                4,              4]

        # setup tested topology
        self.topology.add_aggregation(kernel_size, [[3, 0], [3, 1]])

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": True
        })

        # assert that each child has the correct number of parents; aggregations have only 1 parent by definition unless another convolution increases the number of parents
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[0, 1]]), axis=1)) == 1)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[1, 0]]), axis=1)) == 1)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[1, 1]]), axis=1)) == 1)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[2, 0]]), axis=1)) == 1)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[2, 2]]), axis=1)) == 1)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[2, 3]]), axis=1)) == 1)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[3, 3]]), axis=1)) == 1)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[5, 5]]), axis=1)) == 1)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[4, 4]]), axis=1)) == 1)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[5, 4]]), axis=1)) == 1)

    def test_aggregation_input_mapping_pose_to_kernel_projection(self):
        [child_row_count, child_column_count] = [14, 14]

        kernel_size = 14

        [batch_size, kernel_row_count, kernel_column_count, parent_row_count, parent_column_count, pose_row_count, pose_column_count] =\
        [3,          kernel_size,      kernel_size,         1,                1,                4,              4]

        # setup tested topology
        self.topology.add_aggregation(kernel_size, [[3, 0], [3, 1]])

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": True
        })

        # test if input is mapped to kernel parent space correctly
        input_template = np.zeros([batch_size, child_row_count, child_column_count, pose_row_count, pose_column_count])

        row_values = np.arange(child_row_count).reshape([1, child_row_count, 1, 1, 1])
        column_values = np.arange(child_column_count).reshape([1, 1, child_column_count, 1, 1]) * 100
        batch_values = np.arange(batch_size).reshape([batch_size, 1, 1, 1, 1]) * 100 * 100

        input_layer_traceable_values = input_template + row_values + column_values + batch_values
        # [batch, row, column, m, m]
        input_layer = tf.constant(input_layer_traceable_values.astype(np.float32))

        mapped_input = self.topology.map_input_layer_to_parent_kernels(input_layer)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # test if input is mapped to parent kernel space correctly
            # setup variables
            full_input_map = mapped_input.eval()
            i = full_input_map[:, :, :, :, :, 0, 0]
            correct_first_kernel_values = input_layer_traceable_values[0, :, :, 0, 0]
            actual_first_kernel_values = i[0, :, :, 0, 0]

            self.assertTrue(np.all(actual_first_kernel_values == correct_first_kernel_values),
                            "first kernel must be mapped accurately")

            correct_next_batch_kernel_values = input_layer_traceable_values[1, :, :, 1, 1]
            actual_next_batch_kernel_values = i[1, :, :, 0, 0]

            self.assertTrue(np.all(actual_next_batch_kernel_values == correct_next_batch_kernel_values),
                            "next batch kernel must be mapped accurately")

    def test_aggregation_child_summing_logic(self):
        [child_row_count, child_column_count] = [14, 14]

        kernel_size = 14

        [batch_size, kernel_row_count, kernel_column_count, parent_row_count, parent_column_count, pose_row_count, pose_column_count] =\
        [3,          kernel_size,      kernel_size,         1,                1,                4,              4]

        # setup tested topology
        self.topology.add_aggregation(kernel_size, [[3, 0], [3, 1]])

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": True
        })


        # test if input is mapped to kernel parent space correctly
        input_template = np.zeros([batch_size, child_row_count, child_column_count, pose_row_count, pose_column_count])

        row_values = np.arange(child_row_count).reshape([1, child_row_count, 1, 1, 1])
        column_values = np.arange(child_column_count).reshape([1, 1, child_column_count, 1, 1]) * 100
        batch_values = np.arange(batch_size).reshape([batch_size, 1, 1, 1, 1]) * 100 * 100

        input_layer_traceable_values = input_template + row_values + column_values + batch_values
        # [batch, row, column, m, m]
        input_layer = tf.constant(input_layer_traceable_values.astype(np.float32))

        mapped_input = self.topology.map_input_layer_to_parent_kernels(input_layer)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # test if input is mapped to parent kernel space correctly
            # setup variables
            full_input_map = mapped_input.eval()

            m = self.topology.make_sparse_linearized_child_to_parent_kernel(self.topology.child_parent_kernel_mapping)

            dense_linearized_child_to_parent_kernel = tf.sparse_to_dense(m.indices, m.dense_shape, m.values).eval()

            self.assertTrue(np.sum(dense_linearized_child_to_parent_kernel) == self.topology.child_parent_kernel_mapping.shape[0])

            # mock assuming weights are all identity matrices
            linearized_kernel_parent_mock = self.topology._linearize_potential_parent_poses_map(tf.constant(full_input_map)).eval()[:, :, :, 0, 0]
            mapped_kernel_parent_mock = full_input_map[:, :, :, :, :, 0, 0]

            children_sum = self.topology._compute_sum_for_children(tf.constant(linearized_kernel_parent_mock)).eval()

            mapped_children_sum = np.reshape(children_sum, input_template.shape[:3])

            # a corner child is projected to only one parent
            self.assertTrue(np.all(mapped_children_sum[:, 0, 0] == mapped_kernel_parent_mock[:, 0, 0, 0, 0]))
            # with a stride of 2 the next value is projected to two parents
            self.assertTrue(np.all(mapped_children_sum[:, 1, 0] == mapped_kernel_parent_mock[:, 1, 0, 0, 0]))

            # the next node is projected to only one parent
            self.assertTrue(np.all(mapped_children_sum[:, 2, 0] == mapped_kernel_parent_mock[:, 2, 0, 0, 0]))

            # The same holds for columns
            self.assertTrue(np.all(mapped_children_sum[:, 0, 1] == mapped_kernel_parent_mock[:, 0, 1, 0, 0]))
            self.assertTrue(np.all(mapped_children_sum[:, 0, 2] == mapped_kernel_parent_mock[:, 0, 2, 0, 0]))

            # nodes at 1 1 exist in 4 kernels
            self.assertTrue(np.all(mapped_children_sum[:, 1, 1] == mapped_kernel_parent_mock[:, 1, 1, 0, 0]))

            # nodes at 2 2 exist in only one kernel surprisingly
            self.assertTrue(np.all(mapped_children_sum[:, 2, 2] == mapped_kernel_parent_mock[:, 2, 2, 0, 0]))

            # but nodes at 3 3 exist in 4 kernels again
            self.assertTrue(np.all(mapped_children_sum[:, 13, 13] == mapped_kernel_parent_mock[:, 13, 13, 0, 0]))

            projected_children_sums = self.topology._project_child_scalars_to_parent_kernels(children_sum, tf.shape(tf.constant(linearized_kernel_parent_mock))).eval()

            projected_children_sums_map = np.reshape(projected_children_sums, [batch_size, kernel_row_count, kernel_column_count, parent_row_count, parent_column_count])

            self.assertTrue(np.all(projected_children_sums_map[:, 0, 0, 0, 0] == mapped_children_sum[:, 0, 0]))

            self.assertTrue(np.all(projected_children_sums_map[:, 1, 0, 0, 0] == mapped_children_sum[:, 1, 0]))
            self.assertTrue(np.all(projected_children_sums_map[:, 0, 1, 0, 0] == mapped_children_sum[:, 0, 1]))

            self.assertTrue(np.all(projected_children_sums_map[:, 2, 2, 0, 0] == mapped_children_sum[:, 2, 2]))
            self.assertTrue(np.all(projected_children_sums_map[:, 0, 2, 0, 0] == mapped_children_sum[:, 0, 2]))
            self.assertTrue(np.all(projected_children_sums_map[:, 2, 0, 0, 0] == mapped_children_sum[:, 2, 0]))

    def test_dense_weight_construction(self):
        # name numbers for legibility
        input_count, output_count = 4, 3
        batch_size = 2

        pose_row_count = 4
        pose_column_count = 4

        tiled_weight_shape = [batch_size, input_count, output_count, pose_row_count, pose_column_count]

        # setup tested topology
        self.topology.add_dense_connection(input_count, output_count)

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": True
        })

        # assert internal state is correct
        self.assertTrue(self.topology.weight_shape == [input_count, output_count], "weights must have the correct shape")
        self.assertTrue(self.topology.weight_tiling == [1, 1], "weight tiling must be correct")

        # test if weights are constructed correctly, and mapped to kernels correctly
        tiled_weights = self.topology.map_weights_to_parent_kernels(batch_size, 4, 4)

        #                                             b  k  k  p  p  m  m
        self.assertTrue(tiled_weights.get_shape() == [batch_size, input_count, output_count, pose_row_count, pose_column_count], "weight matrix must have the correct shape")

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # test if weights are initialized and constructed correctly
            w = tiled_weights.eval()
            self.assertTrue(np.all(np.array(w.shape) == np.array(tiled_weight_shape)))
            data_one_batch = w[0, :, :, :, :]
            data_other_batch = w[1, :, :, :]

            self.assertTrue(np.all(data_one_batch == data_other_batch), "weights must be tiled across batches")

            data_one_parent_on_row = w[:, 0, 0, :, :]
            data_other_parent_on_row = w[:, 1, 0, :, :]

            self.assertTrue(np.sum(data_one_parent_on_row == data_other_parent_on_row) == 0, "weights must not be tiled across output")

            data_one_parent_on_column = w[:, 1, 0, :, :]
            data_other_parent_on_column = w[:, 1, 1, :, :]

            self.assertTrue(np.sum(data_one_parent_on_column == data_other_parent_on_column) == 0,
                            "weights must not be tiled across input")

    def test_dense_input_mapping_internal_state(self):
        # name numbers for legibility
        input_count, output_count = 4, 3

        # setup tested topology
        self.topology.add_dense_connection(input_count, output_count)

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": True
        })

        # assert that each child has the correct number of parents
        self.assertTrue(np.sum(self.topology.child_parent_kernel_mapping[:, 0] == 0) == output_count, "child is connected to output_count outputs")
        self.assertTrue(np.sum(self.topology.child_parent_kernel_mapping[:, 1] == 1) == output_count, "kernel is connected to output_count outputs" )
        self.assertTrue(np.sum(self.topology.child_parent_kernel_mapping[:, 2] == 2) == input_count, "parent is connected to input_count inputs" )


    def test_dense_input_mapping_pose_to_kernel_projection(self):
        # name numbers for legibility
        input_count, output_count = 4, 3
        batch_size = 2
        pose_row_count, pose_column_count = 4, 4
        # setup tested topology
        self.topology.add_dense_connection(input_count, output_count)

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": True
        })

        # test if input is mapped to kernel parent space correctly
        input_template = np.zeros([batch_size, input_count, pose_row_count, pose_column_count])

        input_values = np.arange(input_count).reshape([1, input_count, 1, 1])
        batch_values = np.arange(batch_size).reshape([batch_size, 1, 1, 1]) * 10

        input_layer_traceable_values = input_template + input_values + batch_values
        # [batch, row, column, m, m]
        input_layer = tf.constant(input_layer_traceable_values.astype(np.float32))

        mapped_input = self.topology.map_input_layer_to_parent_kernels(input_layer)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # test if input is mapped to parent kernel space correctly
            # setup variables
            full_input_map = mapped_input.eval()
            i = full_input_map[:, :, :, 0, 0]
            correct_first_kernel_values = np.arange(input_count)
            actual_first_kernel_values = i[0, :, 0]

            self.assertTrue(np.all(actual_first_kernel_values == correct_first_kernel_values),
                            "first kernel must be mapped accurately")

            correct_second_kernel_values = np.arange(input_count)
            actual_second_kernel_values = i[0, :, 1]

            self.assertTrue(np.all(actual_second_kernel_values == correct_second_kernel_values),
                            "next row kernel must be mapped accurately")

    def test_dense_child_summing_logic(self):
        # name numbers for legibility
        input_count, output_count = 4, 3
        batch_size = 2
        pose_row_count, pose_column_count = 4, 4
        # setup tested topology
        self.topology.add_dense_connection(input_count, output_count)

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": True
        })

        # test if input is mapped to kernel parent space correctly
        input_template = np.zeros([batch_size, input_count, pose_row_count, pose_column_count])

        input_values = np.arange(input_count).reshape([1, input_count, 1, 1])
        batch_values = np.arange(batch_size).reshape([batch_size, 1, 1, 1]) * 10

        input_layer_traceable_values = input_template + input_values + batch_values
        # [batch, row, column, m, m]
        input_layer = tf.constant(input_layer_traceable_values.astype(np.float32))

        mapped_input = self.topology.map_input_layer_to_parent_kernels(input_layer)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # test if input is mapped to parent kernel space correctly
            # setup variables
            full_input_map = mapped_input.eval()

            m = self.topology.make_sparse_linearized_child_to_parent_kernel(self.topology.child_parent_kernel_mapping)

            dense_linearized_child_to_parent_kernel = tf.sparse_to_dense(m.indices, m.dense_shape, m.values).eval()

            self.assertTrue(np.sum(dense_linearized_child_to_parent_kernel) == self.topology.child_parent_kernel_mapping.shape[0])

            # mock assuming weights are all identity matrices
            linearized_kernel_parent_mock = self.topology._linearize_potential_parent_poses_map(tf.constant(full_input_map)).eval()[:, :, :, 0, 0]
            mapped_kernel_parent_mock = full_input_map[:, :, :, 0, 0]

            children_sum = self.topology._compute_sum_for_children(tf.constant(linearized_kernel_parent_mock)).eval()

            mapped_children_sum = np.reshape(children_sum, input_template.shape[:2])

            # a corner child is projected to only one parent
            self.assertTrue(np.all(mapped_children_sum[:, 0] == np.sum(mapped_kernel_parent_mock[:, 0, :], axis=1)))
            # with a stride of 2 the next value is projected to two parents
            self.assertTrue(np.all(mapped_children_sum[:, 1] == np.sum(mapped_kernel_parent_mock[:, 1, :], axis=1)))
            self.assertTrue(np.all(mapped_children_sum[:, -1] == np.sum(mapped_kernel_parent_mock[:, -1, :], axis=1)))

            projected_children_sums = self.topology._project_child_scalars_to_parent_kernels(children_sum, tf.shape(tf.constant(linearized_kernel_parent_mock))).eval()

            projected_children_sums_map = np.reshape(projected_children_sums, [batch_size, input_count, output_count])

            self.assertTrue(np.all(projected_children_sums_map[0, 0, :] == mapped_children_sum[0, 0]))

            self.assertTrue(np.all(projected_children_sums_map[1, 0, :] == mapped_children_sum[1, 0]))
            self.assertTrue(np.all(projected_children_sums_map[1, 1, :] == mapped_children_sum[1, 1]))


            self.assertTrue(np.all(projected_children_sums_map[0, -1, :] == mapped_children_sum[0, -1]))

    def test_aggregation_with_dense_weight_construction(self):
        [child_row_count, child_column_count, child_feature_count] = [14, 14, 5]

        kernel_size = 14

        [batch_size, kernel_row_count, kernel_column_count, kernel_feature_count, parent_row_count, parent_column_count, parent_feature_count, pose_row_count, pose_column_count] =\
        [3,          kernel_size,      kernel_size,         child_feature_count,                 1,                   1,                   3,                    4,              4]

        # setup tested topology
        self.topology.add_aggregation(kernel_size, [[3, 0], [3, 1]])
        self.topology.add_dense_connection(child_feature_count, parent_feature_count)

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": True
        })

        # assert internal state is correct
        self.assertTrue(self.topology.weight_shape == [1, 1, kernel_feature_count, 1, 1, parent_feature_count], "weights must have the correct shape")
        self.assertTrue(self.topology.weight_tiling == [kernel_size, kernel_size, 1, 1, 1, 1], "weight tiling must be correct")

        # test if weights are constructed correctly, and mapped to kernels correctly
        tiled_weights = self.topology.map_weights_to_parent_kernels(batch_size, pose_row_count, pose_column_count)

        tiled_weight_shape = [batch_size, kernel_row_count, kernel_column_count, kernel_feature_count, 1, 1, parent_feature_count, pose_row_count, pose_column_count]

        self.assertTrue(tiled_weights.get_shape() == tiled_weight_shape, "weight matrix must have the correct shape")

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # test if weights are initialized and constructed correctly
            w = tiled_weights.eval()
            self.assertTrue(np.all(np.array(w.shape) == np.array(tiled_weight_shape)))
            data_one_batch = w[0, :, :, :, :, :, :, :, :]
            data_other_batch = w[1, :, :, :, :, :, :, :]

            self.assertTrue(np.all(data_one_batch == data_other_batch), "weights must be tiled across batches")

            data_one_parent_on_row = w[:, 0, 0, :, :, :, :, :, :]
            data_other_parent_on_row = w[:, 1, 0, :, :, :, :, :, :]

            self.assertTrue(np.all(data_one_parent_on_row == data_other_parent_on_row), "weights must be tiled across kernel row")

            data_one_parent_on_column = w[:, 1, 0, :, :, :, :, :, :]
            data_other_parent_on_column = w[:, 1, 1, :, :, :, :, :, :]

            self.assertTrue(np.all(data_one_parent_on_column == data_other_parent_on_column),
                            "weights must be tiled across kernel column")

            data_one_parent_feature = w[:, :, :, 0, :, :, :, :, :]
            data_other_parent_feature = w[:, :, :, 1, :, :, :, :, :]

            self.assertTrue(np.sum(data_one_parent_feature == data_other_parent_feature) == 0,
                            "weights must not be tiled across features")

    def test_aggregation_with_dense_input_mapping_internal_state(self):
        [child_row_count, child_column_count, child_feature_count] = [14, 14, 5]

        kernel_size = 14

        [batch_size, kernel_row_count, kernel_column_count, kernel_feature_count, parent_row_count, parent_column_count, parent_feature_count, pose_row_count, pose_column_count] =\
        [3,          kernel_size,      kernel_size,         child_feature_count,                 1,                   1,                   3,                    4,              4]

        # setup tested topology
        self.topology.add_aggregation(kernel_size, [[3, 0], [3, 1]])
        self.topology.add_dense_connection(child_feature_count, parent_feature_count)

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": True
        })

        feature_connection_count = child_feature_count * parent_feature_count

        # assert that each child has the correct number of parents; aggregations have only 1 parent by definition unless another convolution increases the number of parents
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[0, 1]]), axis=1)) == feature_connection_count)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[1, 0]]), axis=1)) == feature_connection_count)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[1, 1]]), axis=1)) == feature_connection_count)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[2, 0]]), axis=1)) == feature_connection_count)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[2, 2]]), axis=1)) == feature_connection_count)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[2, 3]]), axis=1)) == feature_connection_count)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[3, 3]]), axis=1)) == feature_connection_count)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[5, 5]]), axis=1)) == feature_connection_count)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[4, 4]]), axis=1)) == feature_connection_count)
        self.assertTrue(np.sum(np.all(self.topology.child_parent_kernel_mapping[:, :2] == np.array([[5, 4]]), axis=1)) == feature_connection_count)

    def test_aggregation_with_dense_input_mapping_pose_to_kernel_projection(self):
        [child_row_count, child_column_count, child_feature_count] = [14, 14, 5]

        kernel_size = 14

        [batch_size, kernel_row_count, kernel_column_count, kernel_feature_count, parent_row_count, parent_column_count, parent_feature_count, pose_row_count, pose_column_count] =\
        [3,          kernel_size,      kernel_size,         child_feature_count,                 1,                   1,                   3,                    4,              4]

        # setup tested topology
        self.topology.add_aggregation(kernel_size, [[3, 0], [3, 1]])
        self.topology.add_dense_connection(child_feature_count, parent_feature_count)

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": True
        })

        feature_connection_count = child_feature_count * parent_feature_count

        # test if input is mapped to kernel parent space correctly
        input_template = np.zeros([batch_size, child_row_count, child_column_count, child_feature_count, pose_row_count, pose_column_count])

        row_values = np.arange(child_row_count).reshape([1, child_row_count, 1, 1, 1, 1])
        column_values = np.arange(child_column_count).reshape([1, 1, child_column_count, 1, 1, 1]) * 100
        feature_values = np.arange(child_feature_count).reshape([1, 1, 1, child_feature_count, 1, 1]) * 100 * 100

        batch_values = np.arange(batch_size).reshape([batch_size, 1, 1, 1, 1, 1]) * 100 * 100 * 100

        input_layer_traceable_values = input_template + row_values + column_values + feature_values + batch_values
        # [batch, row, column, m, m]
        input_layer = tf.constant(input_layer_traceable_values.astype(np.float32))

        mapped_input = self.topology.map_input_layer_to_parent_kernels(input_layer)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # test if input is mapped to parent kernel space correctly
            # setup variables
            full_input_map = mapped_input.eval()
            i = full_input_map[:, :, :, :, :, :, :, 0, 0]
            correct_first_kernel_values = input_layer_traceable_values[0, :, :, 0, 0, 0]
            actual_first_kernel_values = i[0, :, :, 0, 0, 0, 0]

            self.assertTrue(np.all(actual_first_kernel_values == correct_first_kernel_values),
                            "first kernel must be mapped accurately")

            correct_next_batch_kernel_values = input_layer_traceable_values[1, :, :, 0, 0, 0]
            actual_next_batch_kernel_values = i[1, :, :, 0, 0, 0, 0]

            self.assertTrue(np.all(actual_next_batch_kernel_values == correct_next_batch_kernel_values),
                            "next batch kernel must be mapped accurately")

            correct_feature_kernel_values = np.tile(np.reshape(input_layer_traceable_values[0, 0, 0, :, 0, 0], [child_feature_count, 1]), [1, parent_feature_count])
            actual_feature_kernel_values = i[0, 0, 0, :, 0, 0, :]

            self.assertTrue(np.all(actual_feature_kernel_values == correct_feature_kernel_values),
                            "features must be mapped accurately")

            correct_feature_kernel_values = np.tile(np.reshape(input_layer_traceable_values[0, 1, 0, :, 0, 0], [child_feature_count, 1]), [1, parent_feature_count])
            actual_feature_kernel_values = i[0, 1, 0, :, 0, 0, :]

            self.assertTrue(np.all(actual_feature_kernel_values == correct_feature_kernel_values),
                            "features must be mapped accurately")


    def test_aggregation_with_dense_child_summing_logic(self):
        [child_row_count, child_column_count, child_feature_count] = [14, 14, 5]

        kernel_size = 14

        [batch_size, kernel_row_count, kernel_column_count, kernel_feature_count, parent_row_count, parent_column_count, parent_feature_count, pose_row_count, pose_column_count] =\
        [3,          kernel_size,      kernel_size,         child_feature_count,                 1,                   1,                   3,                    4,              4]

        # setup tested topology
        self.topology.add_aggregation(kernel_size, [[3, 0], [3, 1]])
        self.topology.add_dense_connection(child_feature_count, parent_feature_count)

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": True
        })

        feature_connection_count = child_feature_count * parent_feature_count

        # test if input is mapped to kernel parent space correctly
        input_template = np.zeros([batch_size, child_row_count, child_column_count, child_feature_count, pose_row_count, pose_column_count])

        row_values = np.arange(child_row_count).reshape([1, child_row_count, 1, 1, 1, 1])
        column_values = np.arange(child_column_count).reshape([1, 1, child_column_count, 1, 1, 1]) * 100
        feature_values = np.arange(child_feature_count).reshape([1, 1, 1, child_feature_count, 1, 1]) * 100 * 100

        batch_values = np.arange(batch_size).reshape([batch_size, 1, 1, 1, 1, 1]) * 100 * 100 * 100

        input_layer_traceable_values = input_template + row_values + column_values + feature_values + batch_values
        # [batch, row, column, m, m]
        input_layer = tf.constant(input_layer_traceable_values.astype(np.float32))

        mapped_input = self.topology.map_input_layer_to_parent_kernels(input_layer)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # test if input is mapped to parent kernel space correctly
            # setup variables
            full_input_map = mapped_input.eval()

            m = self.topology.make_sparse_linearized_child_to_parent_kernel(self.topology.child_parent_kernel_mapping)

            dense_linearized_child_to_parent_kernel = tf.sparse_to_dense(m.indices, m.dense_shape, m.values).eval()

            self.assertTrue(np.sum(dense_linearized_child_to_parent_kernel) == self.topology.child_parent_kernel_mapping.shape[0])

            # mock assuming weights are all identity matrices
            linearized_kernel_parent_mock = self.topology._linearize_potential_parent_poses_map(tf.constant(full_input_map)).eval()[:, :, :, 0, 0]
            mapped_kernel_parent_mock = full_input_map[:, :, :, :, :, :, :, 0, 0]

            children_sum = self.topology._compute_sum_for_children(tf.constant(linearized_kernel_parent_mock)).eval()

            mapped_children_sum = np.reshape(children_sum, input_template.shape[:4])

            self.assertTrue(np.all(mapped_children_sum[:, 0, 0, 0] == np.sum(mapped_kernel_parent_mock[:, 0, 0, 0, 0, 0, :], axis=1)))

            self.assertTrue(np.all(mapped_children_sum[:, 1, 0, 1] == np.sum(mapped_kernel_parent_mock[:, 1, 0, 1, 0, 0, :], axis=1)))


            projected_children_sums = self.topology._project_child_scalars_to_parent_kernels(children_sum, tf.shape(tf.constant(linearized_kernel_parent_mock))).eval()

            projected_children_sums_map = np.reshape(projected_children_sums, [batch_size, kernel_row_count, kernel_column_count, kernel_feature_count, parent_row_count, parent_column_count, parent_feature_count])

            self.assertTrue(np.all(projected_children_sums_map[:, 0, 0, 0, 0, 0, 0] == mapped_children_sum[:, 0, 0, 0]))

            self.assertTrue(np.all(projected_children_sums_map[:, 1, 0, 0, 0, 0, 0] == mapped_children_sum[:, 1, 0, 0]))
            self.assertTrue(np.all(projected_children_sums_map[:, 0, 1, 0, 0, 0, 0] == mapped_children_sum[:, 0, 1, 0]))
            self.assertTrue(np.all(projected_children_sums_map[:, 0, 0, 1, 0, 0, 0] == mapped_children_sum[:, 0, 0, 1]))

            self.assertTrue(np.all(projected_children_sums_map[:, 1, 0, 0, 0, 0, 1] == mapped_children_sum[:, 1, 0, 0]))
            self.assertTrue(np.all(projected_children_sums_map[:, 0, 1, 0, 0, 0, 1] == mapped_children_sum[:, 0, 1, 0]))
            self.assertTrue(np.all(projected_children_sums_map[:, 0, 0, 1, 0, 0, 1] == mapped_children_sum[:, 0, 0, 1]))


    def test_spatial_and_semantic_convolution_child_summing_logic(self):
        [child_row_count, child_column_count, child_feature_row_count, child_feature_column_count] = [12, 12, 6, 6]

        spatial_kernel_size = 3
        semantic_kernel_size = 3
        stride = 2

        [batch_size, kernel_row_count, kernel_column_count, kernel_feature_row_count, kernel_feature_column_count, parent_row_count, parent_column_count, parent_feature_row_count, parent_feature_column_count, pose_row_count, pose_column_count] =\
        [3,          spatial_kernel_size,      spatial_kernel_size, semantic_kernel_size, semantic_kernel_size,                   5,                   5,                        3,                           3,                4,              4]

        # setup tested topology
        self.topology.add_spatial_convolution([child_row_count, child_column_count], spatial_kernel_size, stride)
        self.topology.add_semantic_convolution([child_feature_row_count, child_feature_column_count], semantic_kernel_size, stride)

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": True
        })

        # test if input is mapped to kernel parent space correctly
        input_template = np.zeros([batch_size, child_row_count, child_column_count, child_feature_row_count, child_feature_column_count, pose_row_count, pose_column_count])

        row_values = np.arange(child_row_count).reshape([1, child_row_count, 1, 1, 1, 1, 1])
        column_values = np.arange(child_column_count).reshape([1, 1, child_column_count, 1, 1, 1, 1]) * 100

        feature_row_values = np.arange(child_feature_row_count).reshape([1, 1, 1, child_feature_row_count, 1, 1, 1]) * 100 * 100
        feature_column_values = np.arange(child_feature_column_count).reshape([1, 1, 1, 1, child_feature_column_count, 1, 1]) * 100 * 100 * 100

        batch_values = np.arange(batch_size).reshape([batch_size, 1, 1, 1, 1, 1, 1]) * 100 * 100 * 100 * 100

        input_layer_traceable_values = input_template + row_values + column_values + feature_row_values + feature_column_values + batch_values
        # [batch, row, column, m, m]
        input_layer = tf.constant(input_layer_traceable_values.astype(np.float32))

        mapped_input = self.topology.map_input_layer_to_parent_kernels(input_layer)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # test if input is mapped to parent kernel space correctly
            # setup variables
            full_input_map = mapped_input.eval()

            m = self.topology.make_sparse_linearized_child_to_parent_kernel(self.topology.child_parent_kernel_mapping)

            dense_linearized_child_to_parent_kernel = tf.sparse_to_dense(m.indices, m.dense_shape, m.values).eval()

            self.assertTrue(np.sum(dense_linearized_child_to_parent_kernel) == self.topology.child_parent_kernel_mapping.shape[0])


            # mock assuming weights are all identity matrices
            linearized_kernel_parent_mock = self.topology._linearize_potential_parent_poses_map(tf.constant(full_input_map)).eval()[:, :, :, 0, 0]
            mapped_kernel_parent_mock = full_input_map[:, :, :, :, :, :, :, :, :, 0, 0]

            children_sum = self.topology._compute_sum_for_children(tf.constant(linearized_kernel_parent_mock)).eval()

            mapped_children_sum = np.reshape(children_sum, input_template.shape[:5])

            self.assertTrue(np.all(mapped_children_sum[:, 0, 0, 0, 0] ==
                                   mapped_kernel_parent_mock[:, 0, 0, 2, 2, 0, 0, 2, 2] +
                                   mapped_kernel_parent_mock[:, 0, 0, 2, 0, 0, 0, 2, 0] +
                                   mapped_kernel_parent_mock[:, 0, 0, 0, 2, 0, 0, 0, 2] +
                                   mapped_kernel_parent_mock[:, 0, 0, 0, 0, 0, 0, 0, 0]))


            self.assertTrue(np.all(mapped_children_sum[:, 2, 2, 0, 0] ==
                                   mapped_kernel_parent_mock[:, 2, 0, 2, 2, 0, 1, 2, 2] +
                                   mapped_kernel_parent_mock[:, 2, 0, 2, 0, 0, 1, 2, 0] +
                                   mapped_kernel_parent_mock[:, 2, 0, 0, 2, 0, 1, 0, 2] +
                                   mapped_kernel_parent_mock[:, 2, 0, 0, 0, 0, 1, 0, 0] +

                                   mapped_kernel_parent_mock[:, 0, 2, 2, 2, 1, 0, 2, 2] +
                                   mapped_kernel_parent_mock[:, 0, 2, 2, 0, 1, 0, 2, 0] +
                                   mapped_kernel_parent_mock[:, 0, 2, 0, 2, 1, 0, 0, 2] +
                                   mapped_kernel_parent_mock[:, 0, 2, 0, 0, 1, 0, 0, 0] +

                                   mapped_kernel_parent_mock[:, 2, 2, 2, 2, 0, 0, 2, 2] +
                                   mapped_kernel_parent_mock[:, 2, 2, 2, 0, 0, 0, 2, 0] +
                                   mapped_kernel_parent_mock[:, 2, 2, 0, 2, 0, 0, 0, 2] +
                                   mapped_kernel_parent_mock[:, 2, 2, 0, 0, 0, 0, 0, 0] +

                                   mapped_kernel_parent_mock[:, 0, 0, 2, 2, 1, 1, 2, 2] +
                                   mapped_kernel_parent_mock[:, 0, 0, 2, 0, 1, 1, 2, 0] +
                                   mapped_kernel_parent_mock[:, 0, 0, 0, 2, 1, 1, 0, 2] +
                                   mapped_kernel_parent_mock[:, 0, 0, 0, 0, 1, 1, 0, 0]))

            projected_children_sums = self.topology._project_child_scalars_to_parent_kernels(children_sum, tf.shape(tf.constant(linearized_kernel_parent_mock))).eval()

            projected_children_sums_map = np.reshape(projected_children_sums, [batch_size,
                                                                               kernel_row_count,
                                                                               kernel_column_count,
                                                                               kernel_feature_row_count,
                                                                               kernel_feature_column_count,
                                                                               parent_row_count,
                                                                               parent_column_count,
                                                                               parent_feature_row_count,
                                                                               parent_feature_column_count])

            self.assertTrue(np.all(projected_children_sums_map[:, 0, 0, 0, 0, 0, 0, 0, 0] == mapped_children_sum[:, 0, 0, 0, 0]))
            self.assertTrue(np.all(projected_children_sums_map[:, 0, 0, 2, 0, 0, 0, 2, 0] == mapped_children_sum[:, 0, 0, 0, 0]))
            self.assertTrue(np.all(projected_children_sums_map[:, 0, 0, 0, 2, 0, 0, 0, 2] == mapped_children_sum[:, 0, 0, 0, 0]))
            self.assertTrue(np.all(projected_children_sums_map[:, 0, 0, 2, 2, 0, 0, 2, 2] == mapped_children_sum[:, 0, 0, 0, 0]))

            self.assertTrue(np.all(projected_children_sums_map[:, 0, 2, 0, 2, 1, 0, 0, 2] == mapped_children_sum[:, 2, 2, 0, 0]))
            self.assertTrue(np.all(projected_children_sums_map[:, 0, 2, 0, 2, 1, 0, 0, 2] == mapped_children_sum[:, 2, 2, 0, 0]))
            self.assertTrue(np.all(projected_children_sums_map[:, 0, 0, 2, 2, 1, 1, 2, 2] == mapped_children_sum[:, 2, 2, 0, 0]))
            self.assertTrue(np.all(projected_children_sums_map[:, 0, 0, 0, 0, 1, 1, 0, 0] == mapped_children_sum[:, 2, 2, 0, 0]))

    def test_add_orthogonal_unit_axis_loss_with_rotation_and_translation(self):

        a = .33 * np.pi

        m = np.identity(4, dtype=np.float32)

        m[0, 0] = np.cos(a)
        m[0, 1] = np.sin(a)

        m[1, 0] = -np.sin(a)
        m[1, 1] = np.cos(a)

        m[3, :] = [1, 2, 3, 1]

        tf_matrix = tf.reshape(tf.constant(m), [-1, 4, 4])

        double_row_matrix = tf.concat([tf_matrix, tf_matrix], axis=0)

        loss = TopologyBuilder.add_orthogonal_unit_axis_loss(double_row_matrix)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            l = sess.run(loss)

            self.assertTrue(l == 0, "loss for a orthogonal matrix must be 0")

    def test_add_orthogonal_unit_axis_loss_with_scaling(self):

        a = .33 * np.pi

        m = np.identity(4, dtype=np.float32)

        m[0, 0] = np.cos(a) * 3.0
        m[0, 1] = np.sin(a)

        m[1, 0] = -np.sin(a)
        m[1, 1] = np.cos(a)

        m[3, :] = [1, 2, 3, 1]

        tf_matrix = tf.reshape(tf.constant(m), [-1, 4, 4])

        double_row_matrix = tf.concat([tf_matrix, tf_matrix], axis=0)

        loss = TopologyBuilder.add_orthogonal_unit_axis_loss(double_row_matrix)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            l = sess.run(loss)

            self.assertTrue(l > 0.0, "loss for a non orthogonal matrix (due to scaling) must be greater than 0")

    def test_add_orthogonal_unit_axis_without_orthogonality(self):

        a = .33 * np.pi

        m = np.identity(4, dtype=np.float32)

        m[0, 0] = np.cos(a)
        m[0, 1] = np.sin(a)

        m[3, :] = [1, 2, 3, 1]

        tf_matrix = tf.reshape(tf.constant(m), [-1, 4, 4])

        double_row_matrix = tf.concat([tf_matrix, tf_matrix], axis=0)

        loss = TopologyBuilder.add_orthogonal_unit_axis_loss(double_row_matrix)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            l = sess.run(loss)

            self.assertTrue(l > 0.0, "loss for a non orthogonal matrix (due to non orthogonal axes) must be greater than 0")

    def test_axial_system_weights(self):

        input_count, output_count = 2, 3
        batch_size = 2

        pose_row_count = 4
        pose_column_count = 4

        tiled_weight_shape = [batch_size, input_count, output_count, pose_row_count, pose_column_count]

        # setup tested topology
        self.topology.set_is_axial_system(True)
        self.topology.add_dense_connection(input_count, output_count)

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": True
        })

        axial_system_input = np.random.random([batch_size, input_count, pose_row_count, pose_column_count])
        axial_system_input[:, :, :, 3] = np.array([0.0, 0.0, 0.0, 1.0])

        output = self.topology._compute_potential_parent_poses_map(tf.constant(axial_system_input, dtype=tf.float32))

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            output_values = sess.run(output)

            self.assertTrue(np.all(output_values[:, :, :, :, 3] == np.array([0.0, 0.0, 0.0, 1.0])), "output must remain axial system" )

    def test_build_xavier_kernel_init(self):
        input_count, output_count = 2, 3
        batch_size = 200

        pose_row_count = 4
        pose_column_count = 4

        tiled_weight_shape = [batch_size, input_count, output_count, pose_row_count, pose_column_count]

        # setup tested topology
        self.topology.add_dense_connection(input_count, output_count)

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": False
        })

        output = self.topology.map_weights_to_parent_kernels(batch_size, 4, 4)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            output_values = sess.run(output)

            output_value_list = output_values.reshape(-1)

            self.assertTrue(np.abs(np.mean(output_value_list)) < .1)
            self.assertTrue(np.abs(np.std(output_value_list)  - np.sqrt(1 / input_count)) < .1)
            self.assertTrue(list(output_values.shape) == tiled_weight_shape)

    def test_build_xavier_matrix_init(self):
        input_count, output_count = 2, 3
        batch_size = 200

        pose_row_count = 4
        pose_column_count = 4

        tiled_weight_shape = [batch_size, input_count, output_count, pose_row_count, pose_column_count]

        # setup tested topology
        self.topology.add_dense_connection(input_count, output_count)

        self.topology.finish({
            "type": "xavier",
            "kernel": False,
            "matrix": True
        })

        output = self.topology.map_weights_to_parent_kernels(batch_size, 4, 4)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            output_values = sess.run(output)

            output_value_list = output_values.reshape(-1)

            self.assertTrue(np.abs(np.mean(output_value_list)) < .1)
            self.assertTrue(np.abs(np.std(output_value_list)  - .5) < .1)
            self.assertTrue(list(output_values.shape) == tiled_weight_shape)

    def test_build_xavier_matrix_kernel_init(self):
        input_count, output_count = 2, 3
        batch_size = 200

        pose_row_count = 4
        pose_column_count = 4

        tiled_weight_shape = [batch_size, input_count, output_count, pose_row_count, pose_column_count]

        # setup tested topology
        self.topology.add_dense_connection(input_count, output_count)

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": True
        })

        output = self.topology.map_weights_to_parent_kernels(batch_size, 4, 4)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            output_values = sess.run(output)

            output_value_list = output_values.reshape(-1)

            self.assertTrue(np.abs(np.mean(output_value_list)) < .1)
            self.assertTrue(np.abs(np.std(output_value_list) - np.sqrt(1 / (input_count * 4))) < .1)
            self.assertTrue(list(output_values.shape) == tiled_weight_shape)

    def test_build_normal_init(self):
        input_count, output_count = 2, 3
        batch_size = 20

        pose_row_count = 4
        pose_column_count = 4

        tiled_weight_shape = [batch_size, input_count, output_count, pose_row_count, pose_column_count]

        # setup tested topology
        self.topology.add_dense_connection(input_count, output_count)

        normal_options = {
            "type": "normal",
            "deviation": [.4, .1]
        }

        self.topology.finish(normal_options)

        self.assertTrue(normal_options["deviation"] == [.1])

        output = self.topology.map_weights_to_parent_kernels(batch_size, 4, 4)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            output_values = sess.run(output)

            output_value_list = output_values.reshape(-1)

            self.assertTrue(np.abs(np.mean(output_value_list)) < .1)
            self.assertTrue(np.abs(np.std(output_value_list)  -.4) < .1)
            self.assertTrue(list(output_values.shape) == tiled_weight_shape)

    def test_build_identity_init(self):
        input_count, output_count = 2, 3
        batch_size = 10

        pose_row_count = 4
        pose_column_count = 4

        tiled_weight_shape = [batch_size, input_count, output_count, pose_row_count, pose_column_count]

        # setup tested topology
        self.topology.add_dense_connection(input_count, output_count)

        self.topology.finish({
            "type": "identity",
            "uniform": .5
        })

        output = self.topology.map_weights_to_parent_kernels(batch_size, 4, 4)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            output_values = sess.run(output)

            output_matrix_list = output_values.reshape(-1, 4, 4)

            self.assertTrue(np.all(output_matrix_list[:, np.eye(4).astype('bool')] == 1))

            random_values = output_matrix_list[:, ~np.eye(4).astype('bool')]

            self.assertTrue(np.abs(np.mean(random_values)) < .1 )
            self.assertTrue(np.abs(np.mean(np.abs(random_values)) - 0.25) < .1)
            self.assertTrue(list(output_values.shape) == tiled_weight_shape)

    def test_build_wrong_init(self):
        input_count, output_count = 2, 3
        batch_size = 10

        pose_row_count = 4
        pose_column_count = 4

        tiled_weight_shape = [batch_size, input_count, output_count, pose_row_count, pose_column_count]

        # setup tested topology
        self.topology.add_dense_connection(input_count, output_count)

        try:
            self.topology.finish({
                "type": "jhfgjhgf",
                "uniform": .5
            })
            self.assertTrue(False)
        except UnknownInitializerException as e:
            self.assertTrue(True)

    def test_replace_kernel_elements_with_max_of_child(self):
        # name numbers for legibility
        [child_row_count, child_column_count] = [5, 5]

        kernel_size = 3
        stride = 2

        [batch_size, kernel_row_count, kernel_column_count, parent_row_count, parent_column_count, pose_row_count, pose_column_count] =\
        [2,          kernel_size,      kernel_size,         2,                2,                   4,              4]

        tiled_weight_shape = [batch_size, kernel_row_count, kernel_column_count, parent_row_count, parent_column_count, pose_row_count, pose_column_count]

        # setup tested topology
        self.topology.add_spatial_convolution([child_row_count, child_column_count], kernel_size, stride)

        self.topology.finish({
            "type": "xavier",
            "kernel": True,
            "matrix": True
        })

        input_full_shape = [batch_size, kernel_row_count, kernel_column_count, parent_row_count, parent_column_count, 1]
        input_full = np.zeros(input_full_shape)

        # BATCH 0
        input_full[0, 0, 0, 0, 0] = -1  # child at 0, 0
        input_full[0, 1, 0, 0, 0] = -2  # child at 1, 0
        input_full[0, 2, 0, 0, 0] = -3  # child at 2, 0
        input_full[0, 0, 0, 1, 0] = -2  # child at 2, 0

        # greatest value of child at 2, 0 === -2

        input_full[0, 2, 2, 0, 0] = -4  # child at 2, 2
        input_full[0, 0, 2, 1, 0] = -5  # child at 2, 2
        input_full[0, 2, 0, 0, 1] = -6  # child at 2, 2
        input_full[0, 0, 0, 1, 1] = -7  # child at 2, 2

        # greatest value of child at 2, 2 === -4

        # BATCH 1
        input_full[1, 2, 2, 0, 0] = -2  # child at 2, 2
        input_full[1, 0, 2, 1, 0] = -3  # child at 2, 2
        input_full[1, 2, 0, 0, 1] = -4  # child at 2, 2
        input_full[1, 0, 0, 1, 1] = -1  # child at 2, 2

        # greatest value of child at 2, 2 === -2

        input_collapsed_shape = [batch_size, kernel_row_count * kernel_column_count, parent_row_count * parent_column_count, 1]
        input_collapsed = np.reshape(input_full, input_collapsed_shape)

        with self.test_session() as sess:
            input_layer = tf.constant(input_collapsed.astype(np.float32))
            child_maxima = self.topology.replace_kernel_elements_with_max_of_child(input_layer)

            sess.run(tf.global_variables_initializer())

            result = tf.reshape(child_maxima, [2, kernel_row_count, kernel_column_count, parent_row_count, parent_column_count]).eval()

            # BATCH 0

            self.assertTrue(result[0, 0, 0, 0, 0] == -1)  # child at 0, 0
            self.assertTrue(result[0, 1, 0, 0, 0] == -2)  # child at 1, 0
            self.assertTrue(result[0, 2, 0, 0, 0] == -2)  # child at 2, 0
            self.assertTrue(result[0, 0, 0, 1, 0] == -2 ) # child at 2, 0

            self.assertTrue(result[0, 2, 2, 0, 0] == -4)  # child at 2, 2
            self.assertTrue(result[0, 0, 2, 1, 0] == -4)  # child at 2, 2
            self.assertTrue(result[0, 2, 0, 0, 1] == -4)  # child at 2, 2
            self.assertTrue(result[0, 0, 0, 1, 1] == -4)  # child at 2, 2

            # BATCH 1
            # child at 2, 2
            self.assertTrue(result[1, 2, 2, 0, 0] == -1)
            self.assertTrue(result[1, 0, 2, 1, 0] == -1)
            self.assertTrue(result[1, 2, 0, 0, 1] == -1)
            self.assertTrue(result[1, 0, 0, 1, 1] == -1)


