import numpy as np
import tensorflow as tf
import os
import os.path
from input.MatFileReaderWriter import MatFileReaderWriter


class TestMatFileReader(tf.test.TestCase):

    def setUp(self):
        self.matrix_file_path = os.path.dirname(os.path.abspath(__file__)) + '/trash/test_matrix.mat'
        self.tearDown()  # invoke tear down to assure clean starting state in case previous test crashed

    def tearDown(self):
        if os.path.isfile(self.matrix_file_path):
            os.remove(self.matrix_file_path)

    def _test_read_write(self, shape, dtype):
        random_matrix = (np.random.random(shape) * 100).astype(dtype)

        MatFileReaderWriter.write_mat_file(self.matrix_file_path, random_matrix)
        read_matrix = MatFileReaderWriter.read_mat_file(self.matrix_file_path)

        self.assertAllEqual(random_matrix, read_matrix, dtype + ' matrix of shape ' + str(shape) + ' must be read correctly')
        self.assertAllEqual(random_matrix.shape, read_matrix.shape, 'read ' + dtype + ' matrix of shape ' + str(shape) + ' must have correct shape')

    def test_read_write_formats(self):
        self._test_read_write([1], 'uint8')
        self._test_read_write([3,2,3], 'uint8')
        self._test_read_write([1], 'int32')
        self._test_read_write([3,2,3], 'int32')

