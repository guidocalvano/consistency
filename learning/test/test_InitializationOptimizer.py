import unittest
from learning.InitializationOptimizer import InitializationOptimizer
from input.SmallNorb import SmallNorb
from learning.MatrixCapsNet import MatrixCapsNet
import tensorflow as tf
import numpy as np


class test_InitializationOptimizer(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_InitializationOptimizer(self):
        # @TODO iteration count should be added to config, but currently run.py also doesn't get it from config
        # so both should be changed at the same time
        iteration_count = 3

        sn = SmallNorb.from_cache()
        sn.randomly_reduce_training_set(6)

        constant_list = InitializationOptimizer.optimize_standard_deviations('build_default_architecture', sn, iteration_count, 6)

        tf.reset_default_graph()
        examples = sn.default_training_set()[0]
        mcn = MatrixCapsNet(tf.float32)
        mcn.set_init_options({
            "type": "",
            "deviation": constant_list
        })
        network_output, reset_routing_configuration_op, regularization_loss = \
            getattr(mcn, 'build_default_architecture')(examples, iteration_count, None)

        activation_layers = mcn.activation_layers

        with tf.Session() as sess:
            activation_output = sess.run(activation_layers)

            for i in range(len(activation_layers)):
                current_output = activation_output[i]

                self.assertTrue(np.abs((np.sum(current_output > .5) / np.product(current_output.shape)) - .5 ) < .1)