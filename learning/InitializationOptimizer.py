import tensorflow as tf
from learning.MatrixCapsNet import MatrixCapsNet
from learning.InfiniteVariableList import InfiniteVariableList
import numpy as np

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

import numpy as np
from scipy.stats import norm as normal_distribution
from scipy.optimize import minimize


class InitializationOptimizer:

    @staticmethod
    def initial_neural_net_output(architecture, sn, iteration_count, batch_size, constants):
        tf.reset_default_graph()
        variable_list = InfiniteVariableList(constants)
        input_layer = sn.default_training_set()[0]

        reduced_input_layer = tf.random_shuffle(input_layer)[:batch_size]

        init_options = {
            "type": "normal",
            "deviation": variable_list
        }
        mcn = MatrixCapsNet()
        mcn.set_init_options(init_options)
        network_output, reset_routing_configuration_op, regularization_loss = \
            getattr(mcn, architecture)(reduced_input_layer, iteration_count, None)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            activation_values = sess.run(mcn.activation_layers)

            return activation_values
    @staticmethod
    def minimize_quadratic_function(a, b):
        minimal_x = -b / (2 * a)

        return minimal_x

    @staticmethod
    def taylor_approximate(d_constant, error_zero, error_one, error_two):
        # taylor series expansion
        d_error_d_constant = (error_two - error_zero) / (2 * d_constant)

        d_error_d_constant_first = (error_one - error_zero) / d_constant
        d_error_d_constant_second = (error_two - error_one) / d_constant

        dd_error_d_constant = (d_error_d_constant_second - d_error_d_constant_first / d_constant)

        a = dd_error_d_constant / 2
        b = d_error_d_constant / (2 * a)
        #c = error_one - a * x * x - b * x

        return a, b

    @staticmethod
    def _compute_error(constants, target_layer, c, architecture, sn, iteration_count, batch_size):
        constants[target_layer] = c
        activations = InitializationOptimizer.initial_neural_net_output(architecture, sn, iteration_count, batch_size,
                                                                        constants)

        target_activation = activations[target_layer]

        # the percentage of active neurons must be 50%
        error = (np.sum(target_activation > 0.5) / np.product(target_activation.shape)) - .5

        return error

    @staticmethod
    def taylor_optimize_standard_deviations(architecture, sn, iteration_count, batch_size):

        constants = []
        activations = None
        target_layer = 0
        while activations is None or len(activations) == len(constants):
            next_constant = 1.0
            d_constant = .01
            last_error = None

            constants.append(next_constant)
            while True:

                error_zero = InitializationOptimizer._compute_error(
                    constants,
                    target_layer,
                    next_constant - d_constant,
                    architecture,
                    sn,
                    iteration_count,
                    batch_size)

                error_one = InitializationOptimizer._compute_error(
                    constants,
                    target_layer,
                    next_constant,
                    architecture,
                    sn,
                    iteration_count,
                    batch_size)


                if np.abs(error_one) < .01:
                    break

                if last_error is not None and np.abs(error_one) >= np.abs(last_error):
                    d_constant = d_constant / 2

                error_two = InitializationOptimizer._compute_error(
                    constants,
                    target_layer,
                    next_constant + d_constant,
                    architecture,
                    sn,
                    iteration_count,
                    batch_size)

                # we want to be on a slope, not around the minimum
                if np.sign(error_two - error_one) != np.sign(error_one - error_zero):
                    d_constant = d_constant / 2
                    continue

                a, b = InitializationOptimizer.taylor_approximate(d_constant, error_zero, error_one, error_two)

                next_constant = InitializationOptimizer.minimize_quadratic_function(a, b)

                last_error = error_one

            target_layer += 1
        return constants

    @staticmethod
    def optimize_standard_deviations(architecture, sn, iteration_count, batch_size):
        return InitializationOptimizer.polynomial_optimize_standard_deviations(architecture, sn, iteration_count, batch_size)

    @staticmethod
    def polynomial_optimize_standard_deviations(architecture, sn, iteration_count, batch_size):
        target_layer = 0
        random_count = 3
        bounds = [-2, 3]
        layer_depth = InitializationOptimizer.layer_depth(architecture)
        constants = [1] * layer_depth

        while target_layer < layer_depth:
            def f(constant_guess):
                scaled_constant = np.power(np.e, constant_guess)
                constants[target_layer] = scaled_constant

                return np.abs(InitializationOptimizer._compute_error(
                    constants,
                    target_layer,
                    scaled_constant,
                    architecture,
                    sn,
                    iteration_count,
                    batch_size))

            next_constant = np.power(np.e, InitializationOptimizer.search_best_parameter(f, random_count, bounds, InitializationOptimizer.polynomial_update_metric))
            # this line is necessary in case the best solution was found during the random probe phase
            constants[target_layer] = next_constant

            target_layer += 1

        return constants

    @staticmethod
    def bayesian_optimize_standard_deviations(architecture, sn, iteration_count, batch_size):

        target_layer = 0
        random_count = 3
        bounds = [-2, 3]
        layer_depth = InitializationOptimizer.layer_depth(architecture)
        constants = [1] * layer_depth

        while target_layer < layer_depth:

            def f(constant_guess):
                scaled_constant = np.power(np.e, constant_guess)
                constants[target_layer] = scaled_constant

                return np.abs(InitializationOptimizer._compute_error(
                    constants,
                    target_layer,
                    scaled_constant,
                    architecture,
                    sn,
                    iteration_count,
                    batch_size))

            next_constant = np.power(np.e, InitializationOptimizer.search_best_parameter(f, random_count, bounds, InitializationOptimizer.bayesian_update_metric))
            constants.append(next_constant)

            target_layer += 1

        return constants

    @staticmethod
    def layer_depth(architecture):
        tf.reset_default_graph()
        variable_list = InfiniteVariableList()
        input_layer = tf.constant(np.random.random([1, 32, 32, 1]), dtype=tf.float32)

        init_options = {
            "type": "normal",
            "deviation": variable_list
        }
        mcn = MatrixCapsNet()
        mcn.set_init_options(init_options)
        network_output, reset_routing_configuration_op, regularization_loss = \
            getattr(mcn, architecture)(input_layer, 1, None)

        return variable_list.depth

    @staticmethod
    def search_best_parameter(f, random_count, bounds, update_metric):

        x = []
        y = []

        for i in range(random_count):
            random_x = bounds[0] + np.random.random() * (bounds[1] - bounds[0])
            y_output = f(random_x)
            x.append(random_x)
            y.append(y_output)


        while np.min(y) > .1:
            metric = update_metric(x, y)
            best_guess_parameters = InitializationOptimizer.guess_best_parameters(metric, bounds, 10)[0]

            y_output = f(best_guess_parameters)

            x.append(best_guess_parameters)
            y.append(y_output)

        return x[np.argmin(y)]

    @staticmethod
    def expected_improvement(mean, standard_deviation, best_known, exploration_rate_ei):
        # this will compute the expected increase, not decrease!

        z = (mean - best_known - exploration_rate_ei) / standard_deviation

        return standard_deviation * (z * normal_distribution.cdf(z) + normal_distribution.pdf(z))

    @staticmethod
    def retrain_gaussian_process_model(x, y):
        gaussian_process = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=25
        )

        gaussian_process.fit(np.array(x).reshape(-1, 1), np.array(y))

        return gaussian_process

    @staticmethod
    def guess_best_parameters(metric, x_bounds, guess_count):

        guess_starting_parameters = [x_bounds[0] + np.random.random() * (x_bounds[1] - x_bounds[0]) for i in range(guess_count)]

        best_parameters = None
        best_guess = float("inf")
        for start_parameter in guess_starting_parameters:
            result = minimize(lambda vals: -metric(vals), start_parameter, bounds=[tuple(x_bounds)], method="L-BFGS-B")
            if not result.success:
                continue

            if result.fun[0] <= best_guess:
                best_guess = result.fun[0]
                best_parameters = result.x

        return best_parameters

    @staticmethod
    def bayesian_update_metric(x, y):
        gaussian_process = InitializationOptimizer.retrain_gaussian_process_model(x, y)

        best_known = np.min(y)


        def ei(x):
            mean, standard_deviation = gaussian_process.predict(np.array(x).reshape(-1, 1), return_std=True)

            return InitializationOptimizer.expected_improvement(-mean, standard_deviation, -best_known, 0.0)

        return ei

    @staticmethod
    def polynomial_update_metric(x, y):
        coefficients = np.polyfit(x[-5:], y[-5:], 2)

        def predict(x):
            return -np.array([np.inner(np.power(x, np.array(list(reversed(range(3))))), coefficients)])

        return predict

