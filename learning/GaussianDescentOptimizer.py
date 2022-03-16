import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class GaussianDescentOptimizer(tf.training.optimizer.Optimizer):

    # weight is
    def merge_gaussians(self, a, b):
        # fits gaussian a gaussian c to gaussians a and b
        # This is what I built https://en.wikipedia.org/wiki/Mixture_distribution#Moments
        # It might be incorrect though: https://www.wolframalpha.com/input/?i=integrate+x%5E2*e%5E%28-%28x+-+m%29%5E2%29dx

        c = {}
        c["mean"] = (a["mean"] * a["weight"] + b["mean"] * b["weight"]) / (a["weight"] + b["weight"])
        c["variance"] = \
            a["weight"] * (a["variance"] + a["mean"] * a["mean"] - c["mean"] * c["mean"]) + \
            b["weight"] * (b["variance"] + b["mean"] * b["mean"] - c["mean"] * c["mean"])

        return c

    def gradient_change_probability(self, previous_distribution, new_data):
        # we must chose a belief that the new data fits the previous distribution
        # bayes rule for two alternatives
        # initial prior
        prior_odds = 1

        # do computation in log space to prevent numerical instability
        no_change_probabilities = self.normal_pdf(previous_distribution, new_data)
        change_probabilities = -no_change_probabilities + 1.0

        log_p_no_change = tf.reduce_sum(tf.log(no_change_probabilities))
        log_p_change = tf.reduce_sum(tf.log(change_probabilities))

        log_posterior_odds = log_p_no_change - log_p_change

        posterior_odds = tf.exp(log_posterior_odds)

        no_change_probability = posterior_odds / (1 + posterior_odds)

        change_probability = 1.0 - no_change_probability

        return change_probability

    def net_gradient(self, per_example_gradients):
        example_axis = 0
        return tf.reduce_sum(per_example_gradients, axis=example_axis)

    def abs_gradient(self, per_example_gradients):
        example_axis = 0
        return tf.reduce_sum(tf.abs(per_example_gradients), axis=example_axis)

    def probability_global_minimum(self, net_gradient, abs_gradient):
        return net_gradient / abs_gradient



    def compute_gradients(self, example_losses, variables):
        example_axis = 0  # batch axis is example axis

        # Computing per-example gradients

        def model_fn(example_loss):
            return tf.gradients(example_loss, variables)

        per_example_gradients = tf.vectorized_map(model_fn, example_losses)

        total_gradient_energy = tf.reduce_sum(tf.abs(per_example_gradients))
        kinetic_gradient_energy = tf.reduce_sum(per_example_gradients, axis=example_axis)

        probability_global_minimum = kinetic_gradient_energy / total_gradient_energy

        weighted_current_expected_gradient = (1.0 - probability_global_minimum) * self.current_expected_global_gradient

        next_kinetic_gradient_energy = weighted_current_expected_gradient + kinetic_gradient_energy

        total_momentum_energy = tf.abs(kinetic_gradient_energy) + tf.abs(weighted_current_expected_gradient)

        local_heat = total_gradient_energy * (1.0 - probability_global_minimum)

        momentum_pressure_p = 1.0 - next_kinetic_gradient_energy / total_momentum_energy

        momentum_pressure = momentum_pressure_p * total_momentum_energy

        total_heat_pressure_per_variable = local_heat + momentum_pressure

        step_distance_ratios = 1.0 / total_heat_pressure_per_variable

        step_distance_proportions = step_distance_ratios / tf.reduce_sum(step_distance_ratios)



        unscaled_update = next_kinetic_gradient_energy + tf.random.normal(tf.shape(next_kinetic_gradient_energy)) * total_heat_pressure_per_variable

        update_proportioned = step_distance_proportions * unscaled_update


        # https://en.wikipedia.org/wiki/Welch%27s_t-test I take the product of all p_values to compute the probability of difference
        batch_size tf.shape(example_losses)[0]

        projection_axis = next_gradient_mean - current_gradient_mean
        projection_axis_length = tf.tensordot(projection_axis, projection_axis)
        current_projected_gradient_mean = tf.tensordot(projection_axis, current_gradient_mean) / projection_axis_length
        next_projected_gradient_mean = tf.tensordot(projection_axis, next_gradient_mean) / projection_axis_length

        current_projected_gradient_variance = tf.tensordot(projection_axis, current_gradient_variance) / projection_axis_length
        next_projected_gradient_variance = tf.tensordot(projection_axis, next_gradient_variance) / projection_axis_length

        standard_error = tf.sqrt(current_projected_gradient_variance / batch_size + next_projected_gradient_variance/ batch_size)

        t_values = (current_projected_gradient_mean - next_projected_gradient_mean) / standard_error

        degrees_of_freedom = tf.sqrt(current_projected_gradient_variance / batch_size + next_projected_gradient_variance/batch_size) /
                            (current_projected_gradient_variance*current_projected_gradient_variance / (batch_size * batch_size) + next_projected_gradient_variance*next_projected_gradient_variance / (batch_size * batch_size)  )


        p_values = tfp.distributions.StudentT(df=degrees_of_freedom)
        log_p_values = tf.log(p_values) # log for numerical stability
        log_full_p_value = tf.reduce_sum(log_p_values)

        # log 1/2 is - 1
        p_greater_than_half = (log_full_p_value + 1) / (tf.abs(log_full_p_value + 1) + sys.float_info.epsilon)
        # 1 if greater, -1 if smaller, 0 if same

        # A bit scared that this will lead to very small learning rates, so earlier
        self.learning_rate = self.learning_rate * tf.power(self.learning_rate_change_rate, p_greater_than_half)

        update = self.learning_rate * update_proportioned

        # total_heat_pressure = tf.reduce_sum(total_heat_pressure_per_variable)
        #
        # mean_heat_pressure = tf.reduce_mean(total_heat_pressure_per_variable)
        #
        # mean_gradient = tf.reduce_mean(per_example_gradients, axis=example_axis)
        # stddev_gradient = tf.reduce_mean(tf.power(per_example_gradients - mean_gradient, 2), axis=example_axis)

        return tf.random.normal(posterior_mean, posterior_std) * posterior_total_gradient_energy

