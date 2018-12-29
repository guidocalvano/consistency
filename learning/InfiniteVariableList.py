import tensorflow as tf


class InfiniteVariableList:

    def __init__(self, constants=None):
        if constants is None:
            constants = []

        self.constants = constants.copy()
        self.depth = 0
        # self.variables = []

    def pop(self, a=0):
        self.depth += 1
        if len(self.constants) > 0:
            return self.constants.pop(0)

        return 1.0
        # return self.pop_variable()

    # def pop_variable(self):
    #     next_var = tf.Variable(1.0)
    #
    #     self.variables.append(next_var)
    #
    #     return next_var

