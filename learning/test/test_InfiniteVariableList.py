import unittest
from learning.InfiniteVariableList import InfiniteVariableList


class test_InfiniteVariableList(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_InfiniteVariableList(self):
        l = InfiniteVariableList()

        self.assertTrue(l.variables == [])
        self.assertTrue(l.constants == [])

        v = l.pop(0)

        self.assertTrue(l.variables == [v])
        self.assertTrue(l.constants == [])

        constants = [1,2,3]
        l = InfiniteVariableList(constants)

        constants = [1,2,3]  # make a copy
        correct_variables = []

        self.assertTrue(l.variables == [])
        self.assertTrue(l.constants == constants)

        l.pop()

        self.assertTrue(l.variables == [])
        self.assertTrue(l.constants == constants[1:])

        l.pop()

        self.assertTrue(l.variables == [])
        self.assertTrue(l.constants == constants[2:])

        l.pop()

        self.assertTrue(l.variables == correct_variables)
        self.assertTrue(l.constants == [])

        correct_variables.append(l.pop())

        self.assertTrue(l.variables == correct_variables)
        self.assertTrue(l.constants == [])