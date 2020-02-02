
import avoidance

import numpy as np
import random
import unittest

class TestSymmetric(unittest.TestCase):
    """Test avoidance.get_symmteric_matrix."""

    @classmethod
    def get_random_parameters(cls, samples=100):
        """Generate number of samples of parameters size and scale."""
        for _ in range(samples):
            size = random.randint(2,10)
            scale = random.randint(1,100) * random.random()
            yield (size, scale)

    def test_shape(self):
        """Test size parameter."""
        for size, scale in self.get_random_parameters():
            matrix = avoidance.get_symmetric_matrix(size, scale)
            shape = (size, size)
            self.assertEqual(matrix.shape, shape,
                msg='Incorect shape. Expected {}. Got {}.'.format(shape,
                    matrix.shape))

    def test_values_range(self):
        """Test values within given scale."""
        for size, scale in self.get_random_parameters():
            matrix = avoidance.get_symmetric_matrix(size, scale)
            self.assertTrue(np.all(matrix < scale) and np.all(matrix > -scale),
                msg="""Incorect values range. Expected values between (%f, %f).
                    Got min value %f and max value %f.""" % (-scale, scale,
                    np.min(matrix), np.max(matrix)))

    def test_symmetric(self):
        """Test matrix is symmetric i.e. invariant under transpose."""
        for size, scale in self.get_random_parameters():
            matrix = avoidance.get_symmetric_matrix(size, scale)
            self.assertTrue(np.all(matrix == matrix.T), msg="""Matrix not
                symmetric {}""".format(matrix))

if __name__ == '__main__':
    unittest.main()
