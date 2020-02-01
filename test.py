
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

class TestEigenfunction(unittest.TestCase):
    """Test avoidance.get_eigenfunction_values."""

    @classmethod
    def get_pairs_symmetric_matrices(cls, samples=100):
        """Generate sample pairs from avoidance.get_symmetric_matrix."""
        for param in TestSymmetric.get_random_parameters(samples):
            yield (avoidance.get_symmetric_matrix(*param) for _ in range(2))

    @classmethod
    def get_random_time_intervals(cls, samples=100):
        for _ in range(samples):
            yield np.linspace(-100, 100, 100)

    def test_shape(self):
        """Test matrix is of shape (A.shape[0], T.shape[0])."""
        for A,B in self.get_pairs_symmetric_matrices():
            T = next(self.get_random_time_intervals(1))
            F = avoidance.get_eigenfunction_values(A, B, T)
            self.assertEqual(F.shape, (A.shape[0], T.shape[0]))




if __name__ == '__main__':
    unittest.main()
