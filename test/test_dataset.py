import unittest

from muscima.dataset import CVC_MUSCIMA


class CVC_MUSCIMATest(unittest.TestCase):
    def test_init(self):
        root = "/muscima/v1.0/data/images/" # path to the dataset
        
        CVC_MUSCIMA(root=root, validate=False) # without validation
        CVC_MUSCIMA(root=root, validate=True) # with validation


if __name__ == '__main__':
    unittest.main()
