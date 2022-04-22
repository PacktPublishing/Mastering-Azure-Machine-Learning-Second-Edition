import unittest
import pandas as pd

class TestDataFrameStats(unittest.TestCase):
    
    def setUp(self):
        # initialize and load df
        self.df = pd.DataFrame(data={'data': [0,1,2,3]})
    
    def test_min(self):
        self.assertGreaterEqual(self.df.min().values[0], 0)

    def test_max(self):
        self.assertLessEqual(self.df.max().values[0], 100)