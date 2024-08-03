import unittest
from Utils.utils import load_data

class TestUtils(unittest.TestCase):
    
    def test_load_data(self):
        df = load_data('Resource/data_source/named_df.csv')
        self.assertIsInstance(df, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()
