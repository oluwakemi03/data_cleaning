import unittest
import pandas as pd
from app import handle_missing_data, median_filtering, detect_outliers_iqr

class CustomTextTestRunner(unittest.TextTestRunner):
    def run(self, test):
        for test_case in test:
            print(f"Running test: {test_case.id()}")
        return super().run(test)

class TestDataCleaning(unittest.TestCase):
    def setUp(self):
        # Sample input data with missing values
        self.df_missing_values = pd.DataFrame({
            'A': [1, 2, pd.NA, 4],
            'B': [5, pd.NA, 7, 8],
            'C': [pd.NA, 10, 11, 12]
        })
        
        # Sample input data with noise
        self.df_noisy_data = pd.DataFrame({
            'A': [1, 2, 5, 10],
            'B': [5, 6, 25, 30],
            'C': [10, 20, 50, 100]
        })
        
        # Sample input data with outliers
        self.df_outliers = pd.DataFrame({
            'A': [1, 2, 3, 100],
            'B': [5, 6, 7, 200],
            'C': [10, 20, 30, 1000]
        })

    def test_handle_missing_data(self):
        cleaned_df = handle_missing_data(self.df_missing_values)
        num_missing_values = cleaned_df.isnull().sum().sum()
        self.assertEqual(num_missing_values, 3)  # Assert no missing values

        
    def test_median_filtering(self):
        cleaned_df = median_filtering(self.df_noisy_data)
        self.assertTrue((cleaned_df - self.df_noisy_data).abs().max().max() < 1)  # Assert noise reduction
        
    def test_detect_outliers_iqr(self):
        cleaned_df = detect_outliers_iqr(self.df_outliers)
        self.assertTrue((cleaned_df - self.df_outliers).abs().max().max() < 50)  # Assert outliers removed within expected range

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataCleaning)
    runner = CustomTextTestRunner()
    runner.run(suite)
