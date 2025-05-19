import unittest
from data_preprocessing import load_and_preprocess_data
from train_model import model
from sklearn.metrics import mean_squared_error

class TestPipeline(unittest.TestCase):
    def test_no_missing_values(self):
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
        self.assertFalse((X_train == float('nan')).any(), "Missing values found")

    def test_feature_scaling_shape(self):
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
        self.assertEqual(X_train.shape[1], 8, "Unexpected number of features")

    def test_model_performance(self):
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        self.assertLess(mse, 30, "MSE is too high")

if __name__ == '__main__':
    unittest.main()