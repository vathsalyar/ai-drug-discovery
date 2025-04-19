import unittest
import numpy as np
from ml_model import train_ml
from dl_model import train_dl
from qml_model import train_qml
from qnn_model import create_qnn

class TestDrugModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create fake data for all tests
        np.random.seed(42)
        cls.X = np.random.rand(100, 6)  # 100 samples, 6 features
        cls.y = np.random.rand(100)      # Random target values
        cls.X_train, cls.X_test = cls.X[:80], cls.X[80:]
        cls.y_train, cls.y_test = cls.y[:80], cls.y[80:]

    def test_ml_model(self):
        model, preds = train_ml(self.X_train, self.y_train, self.X_test)
        self.assertEqual(len(preds), len(self.y_test))

    def test_dl_model(self):
        model, preds = train_dl(self.X_train, self.y_train, self.X_test)
        self.assertEqual(len(preds), len(self.y_test))

    def test_qml_model(self):
        # Test with reduced features for quantum models
        X_train_reduced = self.X_train[:, :4]  # Use only 4 features
        X_test_reduced = self.X_test[:, :4]
        circuit, params, preds = train_qml(X_train_reduced, self.y_train, X_test_reduced)
        self.assertTrue(len(preds) > 0)

    def test_qnn_model(self):
        X_train_reduced = self.X_train[:, :4]
        X_test_reduced = self.X_test[:, :4]
        qnn, preds = create_qnn(X_train_reduced, self.y_train, X_test_reduced)
        self.assertTrue(len(preds) > 0)

if __name__ == '__main__':
    unittest.main()