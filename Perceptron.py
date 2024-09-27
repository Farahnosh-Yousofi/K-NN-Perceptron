import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

#Question 6: Implement Perceptron for Custom Dataset
class Perceptron:
    def __init__(self, learning_rate=0.1, n_iters=10):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self._activation(linear_output)
                update = self.learning_rate * (y[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self._activation(linear_output)
        return y_pred

    def _activation(self, x):
        return np.where(x >= 0, 1, 0)

# Custom dataset
custom_data = {
    'feature1': [2, 4, 4, 6, 6, 8, 8, 10, 10, 12],
    'feature2': [1, 2, 4, 4, 6, 6, 8, 8, 10, 10],
    'label': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
custom_df = pd.DataFrame(custom_data)
X_custom = custom_df[['feature1', 'feature2']].values
y_custom = custom_df['label'].values

# Train Perceptron
perceptron = Perceptron(learning_rate=0.1, n_iters=10)
perceptron.fit(X_custom, y_custom)

# Predict on the custom dataset
predictions = perceptron.predict(X_custom)
accuracy = accuracy_score(y_custom, predictions)
print("Perceptron Accuracy on Custom Dataset:", accuracy)