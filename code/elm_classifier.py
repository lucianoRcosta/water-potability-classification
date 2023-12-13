import numpy as np

class ELMClassifier:
    def __init__(self, input_size, hidden_size, regularization=1e-3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.regularization = regularization
        self.weights_input_hidden = np.random.rand(hidden_size, input_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.weights_hidden_output = None
        self.bias_output = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, X, y):
        hidden_output = self._sigmoid(np.dot(X, self.weights_input_hidden.T) + self.bias_hidden)

        hidden_output_bias = np.column_stack([hidden_output, np.ones((X.shape[0], 1))])

        cov_matrix = hidden_output_bias.T @ hidden_output_bias
        cov_matrix += self.regularization * np.eye(cov_matrix.shape[0])  # Add regularization to the diagonal
        self.weights_hidden_output = np.dot(np.linalg.pinv(cov_matrix), hidden_output_bias.T @ y)

    def predict(self, X):
        hidden_output = self._sigmoid(np.dot(X, self.weights_input_hidden.T) + self.bias_hidden)

        hidden_output_bias = np.column_stack([hidden_output, np.ones((X.shape[0], 1))])

        y_pred = np.dot(hidden_output_bias, self.weights_hidden_output)

        y_pred_binary = (y_pred >= 0.5).astype(int)

        return y_pred_binary