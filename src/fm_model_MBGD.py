import numpy as np
import pickle

class FactorizationMachine:
    def __init__(self, X=None, y=None, k=10, lambda_L2=0.01, learning_rate=0.001, batch_size=32):
        self.k = k
        self.batch_size = batch_size 
        self.lambda_L2 = lambda_L2
        self.learning_rate = learning_rate
        
        self.w0 = None
        self.W = None
        self.V = None
        self.n = None
        self.m = None
        self.max_values = None
        self.costs = []

    def _initialize_weights(self, X):
        self.n = X.shape[1]
        self.m = X.shape[0]
        max_values = np.max(X, axis=0)
        self.V = np.random.normal(size=(sum(max_values) + 2, self.k))
        self.w = np.random.randn(1, sum(max_values) + 2)
        self.w0 = 0

    
    def _one_hot_encoder(self, Xi, max_values):
        encoded_Xi = np.empty(0, dtype=int)
        for idx, feature in enumerate(Xi):
            encoded_feature = np.zeros(max_values[idx] + 1)
            encoded_feature[feature] = 1
            encoded_Xi = np.concatenate((encoded_Xi, encoded_feature))
        return encoded_Xi

    def _predict_row(self, Xi):
        y_pred = (
            self.w0
            + np.dot(self.w, Xi.T)
            + 0.5 * np.sum((np.dot(Xi, self.V) ** 2 - np.dot(Xi**2, self.V**2)))
        )
        return np.array(y_pred)

    def predict(self, X):
        max_values = np.max(X, axis=0)
        X_encoded = np.array([self._one_hot_encoder(X[i], max_values) for i in range(X.shape[0])])
        linear_term = self.w0 + np.dot(X_encoded, self.w.T)
        interaction_term = 0.5 * np.sum(
            (X_encoded @ self.V) ** 2 - (X_encoded**2) @ (self.V**2), axis=1
        )
        return linear_term + interaction_term

    def _calc_cost(self, y_true, y_pred):
        MSE = (y_true - y_pred) ** 2
        return MSE

    def _compute_gradients(self, X_batch, y_batch):
        batch_size = len(y_batch)
        grad_w0 = 0
        grad_w = np.zeros_like(self.w)
        grad_V = np.zeros_like(self.V)

        for i in range(batch_size):
            Xi = X_batch[i]
            y_i = y_batch[i]
            y_hat = self._predict_row(Xi)
            error = y_i - y_hat

            grad_w0 += -2 * error
            grad_w += -2 * Xi * error + 2 * self.lambda_L2 * self.w
            grad_V += (
                -2
                * (np.outer(Xi, np.dot(Xi, self.V)) - self.V * (Xi**2).reshape(-1, 1))
                * error
                + 2 * self.lambda_L2 * self.V
            )

        # Average over batch
        grad_w0 /= batch_size
        grad_w /= batch_size
        grad_V /= batch_size

        return grad_w0, grad_w, grad_V

    def _update_parameters(self, grad_w0, grad_w, grad_V):
        self.w0 -= self.learning_rate * grad_w0
        self.w -= self.learning_rate * grad_w
        self.V -= self.learning_rate * grad_V

    def fit(self,X, y, num_epochs=10, patience=10, tol=1e-6, print_cost=False):
        self._initialize_weights(X)
        best_cost = float("inf")
        no_improvement_count = 0

        for epoch in range(num_epochs):
            cost = 0

            # Shuffle dataset
            indices = np.arange(self.m)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Process mini-batches
            for i in range(0, self.m, self.batch_size):
                X_batch = []
                y_batch = []
                batch_indices = indices[i : i + self.batch_size]
                for j in batch_indices:
                    X_batch.append(self._one_hot_encoder(X_shuffled[j]))
                    y_batch.append(y_shuffled[j])

                X_batch = np.array(X_batch)
                y_batch = np.array(y_batch)

                # Compute gradients for the batch
                grad_w0, grad_w, grad_V = self._compute_gradients(X_batch, y_batch)

                # Update parameters
                self._update_parameters(grad_w0, grad_w, grad_V)

                # Compute cost for the batch
                y_pred_batch = np.array([self._predict_row(Xi) for Xi in X_batch])
                cost += np.mean(self._calc_cost(y_batch, y_pred_batch))

            cost /= self.m // self.batch_size  # Average cost over mini-batches
            self.costs.append(cost)

            # Early stopping
            if cost < best_cost - tol:
                best_cost = cost
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                print(f"Early stopping at epoch {epoch}, cost: {cost}")
                break

            if print_cost:
                print(f"Cost after epoch {epoch}: {cost}")

        return self

    def evaluate(self, y_pred, y_true):
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        return mse, rmse


    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)