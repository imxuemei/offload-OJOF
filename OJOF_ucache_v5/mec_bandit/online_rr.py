# -*- coding:utf-8 -*-
import numpy as np


class OnlineRidgeRegression:

    def __init__(self, num_features: int, alpha=1):
        self.alpha = alpha
        self.num_features = num_features
        self.A = np.identity(num_features)  # (d, d), A=transpose(X) * X + I
        self.b = np.zeros((num_features, 1))  # (d, 1)

    def reset(self):
        self.A = np.identity(self.num_features)  # (d, d), A=transpose(X) * X + I
        self.b = np.zeros((self.num_features, 1))  # (d, 1)

    def validate_x(self, x):
        # x is (d,1)
        assert len(x.shape) == 2, "x should be 2-D array"
        assert x.shape[0] == self.num_features and x.shape[1] == 1, "x should be ({},1), not ({},{})".format(
            self.num_features, x.shape[0], x.shape[1])

    def fit_one(self, x: np.ndarray, y: float):
        self.validate_x(x)
        self.A = self.A + np.matmul(x, x.T)
        self.b = self.b + y * x

    def predict(self, x: np.ndarray) -> float:
        self.validate_x(x)
        A_inv = np.linalg.inv(self.A)
        theta = np.matmul(A_inv, self.b)  # (d,d)*(d,1) = (d,1)
        norm_coeff = np.sqrt(np.matmul(np.matmul(x.T, A_inv), x))
        y_hat = np.matmul(theta.T, x) + self.alpha * norm_coeff #(1,1)
        return y_hat[0][0]

    def get_theta(self):
        A_inv = np.linalg.inv(self.A)
        return np.matmul(A_inv, self.b)  # (d,d)*(d,1) = (d,1)



