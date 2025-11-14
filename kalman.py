from libraries import np 

class KalmanFilter:
    def __init__(self, n: int, R: float = 1.0, F: np.ndarray = None, Q: np.ndarray = None, P0: np.ndarray = None, w0: np.ndarray = None):
        """
        Kalman Filter for online linear regression (time-varying weights).
        """
        self.n = n
        self.R = R
        self.F = F
        self.Q = Q
        self.P0 = P0
        self.w0 = w0

        if self.F is None:
            self.F = np.eye(n)
        if self.P0 is None:
            self.P0 = np.eye(n) * 0.01
        if self.w0 is None:
            self.w0 = np.zeros(n)
        if self.Q is None:
            self.Q = np.eye(n) * 0.001

        self.w_t = self.w0.copy()
        self.P_t = self.P0.copy()
        self.initialized = True

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """Prediction step: returns w_pred, P_pred."""

        w_pred = self.F @ self.w_t
        P_pred = self.F @ self.P_t @ self.F.T + self.Q
        return w_pred, P_pred

    def update(self, x_t: np.ndarray, y_t: float, w_pred: np.ndarray, P_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Update step: given observation (x_t, y_t), return updated weights and covariance."""
        H_t = x_t.reshape(1, -1)

        K_t = P_pred @ H_t.T @ np.linalg.inv(H_t @ P_pred @ H_t.T + self.R)
        w_upd = w_pred + K_t.flatten() * (y_t - H_t @ w_pred)
        P_upd = (np.eye(self.n) - K_t @ H_t) @ P_pred

        self.w_t = w_upd
        self.P_t = P_upd

        return w_upd, P_upd