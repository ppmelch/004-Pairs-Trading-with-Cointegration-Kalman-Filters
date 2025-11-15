from libraries import np 


class KalmanFilter:
    def __init__(self, n, R=1.0, F=None, Q=None, P0=None, w0=None):
        self.n = n
        self.R = R
        self.F = np.eye(n) if F is None else F
        self.Q = np.eye(n)*1e-3 if Q is None else Q
        self.P_t = np.eye(n)*1e-2 if P0 is None else P0
        self.w_t = np.zeros(n) if w0 is None else w0

    def predict(self):
        w_pred = self.F @ self.w_t
        P_pred = self.F @ self.P_t @ self.F.T + self.Q
        return w_pred, P_pred

    def update(self, x, y, w_pred, P_pred):
        H = x.reshape(1, -1)
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        innovation = (y - H @ w_pred).item()
        w_upd = w_pred + (K.flatten() * innovation)
        P_upd = (np.eye(self.n) - K @ H) @ P_pred
        self.w_t = w_upd
        self.P_t = P_upd
        return w_upd, P_upd

def compute_spread(y, x, beta):
    return y - beta * x

def compute_zscore(series, window):
    if len(series) < window:
        return None
    mu = np.mean(series[-window:])
    sd = np.std(series[-window:])
    return (series[-1] - mu) / (sd if sd > 0 else 1e-8)