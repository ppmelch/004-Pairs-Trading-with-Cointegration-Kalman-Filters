from libraries import np


class KalmanFilter:
    """
    Linear Kalman Filter for sequential state estimation of dynamic parameters.

    Attributes
    ----------
    n : int
        Dimensionality of the state vector.
    R : float
        Observation noise variance.
    F : np.ndarray
        Transition matrix.
    Q : np.ndarray
        Process noise covariance.
    P_t : np.ndarray
        Current covariance estimate.
    w_t : np.ndarray
        Current state vector.
    """

    def __init__(self, n, R=1.0, F=None, Q=None, P0=None, w0=None):
        self.n = n
        self.R = R
        self.F = np.eye(n) if F is None else F
        self.Q = np.eye(n)*1e-3 if Q is None else Q
        self.P_t = np.eye(n)*1e-2 if P0 is None else P0
        self.w_t = np.zeros(n) if w0 is None else w0

    def predict(self):
        """
        Perform the prediction step of the Kalman Filter.

        Returns
        -------
        tuple
            (predicted_state, predicted_covariance)
        """
        w_pred = self.F @ self.w_t
        P_pred = self.F @ self.P_t @ self.F.T + self.Q
        return w_pred, P_pred

    def update(self, x, y, w_pred, P_pred):
        """
        Update the state estimate with a new observation.

        Parameters
        ----------
        x : np.ndarray
            Observation vector.
        y : float
            Observed value.
        w_pred : np.ndarray
            Predicted state.
        P_pred : np.ndarray
            Predicted covariance.

        Returns
        -------
        tuple
            (updated_state, updated_covariance)
        """
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
    """
    Compute price spread for a given hedge ratio.

    Returns
    -------
    float
        Spread value.
    """
    return y - beta * x


def compute_zscore(series, window):
    """
    Compute rolling z-score of a series.

    Parameters
    ----------
    series : list or np.ndarray
        Time series values.
    window : int
        Rolling window size.

    Returns
    -------
    float or None
        Z-score if enough data points are available, otherwise None.
    """
    if len(series) < window:
        return None
    mu = np.mean(series[-window:])
    sd = np.std(series[-window:])
    return (series[-1] - mu) / (sd if sd > 0 else 1e-8)
