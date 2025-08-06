import numpy as np
from sklearn.base import BaseEstimator
from pflm.interp import interp1d, interp2d
from pflm.smooth import Polyfit1DModel, Polyfit2DModel
from pflm.utils.utility import trapz


class FunctionalPCA(BaseEstimator):
    def __init__(
        self,

    ) -> None:
        pass

    def fit(self, t: np.ndarray, y: np.ndarray, w: np.ndarray = None) -> "FunctionalPCA":
        if w is None:
            w = np.ones_like(y)
            w[np.isnan(y)] = np.nan
        elif w.ndim != 1 or w.shape[0] != y.shape[0]:
            raise ValueError("Weights w must be a 1D array with the same length as the number of samples in y.")

        return self
