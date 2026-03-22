import numpy as np
from sklearn.base import MultiOutputMixin, RegressorMixin
from typing import Dict, List, Optional
from pflm.pflm.utils import ElasticNet
from pflm.fpca import FunctionalPCA


class PartialFLMParams:
    """
    Parameters for Partial Functional Linear Model (PFLM).

    Attributes
    ----------
    linear_opts: dict, default={}. The options for ElasticNet class.
    fpca_opts: dict, default={}. The options for FPCA class.
    """
    linear_opts: Dict = {}
    fpca_opts: Dict = {}


class PartialFunctionalLinearModel(MultiOutputMixin, RegressorMixin):
    """
    Partial Functional Linear Model with L1 and L2 regularizations.

    Parameters
    ----------
    linear_opts: dict, default={}. The options for ElasticNet class.
    fpca_opts: dict, default={}. The options for FPCA class.

    Attributes
    ----------
    linear_model: {ElasticNet}. The linear model.
    fpca_models: {List[FPCA]}. The FPCA models of the functional features.
    fitted: bool, this attribute indicates whether the model has been fitted or not.
    """
    fitted: bool = False

    def __init__(self, functional_plm_params: Optional[PartialFLMParams] = None, linear_model_opts: Optional[Dict] = None):
        if functional_plm_params is None:
            self.functional_plm_params = PartialFLMParams()
        else:
            if not isinstance(functional_plm_params, PartialFLMParams):
                raise ValueError('functional_plm_params must be an instance of PartialFLMParams')
            self.functional_plm_params = functional_plm_params
        if linear_model_opts is not None:
            self.functional_plm_params.linear_opts = linear_model_opts
        else:
            self.functional_plm_params.linear_opts = {}
        self.linear_model = ElasticNet(**self.functional_plm_params.linear_opts)

    def fit(
        self, functional_time: List[np.ndarray], functional_features: List[np.ndarray],
        scalar_features: np.ndarray, y: np.ndarray,
        alpha: float = 1.0, l1_ratio: float = 0.5, rho: float = 1.0
    ):
        """
        Fit the partial functional linear model with elastic net regularization.

        Parameters
        ----------
        functional_time: {List[np.ndarray]}. The functional time points which each list is a 1d np.ndarray.
        functional_features: {List[np.ndarray]}. The functional features which each list is a 2d np.ndarray.
        scalar_features: {array-like} of shape (n_scalar_features,). The features matrix for scalar features.
        y: array-like of shape (n_samples,) or (n_samples, n_targets) Target values.
         alpha: float, default=1.0.
             Constant that multiplies the penalty terms.
         l1_ratio: float, default=0.5.
                Mixing parameter where 0 <= l1_ratio <= 1.
        rho: float, default=1.0. ADMM multiplier.

        Returns
        -------
        self: object. Fitted model.
        """
        if len(functional_time) != len(functional_features):
            raise ValueError('Number of functional time points does not match number of functional features')
        self.__functional_time = functional_time
        self.fpca_models = []
        for i in range(len(functional_features)):
            self.fpca_models.append(FunctionalPCA(**self.functional_plm_params).fit(functional_time[i], functional_features[i]))
        self.__num_scalar_features = scalar_features.shape[1]
        feature_list = [scalar_features]
        for i in range(len(functional_features)):
            feature_list.append(self.fpca_models[i].fpca_score)
        self.linear_model.alpha = alpha
        self.linear_model.l1_ratio = l1_ratio
        self.linear_model.rho = rho
        self.linear_model.fit(np.hstack(feature_list), y)
        self.fitted = True
        return self

    def predict(self, new_functional_features: List[np.ndarray], new_scalar_features: np.ndarray):
        """
        Predict using the partial functional linear model.

        Parameters
        ----------
        new_functional_features: List[np.ndarray]. The functional features.
        new_scalar_features: array-like, shape (n_samples, n_features) New functional features.

        Returns
        -------
        The predicted values of the target values.
        """
        if not self.fitted:
            raise ValueError('Model not fitted yet.')
        if len(new_functional_features) != len(self.fpca_models):
            raise ValueError('Number of new functional features does not match number of FPCA models')
        if new_scalar_features.shape[1] != self.__num_scalar_features:
            raise ValueError('Number of new scalar features does not match number of scalar features to input model')
        feature_list = [new_scalar_features]
        for i in range(len(new_functional_features)):
            if new_functional_features[i].shape[1] != self.__functional_time[i].size:
                raise ValueError(f'The number of new functional feature {i} does not match the number of functional time points.')
            _, new_fpca_score, _ = self.fpca_models[i].predict(new_functional_features[i])
            feature_list.append(new_fpca_score)
        return self.linear_model.predict(np.hstack(feature_list))
