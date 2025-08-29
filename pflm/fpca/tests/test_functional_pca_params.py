import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from pflm.fpca.functional_pca import (
    FunctionalPCA,
    FunctionalPCAMuCovParams,
    FunctionalPCAUserDefinedParams,
)
from pflm.smooth import KernelType


class TestFunctionalPCAMuCovParamsValidation:
    @pytest.mark.parametrize("bad_bw", [0.0, -1.0])
    def test_bw_mu_invalid(self, bad_bw):
        with pytest.raises(ValueError):
            FunctionalPCAMuCovParams(bw_mu=bad_bw)

    @pytest.mark.parametrize("bad_bw", [0.0, -1.0])
    def test_bw_cov_invalid(self, bad_bw):
        with pytest.raises(ValueError):
            FunctionalPCAMuCovParams(bw_cov=bad_bw)

    @pytest.mark.parametrize("bad_method", ["bad", 123, None])
    def test_estimate_method_invalid(self, bad_method):
        with pytest.raises(ValueError):
            FunctionalPCAMuCovParams(estimate_method=bad_method)

    def test_kernel_type_invalid(self):
        with pytest.raises(ValueError):
            # pass a wrong type instead of KernelType enum
            FunctionalPCAMuCovParams(kernel_type=123)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", ["BAD", 1, None])
    def test_method_select_mu_bw_invalid(self, bad):
        with pytest.raises(ValueError):
            FunctionalPCAMuCovParams(method_select_mu_bw=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", ["BAD", 1, None])
    def test_method_select_cov_bw_invalid(self, bad):
        with pytest.raises(ValueError):
            FunctionalPCAMuCovParams(method_select_cov_bw=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_bool", [0, 1, "x"])
    def test_apply_geo_avg_cov_bw_type(self, bad_bool):
        with pytest.raises(ValueError):
            FunctionalPCAMuCovParams(apply_geo_avg_cov_bw=bad_bool)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "10"])
    def test_cv_folds_mu_invalid(self, bad):
        with pytest.raises(ValueError):
            FunctionalPCAMuCovParams(cv_folds_mu=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "10"])
    def test_cv_folds_cov_invalid(self, bad):
        with pytest.raises(ValueError):
            FunctionalPCAMuCovParams(cv_folds_cov=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_seed", [-1, 3.14, "7"])
    def test_random_seed_invalid(self, bad_seed):
        with pytest.raises(ValueError):
            FunctionalPCAMuCovParams(random_seed=bad_seed)  # type: ignore[arg-type]

    def test_params_repr(self):
        p = FunctionalPCAMuCovParams(
            bw_mu=None,
            bw_cov=None,
            estimate_method="smooth",
            kernel_type=KernelType.EPANECHNIKOV,
            method_select_mu_bw="gcv",
            method_select_cov_bw="cv",
            apply_geo_avg_cov_bw=False,
            cv_folds_mu=10,
            cv_folds_cov=5,
            random_seed=42,
        )
        s = repr(p)
        assert "bw_mu=None" in s and "bw_cov=None" in s and "estimate_method='smooth'" in s


class TestFunctionalPCAUserDefinedParamsValidation:
    def test_t_mu_mu_only_one_provided(self):
        with pytest.raises(ValueError):
            FunctionalPCAUserDefinedParams(t_mu=np.array([0.0, 1.0]), mu=None)
        with pytest.raises(ValueError):
            FunctionalPCAUserDefinedParams(t_mu=None, mu=np.array([0.0, 1.0]))

    def test_t_mu_mu_length_mismatch(self):
        with pytest.raises(ValueError):
            FunctionalPCAUserDefinedParams(t_mu=np.array([0.0, 1.0]), mu=np.array([1.0]))

    def test_t_cov_cov_only_one_provided(self):
        with pytest.raises(ValueError):
            FunctionalPCAUserDefinedParams(t_cov=np.array([0.0, 1.0]), cov=None)
        with pytest.raises(ValueError):
            FunctionalPCAUserDefinedParams(t_cov=None, cov=np.eye(2))

    def test_cov_not_square_or_mismatch_with_t_cov(self):
        with pytest.raises(ValueError):
            FunctionalPCAUserDefinedParams(t_cov=np.array([0.0, 1.0]), cov=np.array([[1.0, 2.0, 3.0]]))
        with pytest.raises(ValueError):
            FunctionalPCAUserDefinedParams(t_cov=np.array([0.0, 1.0, 2.0]), cov=np.eye(2))

    @pytest.mark.parametrize("bad", [-1.0, -1, "a"])
    def test_sigma2_invalid(self, bad):
        with pytest.raises(ValueError):
            FunctionalPCAUserDefinedParams(sigma2=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [-1.0, -1, "a"])
    def test_rho_invalid(self, bad):
        with pytest.raises(ValueError):
            FunctionalPCAUserDefinedParams(rho=bad)  # type: ignore[arg-type]

    def test_user_params_repr(self):
        p = FunctionalPCAUserDefinedParams(
            t_mu=np.array([0.0, 1.0]),
            mu=np.array([2.0, 3.0]),
            t_cov=np.array([0.0, 1.0]),
            cov=np.eye(2),
            sigma2=0.0,
            rho=0.0,
        )
        s = repr(p)
        assert "sigma2=0.0" in s and "rho=0.0" in s


class TestFunctionalPCAAPIAndHyperParamChecks:
    def test_fitted_values_before_fit_raises(self):
        fpca = FunctionalPCA()
        with pytest.raises(NotFittedError):
            fpca.fitted_values()

    def test_predict_before_fit_raises(self):
        fpca = FunctionalPCA()
        with pytest.raises(NotFittedError):
            fpca.predict(y=[np.array([0.0, 1.0])], t=[np.array([0.0, 1.0])])

    @pytest.mark.parametrize("bad", ["BAD", 123, None])
    def test_check_fit_params_method_pcs_invalid(self, bad):
        fpca = FunctionalPCA()
        with pytest.raises(ValueError):
            fpca._FunctionalPCA__check_fit_params(
                method_pcs=bad,  # type: ignore[arg-type]
                method_select_num_pcs="FVE",
                method_rho="vanilla",
                max_num_pcs=10,
                if_impute_scores=True,
                if_shrinkage=False,
                if_fit_eigen_values=False,
                fve_threshold=0.99,
            )

    def test_check_fit_params_method_select_num_pcs_type_and_value(self):
        fpca = FunctionalPCA()
        # wrong type
        with pytest.raises(ValueError):
            fpca._FunctionalPCA__check_fit_params(
                "CE",
                method_select_num_pcs=3.14,
                method_rho="vanilla",
                max_num_pcs=10,
                if_impute_scores=True,
                if_shrinkage=False,
                if_fit_eigen_values=False,
                fve_threshold=0.99,
            )
        # bad int
        with pytest.raises(ValueError):
            fpca._FunctionalPCA__check_fit_params(
                "CE",
                method_select_num_pcs=0,
                method_rho="vanilla",
                max_num_pcs=10,
                if_impute_scores=True,
                if_shrinkage=False,
                if_fit_eigen_values=False,
                fve_threshold=0.99,
            )
        # bad str
        with pytest.raises(ValueError):
            fpca._FunctionalPCA__check_fit_params(
                "CE",
                method_select_num_pcs="BAD",
                method_rho="vanilla",
                max_num_pcs=10,
                if_impute_scores=True,
                if_shrinkage=False,
                if_fit_eigen_values=False,
                fve_threshold=0.99,
            )

    def test_check_fit_params_method_rho_invalid(self):
        fpca = FunctionalPCA()
        with pytest.raises(ValueError):
            fpca._FunctionalPCA__check_fit_params(
                "CE",
                method_select_num_pcs="FVE",
                method_rho="bad",
                max_num_pcs=10,
                if_impute_scores=True,
                if_shrinkage=False,
                if_fit_eigen_values=False,
                fve_threshold=0.99,
            )

    @pytest.mark.parametrize("bad", [None, 0, -1, 3.14, "10"])
    def test_check_fit_params_max_num_pcs_invalid(self, bad):
        fpca = FunctionalPCA()
        with pytest.raises(ValueError):
            fpca._FunctionalPCA__check_fit_params(
                "CE",
                method_select_num_pcs="FVE",
                method_rho="vanilla",
                max_num_pcs=bad,  # type: ignore[arg-type]
                if_impute_scores=True,
                if_shrinkage=False,
                if_fit_eigen_values=False,
                fve_threshold=0.99,
            )

    @pytest.mark.parametrize("bad_fve", [-0.1, 0.0, 1.1, "x"])
    def test_check_fit_params_fve_threshold_invalid(self, bad_fve):
        fpca = FunctionalPCA()
        with pytest.raises(ValueError):
            fpca._FunctionalPCA__check_fit_params(
                "CE",
                method_select_num_pcs="FVE",
                method_rho="vanilla",
                max_num_pcs=10,
                if_impute_scores=True,
                if_shrinkage=False,
                if_fit_eigen_values=False,
                fve_threshold=bad_fve,
            )  # type: ignore[arg-type]

    @pytest.mark.parametrize("flag_name", ["if_impute_scores", "if_shrinkage", "if_fit_eigen_values"])
    def test_check_fit_params_flags_type(self, flag_name):
        fpca = FunctionalPCA()
        kwargs = dict(
            method_pcs="CE",
            method_select_num_pcs="FVE",
            method_rho="vanilla",
            max_num_pcs=10,
            if_impute_scores=True,
            if_shrinkage=False,
            if_fit_eigen_values=False,
            fve_threshold=0.99,
        )
        kwargs[flag_name] = "x"  # wrong type
        with pytest.raises(ValueError):
            fpca._FunctionalPCA__check_fit_params(**kwargs)
