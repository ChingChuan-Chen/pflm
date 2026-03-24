Partial Functional Linear Model
================================

``PartialFunctionalLinearModel`` combines scalar predictors with one or
more functional predictors (via FPCA) and fits an ``ElasticNet`` on the
concatenated design matrix ``[scalar_features | FPCA_scores]``.

Single functional feature
--------------------------

.. code-block:: python

   import numpy as np
   from pflm.fpca import FunctionalDataGenerator, FunctionalPCA
   from pflm.pflm.partial_flm import PartialFunctionalLinearModel, FPCAConfig

   # --- Synthetic functional data ---
   t = np.linspace(0.0, 10.0, 51)
   gen = FunctionalDataGenerator(
       t,
       lambda x: np.sin(x) * 0.5,
       lambda x: 1.0 + 0.2 * np.cos(x),
   )
   n = 100
   y_list, t_list = gen.generate(n=n, seed=42)

   # --- Scalar features and response ---
   rng = np.random.default_rng(0)
   scalar = rng.standard_normal((n, 3))
   response = scalar @ np.array([1.0, -0.5, 0.3]) + rng.normal(0, 0.2, n)

   # --- Fit ---
   model = PartialFunctionalLinearModel()
   model.fit([t_list], [y_list], scalar, response)

   print("Scalar features:", model.n_scalar_features_in_)
   print("Functional features:", model.n_functional_features_in_)
   print("Selected #PCs:", model.fpca_models_[0].num_pcs_)
   print("Total coefficients:", model.linear_model_.coef_.shape)

Customise ElasticNet and FPCA
-----------------------------

Use ``linear_opts`` to control the ElasticNet regularisation and
``fpca_configs`` to tune the FPCA pipeline.

.. code-block:: python

   from pflm.fpca import FunctionalPCAMuCovParams

   model = PartialFunctionalLinearModel(
       linear_opts=dict(alpha=0.1, l1_ratio=0.8, max_iter=2000),
       fpca_configs=FPCAConfig(
           num_points_reg_grid=101,
           mu_cov_params=FunctionalPCAMuCovParams(
               method_select_mu_bw="gcv",
               method_select_cov_bw="gcv",
           ),
           fit_params=dict(
               fve_threshold=0.95,
               method_pcs="CE",
           ),
       ),
   )
   model.fit([t_list], [y_list], scalar, response)
   print("Coefficients:", model.linear_model_.coef_)

Multiple functional features
-----------------------------

When more than one functional predictor is available, pass each as a
separate element in the lists and optionally provide per-feature FPCA
configs.

.. code-block:: python

   # Generate a second functional feature
   gen2 = FunctionalDataGenerator(
       t,
       lambda x: np.cos(x) * 0.3,
       lambda x: 0.5 + 0.1 * np.sin(x),
   )
   y_list2, t_list2 = gen2.generate(n=n, seed=7)

   model = PartialFunctionalLinearModel(
       fpca_configs=[
           FPCAConfig(fit_params=dict(method_pcs="IN")),
           FPCAConfig(fit_params=dict(method_pcs="CE", fve_threshold=0.99)),
       ],
   )
   model.fit(
       [t_list, t_list2],
       [y_list, y_list2],
       scalar,
       response,
   )
   print("PCs feature 0:", model.fpca_models_[0].num_pcs_)
   print("PCs feature 1:", model.fpca_models_[1].num_pcs_)

Prediction
----------

.. code-block:: python

   y_pred = model.predict(
       [t_list[:10], t_list2[:10]],
       [y_list[:10], y_list2[:10]],
       scalar[:10],
   )
   print("Predicted (first 5):", y_pred[:5])
