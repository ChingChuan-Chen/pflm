ElasticNet via ADMM
===================

The ``ElasticNet`` estimator solves the elastic-net problem using ADMM
(Alternating Direction Method of Multipliers).  It supports six distribution
families: Gaussian, Binomial, Poisson, Gamma, Tweedie, and Multinomial.

Gaussian regression
-------------------

Gaussian is the default and most common family.

.. code-block:: python

   import numpy as np
   from pflm.pflm.utils import ElasticNet

   rng = np.random.default_rng(0)
   n, p = 200, 5
   X = rng.standard_normal((n, p))
   true_coef = np.array([1.5, -2.0, 0.0, 0.0, 3.0])
   y = X @ true_coef + 0.5 * rng.standard_normal(n)

   model = ElasticNet(alpha=0.1, l1_ratio=0.5)
   model.fit(X, y)
   print("Coefficients:", model.coef_)
   print("Intercept:", model.intercept_)

   # Predict on new data
   X_new = rng.standard_normal((10, p))
   y_pred = model.predict(X_new)
   print("Predictions:", y_pred[:5])

Logistic regression (Binomial)
------------------------------

For binary classification, use the Binomial family.  ``predict`` returns
probabilities ∈ (0, 1).

.. code-block:: python

   from pflm.pflm.utils import ElasticNet, LinearModelFamily

   rng = np.random.default_rng(42)
   n, p = 300, 4
   X = rng.standard_normal((n, p))
   prob = 1 / (1 + np.exp(-(X @ np.array([1.0, -0.5, 0.0, 2.0]))))
   y = rng.binomial(1, prob).astype(float)

   model = ElasticNet(alpha=0.05, l1_ratio=0.8, family=LinearModelFamily.BINOMIAL)
   model.fit(X, y)
   probs = model.predict(X[:5])
   print("Predicted probabilities:", probs)

Poisson regression
------------------

For count data use the Poisson family.  ``predict`` returns the rate
λ = exp(Xβ).

.. code-block:: python

   from pflm.pflm.utils import ElasticNet, LinearModelFamily

   rng = np.random.default_rng(7)
   n, p = 200, 3
   X = rng.standard_normal((n, p))
   lam = np.exp(X @ np.array([0.5, -0.3, 0.2]))
   y = rng.poisson(lam).astype(float)

   model = ElasticNet(alpha=0.01, l1_ratio=0.5, family=LinearModelFamily.POISSON)
   model.fit(X, y)
   print("Coefficients:", model.coef_)

Tuning the ADMM solver
-----------------------

The ``rho``, ``abs_tol``, ``rel_tol``, ``max_iter`` and ``min_iter``
parameters control the ADMM convergence behaviour.

.. code-block:: python

   model = ElasticNet(
       alpha=0.1,
       l1_ratio=0.9,
       rho=2.0,           # larger rho → faster primal convergence
       max_iter=2000,
       abs_tol=1e-6,
       rel_tol=1e-7,
       min_iter=5,
   )
   model.fit(X, y)
   print(f"Converged in {model.n_iter} iterations")
