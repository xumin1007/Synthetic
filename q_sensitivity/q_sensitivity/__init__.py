"""q-sensitivity: online pricing benchmark (MOSEK vs PGD, q = 2 / 4 / 1+ln(d)).

Package layout:

- ``q_sensitivity.config``      : global + q-sensitivity configuration.
- ``q_sensitivity.solvers``     : MLE estimators (MOSEK / projected gradient).
- ``q_sensitivity.features``    : feature-generation samplers.
- ``q_sensitivity.utils``       : small numerical helpers (norms, projection, prices).
- ``q_sensitivity.experiments`` : runnable experiment scripts.

The heavy submodules (which pull in cvxpy / matplotlib / mosek) are intentionally
not imported here so that ``import q_sensitivity`` stays cheap; import the
relevant subpackage explicitly instead.
"""

__version__ = "0.1.0"
