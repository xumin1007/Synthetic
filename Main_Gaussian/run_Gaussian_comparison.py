"""Entry point for the Gaussian-feature online pricing benchmark.

Thin launcher that delegates to :func:`gaussian_pricing.experiment.main`, so the
experiment can be started with either of:

    python run_Gaussian_comparison.py
    python -m gaussian_pricing.experiment

Set ``Gaussian_SMOKE=1`` for a fast smoke test. Results are written under
``results/<Tier>_SNR/``.
"""

from gaussian_pricing.experiment import main

if __name__ == "__main__":
    main()
