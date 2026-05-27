"""Metrics for the FAIR-Universe Weak Lensing ML Uncertainty Challenge.

Naming follows the competition overview PDF:

  * ``score_inference`` — "KL divergence between the true Gaussian-like
    posterior and the Gaussian with the predicted mean and standard
    deviation," with a ``lambda = 1e3`` penalty on the point estimates.
    The leaderboard maximises this score; the loss is its negative.

  * ``coverage_68`` — empirical fraction of test samples whose truth
    falls inside the predicted 68 percent confidence interval
    ``[mu - sigma, mu + sigma]``.  A well-calibrated Gaussian predictor
    should sit at ~0.68.  The PDF reports a single "Coverage" number on
    the leaderboard without spelling out how it aggregates the two
    parameters, so we log per-parameter coverages plus their mean (which
    equals pooled coverage when both parameters have the same N).

Model output convention (B, 4):
    pred[:, 0:2] = (mu_Om,        mu_S8)
    pred[:, 2:4] = (log_sigma_Om, log_sigma_S8)

We predict ``log sigma`` rather than ``sigma`` for numerical stability
(keeps sigma strictly positive, no exp blow-up in the loss).
"""

from __future__ import annotations

import torch


#: Hard bounds on log(sigma) to prevent the canonical "sigma collapse"
#: failure mode of Gaussian NLL: the network learns log_sigma -> -inf
#: because the +2*log_sigma term in the score rewards it, and coverage
#: drops to ~0. We clamp wide enough that the loss can still adapt
#: per-sample (sigma in [~0.002, ~7]) but can't run to zero.
LOG_SIGMA_MIN = -6.0
LOG_SIGMA_MAX = 2.0


def _split(pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(mu, log_sigma, sigma)`` from a (B, 4) prediction.

    ``log_sigma`` is clamped to a finite range so training cannot drive
    it to -infinity (the sigma-collapse failure mode of Gaussian NLL).
    """
    mu = pred[:, :2]
    log_sigma = pred[:, 2:].clamp(min=LOG_SIGMA_MIN, max=LOG_SIGMA_MAX)
    sigma = torch.exp(log_sigma)
    return mu, log_sigma, sigma


def score_inference_loss(
    pred: torch.Tensor, target: torch.Tensor, lam: float = 1e3
) -> torch.Tensor:
    """Negative of the PDF's ``score_inference`` (we minimise).

    .. math::
        \\text{score} = -\\frac{1}{N}\\sum_i
        \\left[\\frac{(\\hat\\mu_i - \\mu^\\text{truth}_i)^2}{\\hat\\sigma_i^2}
        + \\log\\hat\\sigma_i^2
        + \\lambda\\,(\\hat\\mu_i - \\mu^\\text{truth}_i)^2\\right]

    summed over both cosmological parameters.  The ``lambda = 1e3`` term
    prevents the trivial "huge sigma" solution by keeping the point
    estimate honest.
    """
    mu, log_sigma, _ = _split(pred)
    sq_err = (mu - target) ** 2
    inv_var = torch.exp(-2.0 * log_sigma)
    # log(sigma^2) = 2 * log_sigma
    per_param = sq_err * inv_var + 2.0 * log_sigma + lam * sq_err
    return per_param.sum(dim=1).mean()


@torch.no_grad()
def score_inference_value(
    pred: torch.Tensor, target: torch.Tensor, lam: float = 1e3
) -> torch.Tensor:
    """The signed leaderboard score (higher = better)."""
    return -score_inference_loss(pred, target, lam=lam)


@torch.no_grad()
def mse_mu(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error on the point estimates only (mu vs truth)."""
    mu, _, _ = _split(pred)
    return ((mu - target) ** 2).mean()


@torch.no_grad()
def coverage_68(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Per-parameter 68% coverage plus the mean across parameters."""
    mu, _, sigma = _split(pred)
    inside = (target - mu).abs() <= sigma  # (B, 2) bool
    return {
        "coverage_Om": inside[:, 0].float().mean().item(),
        "coverage_S8": inside[:, 1].float().mean().item(),
        "coverage_mean": inside.float().mean().item(),
    }


# -----------------------------------------------------------------------------
# Sum-form helpers for batched accumulation inside the training loop.
# Each returns a (sum_over_batch, count_added) pair so the trainer can
# average correctly across uneven final batches.
# -----------------------------------------------------------------------------


@torch.no_grad()
def eval_metric_sums(pred: torch.Tensor, target: torch.Tensor, lam: float = 1e3) -> dict:
    """Per-batch sums for the metrics we want to track on the val set.

    Each value is a batch-sum; the trainer divides by the running sample
    count to get per-sample averages.  Keys are the final MLflow metric
    names (the trainer prefixes with ``train_``/``test_`` as appropriate).
    """
    mu, log_sigma, sigma = _split(pred)
    sq_err = (mu - target) ** 2
    inv_var = torch.exp(-2.0 * log_sigma)
    score_loss_per_sample = (sq_err * inv_var + 2.0 * log_sigma + lam * sq_err).sum(dim=1)
    inside = (target - mu).abs() <= sigma

    return {
        "score_loss": score_loss_per_sample.sum(),               # leaderboard score = -mean
        "mse": sq_err.sum() / sq_err.shape[1],                   # MSE on mu, averaged over the 2 params
        "coverage_Om": inside[:, 0].float().sum(),
        "coverage_S8": inside[:, 1].float().sum(),
        "coverage_mean": inside.float().mean(dim=1).sum(),
    }
