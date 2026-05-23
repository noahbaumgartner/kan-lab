"""FAIR Universe Weak Lensing ML Uncertainty Challenge - Phase 1 dataset.

Loads the convergence maps released by the FAIR-Universe Cosmology Challenge
and exposes them through PyTorch ``Dataset`` objects that *add pixel-level
shape-noise on the fly* — one sample at a time — so the full noisy training
cube is never materialized.

Memory note
-----------
The naive notebook recipe::

    kappa = np.zeros((Ncosmo, Nsys, H, W), dtype=np.float16)   # ~13 GB
    kappa[..., mask] = np.load(...)                            # scatter
    noisy = Utility.add_noise(kappa.astype(np.float64), ...)   # >150 GB

upcasts to float64 and allocates a second full-size array.  Even the
``add_noise`` variant that loops map-by-map still has to hold a second
(101, 256, 1424, 176) float32 output buffer (~26 GB) plus the float16 input.

This dataset avoids both:

* the maps are kept in their **compact masked-pixel form** loaded straight
  from disk — shape ``(Ncosmo, Nsys, n_unmasked)`` float16, which is the
  natural on-disk layout and roughly ``mask.mean()`` smaller than the dense
  cube;
* noise is drawn and added in ``__getitem__`` for the single map being
  fetched, so peak extra memory is one ``(H, W)`` float32 buffer (~1 MB) per
  worker.

The full noisy cube is never simultaneously resident.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _noise_sigma(ng: float, pixel_size_arcmin: float) -> float:
    """Per-pixel shape-noise sigma used by the challenge."""
    return 0.4 / (2.0 * ng * pixel_size_arcmin ** 2) ** 0.5


def add_noise_inplace(
    kappa_map: np.ndarray,
    mask: np.ndarray,
    ng: float = 30.0,
    pixel_size_arcmin: float = 2.0,
    rng: Optional[np.random.Generator] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Add Gaussian shape-noise to a single convergence map.

    Operates on one (H, W) map.  All temporary buffers are map-sized, so peak
    extra memory is ~``H*W*4`` bytes regardless of how many maps the caller
    will ultimately process.

    Parameters
    ----------
    kappa_map : (H, W) array
        Noiseless map (any float dtype; float16 is fine).
    mask : (H, W) bool/float array
        Survey mask — noise is zeroed outside.
    ng : float
        Galaxy number density (per arcmin^2).
    pixel_size_arcmin : float
        Pixel size in arcmin.
    rng : np.random.Generator or np.random.RandomState, optional
        Random source.  Pass ``np.random.RandomState(seed)`` to reproduce the
        challenge's legacy noise stream.
    out : (H, W) float32 array, optional
        Pre-allocated output buffer.  If None a new float32 array is made.
    """
    if rng is None:
        rng = np.random.default_rng()
    sigma = _noise_sigma(ng, pixel_size_arcmin)

    if out is None:
        out = np.empty(kappa_map.shape, dtype=np.float32)

    # noise ~ N(0, sigma * mask)
    noise = rng.standard_normal(kappa_map.shape).astype(np.float32, copy=False)
    noise *= sigma
    noise *= mask.astype(np.float32, copy=False)

    # out = kappa + noise   (upcast kappa from float16 lazily, one map at a time)
    np.add(kappa_map, noise, out=out, casting="unsafe")
    return out


# -----------------------------------------------------------------------------
# Lazy torch Dataset
# -----------------------------------------------------------------------------

class _LazyNoisyMapDataset(Dataset):
    """Wraps the compact masked-pixel kappa tensor and yields full noisy maps.

    Parameters
    ----------
    kappa_flat : (N, n_unmasked) np.ndarray (float16)
        Flattened (across the cosmology / nuisance axes) maps in compact form.
    mask : (H, W) np.ndarray (bool)
        Survey mask used to scatter back to a dense map.
    labels : (N, n_targets) np.ndarray (float32) or None
    ng, pixel_size_arcmin : noise parameters.
    noise_seed : int, optional
        If given, each index ``i`` uses a deterministic seed derived from
        ``noise_seed`` and ``i``.  This gives reproducible noise *without*
        having to pre-generate the full noisy cube.  If ``None`` a fresh
        random draw is used every epoch — the recommended training regime
        because it acts as data augmentation.
    """

    def __init__(
        self,
        kappa_flat: np.ndarray,
        mask: np.ndarray,
        labels: Optional[np.ndarray],
        ng: float,
        pixel_size_arcmin: float,
        noise_seed: Optional[int] = None,
        standardize: Optional[tuple[float, float]] = None,
    ):
        assert kappa_flat.ndim == 2, "expect compact (N, n_unmasked) form"
        self.kappa_flat = kappa_flat
        self.mask = mask.astype(bool, copy=False)
        self.labels = labels
        self.ng = ng
        self.pixel_size_arcmin = pixel_size_arcmin
        self.noise_seed = noise_seed
        self.standardize = standardize

        self._h, self._w = mask.shape
        # cache a float32 mask for the noise multiplication
        self._mask_f32 = self.mask.astype(np.float32)

    def __len__(self) -> int:
        return self.kappa_flat.shape[0]

    def _rng(self, idx: int):
        if self.noise_seed is None:
            return np.random.default_rng()
        # SeedSequence gives well-spaced independent streams per index.
        return np.random.default_rng(
            np.random.SeedSequence([self.noise_seed, idx])
        )

    def __getitem__(self, idx: int):
        # 1. scatter the compact row back to a dense (H, W) float32 map
        dense = np.zeros((self._h, self._w), dtype=np.float32)
        dense[self.mask] = self.kappa_flat[idx].astype(np.float32, copy=False)

        # 2. add noise in place (dense already float32, reused as out buffer)
        rng = self._rng(idx)
        sigma = _noise_sigma(self.ng, self.pixel_size_arcmin)
        noise = rng.standard_normal(dense.shape).astype(np.float32, copy=False)
        noise *= sigma
        noise *= self._mask_f32
        dense += noise

        if self.standardize is not None:
            mu, std = self.standardize
            dense = (dense - mu) / std

        x = torch.from_numpy(dense).unsqueeze(0)  # (1, H, W)
        if self.labels is None:
            return x
        y = torch.from_numpy(self.labels[idx])
        return x, y


class _DenseMapDataset(Dataset):
    """Test-set wrapper: maps are already noisy, just scatter + standardize."""

    def __init__(
        self,
        kappa_flat: np.ndarray,
        mask: np.ndarray,
        labels: Optional[np.ndarray] = None,
        standardize: Optional[tuple[float, float]] = None,
    ):
        self.kappa_flat = kappa_flat
        self.mask = mask.astype(bool, copy=False)
        self.labels = labels
        self.standardize = standardize
        self._h, self._w = mask.shape

    def __len__(self) -> int:
        return self.kappa_flat.shape[0]

    def __getitem__(self, idx: int):
        dense = np.zeros((self._h, self._w), dtype=np.float32)
        dense[self.mask] = self.kappa_flat[idx].astype(np.float32, copy=False)
        if self.standardize is not None:
            mu, std = self.standardize
            dense = (dense - mu) / std
        x = torch.from_numpy(dense).unsqueeze(0)
        if self.labels is None:
            return x
        y = torch.from_numpy(self.labels[idx])
        return x, y


# -----------------------------------------------------------------------------
# Public dataset class (follows the project convention)
# -----------------------------------------------------------------------------

class WeakLensingDataset:
    """FAIR-Universe weak-lensing challenge (Phase 1) — direct CNN regression.

    ``create()`` returns a dict with PyTorch ``Dataset`` objects rather than
    fully materialized tensors, because the full noisy training cube would
    otherwise use tens of GB of host memory.

    Configuration mirrors the notebook's ``Data`` helper.

    Parameters
    ----------
    data_dir : str
        Directory containing the .npy files released by the challenge.
    use_public_dataset : bool
        If True, loads the full 101x256 cube; otherwise loads the sampled
        3x30 cube shipped with the starting kit.
    n_targets : int
        Number of label columns to keep (default 2 = Omega_m, S_8).
    val_fraction : float
        Fraction of the *nuisance-parameter* axis to hold out for validation
        (recommended split by the challenge organizers).
    split_seed : int
        Seed for the train/val split.
    noise_seed : int or None
        If not None, noise is reproducible per sample-index (useful for eval).
        If None (default for training), a fresh noise draw is used every call
        — this acts as effective data augmentation.
    standardize : bool
        If True, computes per-pixel mean/std on a noiseless mini-sample of
        the training set and standardizes inputs.  Cheap and avoids holding
        the full noisy cube to compute statistics.
    """

    # The samples are 2D maps, so input_dim isn't really meaningful for a CNN
    # — it's reported here for compatibility with the rest of the project's
    # dataset interface (flattened map size).
    img_height = 1424
    img_width = 176
    in_chans = 1
    output_dim = 2  # (Omega_m, S_8)

    def __init__(
        self,
        data_dir: str,
        use_public_dataset: bool = True,
        n_targets: int = 2,
        val_fraction: float = 0.2,
        split_seed: int = 5566,
        noise_seed: Optional[int] = None,
        ng: float = 30.0,
        pixel_size_arcmin: float = 2.0,
        standardize: bool = True,
        **kwargs,
    ):
        self.data_dir = data_dir
        self.use_public_dataset = use_public_dataset
        self.n_targets = n_targets
        self.val_fraction = val_fraction
        self.split_seed = split_seed
        self.noise_seed = noise_seed
        self.ng = ng
        self.pixel_size_arcmin = pixel_size_arcmin
        self.standardize = standardize

        if use_public_dataset:
            self.kappa_file = "WIDE12H_bin2_2arcmin_kappa.npy"
            self.label_file = "label.npy"
            self.test_kappa_file = "WIDE12H_bin2_2arcmin_kappa_noisy_test.npy"
            self.Ncosmo, self.Nsys = 101, 256
        else:
            self.kappa_file = "sampled_WIDE12H_bin2_2arcmin_kappa.npy"
            self.label_file = "sampled_label.npy"
            self.test_kappa_file = "sampled_WIDE12H_bin2_2arcmin_kappa_noisy_test.npy"
            self.Ncosmo, self.Nsys = 3, 30
        self.mask_file = "WIDE12H_bin2_2arcmin_mask.npy"

    # ----- internals -----

    def _load_compact(self, fname: str) -> np.ndarray:
        # mmap so we do not pull the whole file into RAM before flattening
        return np.load(os.path.join(self.data_dir, fname), mmap_mode="r")

    def _estimate_stats(self, kappa_flat: np.ndarray, mask: np.ndarray,
                        n_sample: int = 256) -> tuple[float, float]:
        """Estimate input mean/std from a small noisy mini-sample.

        We sample ``n_sample`` rows uniformly, scatter + add one noise draw
        each, and aggregate.  Peak memory: ``n_sample * H * W * 4`` bytes
        for the sample buffer (~64 MB at n_sample=256), still tiny next to
        the full cube.
        """
        n = kappa_flat.shape[0]
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=min(n_sample, n), replace=False)
        sigma = _noise_sigma(self.ng, self.pixel_size_arcmin)
        mask_f32 = mask.astype(np.float32)

        sum_ = 0.0
        sum_sq = 0.0
        count = 0
        for i in idx:
            dense = np.zeros(mask.shape, dtype=np.float32)
            dense[mask] = kappa_flat[i].astype(np.float32, copy=False)
            noise = rng.standard_normal(mask.shape).astype(np.float32)
            noise *= sigma
            noise *= mask_f32
            dense += noise
            sum_ += float(dense.sum())
            sum_sq += float((dense * dense).sum())
            count += dense.size
        mean = sum_ / count
        var = sum_sq / count - mean * mean
        return mean, float(np.sqrt(max(var, 1e-12)))

    # ----- public API -----

    def create(self) -> dict:
        mask = np.load(os.path.join(self.data_dir, self.mask_file))
        mask = mask.astype(bool)

        # Compact form straight from disk: (Ncosmo*Nsys, n_unmasked) float16.
        # The released file is already stored in this compact shape.
        kappa_compact = self._load_compact(self.kappa_file)
        # Force into memory once as a contiguous float16 array — this is the
        # smallest dense representation we can keep around.
        kappa_compact = np.ascontiguousarray(kappa_compact, dtype=np.float16)
        # Reshape to (Ncosmo, Nsys, n_unmasked) so we can split along Nsys.
        n_unmasked = int(mask.sum())
        kappa_compact = kappa_compact.reshape(self.Ncosmo, self.Nsys, n_unmasked)

        labels = np.load(os.path.join(self.data_dir, self.label_file))
        labels = labels.astype(np.float32)

        # Split along the nuisance-parameter axis (challenge recommendation).
        rng = np.random.default_rng(self.split_seed)
        perm = rng.permutation(self.Nsys)
        n_val = int(round(self.Nsys * self.val_fraction))
        val_idx = np.sort(perm[:n_val])
        train_idx = np.sort(perm[n_val:])

        train_kappa = kappa_compact[:, train_idx].reshape(-1, n_unmasked)
        val_kappa = kappa_compact[:, val_idx].reshape(-1, n_unmasked)
        train_y = labels[:, train_idx].reshape(-1, labels.shape[-1])[:, : self.n_targets]
        val_y = labels[:, val_idx].reshape(-1, labels.shape[-1])[:, : self.n_targets]

        # Standardization stats from a noisy mini-sample of the training set.
        stats = self._estimate_stats(train_kappa, mask) if self.standardize else None

        train_ds = _LazyNoisyMapDataset(
            kappa_flat=train_kappa,
            mask=mask,
            labels=train_y,
            ng=self.ng,
            pixel_size_arcmin=self.pixel_size_arcmin,
            noise_seed=None,  # fresh noise each epoch -> data augmentation
            standardize=stats,
        )
        val_ds = _LazyNoisyMapDataset(
            kappa_flat=val_kappa,
            mask=mask,
            labels=val_y,
            ng=self.ng,
            pixel_size_arcmin=self.pixel_size_arcmin,
            noise_seed=31415,  # reproducible per-sample noise for validation
            standardize=stats,
        )

        # Test set is already noisy on disk — just scatter through the mask.
        test_compact = self._load_compact(self.test_kappa_file)
        test_compact = np.ascontiguousarray(test_compact, dtype=np.float16)
        if test_compact.ndim == 2:
            test_compact = test_compact.reshape(-1, n_unmasked)
        test_ds = _DenseMapDataset(
            kappa_flat=test_compact,
            mask=mask,
            labels=None,
            standardize=stats,
        )

        return {
            "train_input": train_ds,
            "train_label": train_y,
            "val_input": val_ds,
            "val_label": val_y,
            "test_input": test_ds,
            "test_label": None,
            "mask": mask,
            "input_stats": stats,
        }
