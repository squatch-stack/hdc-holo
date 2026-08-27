"""Shrinkage denoising for forward-encoded bundles.

A bundle built by *accumulating* splats carries crosstalk that a bundle
*fitted* to samples would not: forward encoding has no mechanism to
cancel the interference between superposed items, it simply adds them
up. More dimension does not remove it — doubling the fine band's `d`
bought 2-4% for +600 MB, because in dense scenes the residual is
coherent rather than Monte-Carlo (`docs/spatial.md`).

The opening came from the codec measurements. Truncating a forward
bundle's small components — which max-scaled 4-bit quantization does by
accident, as a side effect of spending its levels on the magnitude
distribution's head — did not just cost less, it scored BETTER against
ground truth than the uncompressed decode. That is the signature of a
denoiser, arrived at unintentionally, and it says the small components
carry proportionally more crosstalk than signal.

This module does it on purpose, and does better, because thresholding
is a solved problem in another field. Wavelet shrinkage (Donoho and
Johnstone) sets a threshold at the noise level and either zeroes what
falls below it (*hard*) or pulls every coefficient toward zero by that
amount (*soft*).

Which of the two wins is REGIME-DEPENDENT, and the distinction matters
enough to be pinned by test. Where the signal is sparse and strong and
the noise is weak, there is a real gap to cut at and hard wins — soft's
shrinkage of the surviving components is pure bias. Where magnitudes
OVERLAP, so that components near the threshold are mixtures of signal
and crosstalk, hard discards the signal along with the noise and soft
wins. Capture bundles are firmly in the second regime, because a
mixture codebook weights frequencies unequally and the magnitude
distribution is continuous with no gap in it. Measured on the saguaro
capture at the 25th percentile: soft reaches 0.298/0.203 against hard's
0.344/0.209, from an unshrunk 0.350/0.213. Soft is therefore the
default here — but it is not universally better, and at higher
thresholds hard overtakes it again.

Complex components are shrunk **in magnitude with phase preserved** —
the standard complex extension, and the only choice that keeps the
result a bundle of the same kind.

**Choosing the threshold.** The capacity law gives a readout-space noise
scale `sigma ~ sqrt(N_local / 2d)`, but that is not a per-component
scale: every component of a forward bundle is a sum of the same
`N_local` phasors, so under the i.i.d. model they share one magnitude
distribution and no component is a priori noisier than another. What
actually varies is how much SIGNAL a component carries, because a
mixture codebook weights frequencies unequally and a frequency outside
the local scene's spectral support contributes little but crosstalk.
So the useful threshold is a quantile of the observed magnitudes, not a
constant derived from the law — `percentile_threshold` computes it per
cell and per channel.

**Failure mode.** Shrinkage trades the two slice axes against each
other past its optimum: on the saguaro capture the top-down slice keeps
improving out to the 60th percentile while the side slice starts
degrading after the 25th. Sweep against YOUR evidence slices rather
than inheriting a percentile from this docstring, and prefer the
setting that improves every axis to the one that improves the headline.
"""

import numpy as np

__all__ = ["percentile_threshold", "shrink"]


def percentile_threshold(v, pct):
    """Per-row magnitude percentile of `v`, shaped to broadcast back.

    `v` is (d,) or (channels, d) — a cell bundle's channels are
    thresholded independently, since a premultiplied alpha channel and
    a colour channel do not share a magnitude scale.
    """
    v = np.asarray(v)
    if v.ndim == 1:
        return np.percentile(np.abs(v), pct)
    return np.percentile(np.abs(v), pct, axis=-1, keepdims=True)


def shrink(v, threshold, mode="soft"):
    """Shrink `v`'s components toward zero at `threshold`.

    `threshold` is a magnitude, broadcast against `v` — pass a scalar,
    or the output of `percentile_threshold`. Returns complex64.

        soft:  |z| -> max(|z| - t, 0), phase kept   (less biased)
        hard:  |z| -> 0 where |z| < t, else kept

    Soft is the default because capture bundles sit in the overlapping
    regime where it wins (see the module docstring); it is not
    universally better and hard overtakes it at higher thresholds.
    """
    if mode not in ("soft", "hard"):
        raise ValueError("mode must be 'soft' or 'hard', not %r" % (mode,))
    v = np.asarray(v)
    mag = np.abs(v)
    if mode == "hard":
        return np.where(mag < threshold, 0, v).astype(np.complex64)
    # scale rather than rebuild from angle: this keeps exact zeros exact
    # and avoids an arctan2/exp round trip per component
    keep = np.maximum(mag - threshold, 0.0) / np.maximum(mag, 1e-30)
    return (v * keep).astype(np.complex64)
