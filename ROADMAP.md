# Roadmap

How work gets picked here: items below are claimed in `SDK.md`'s
running log before anyone starts (see `CONTRIBUTING.md` for the
working agreements). Findings — positive or negative — land in the log
and in `docs/`, with figures. Nothing unproven enters the SDK surface.

## Shipped

- **0.1** — the charter executed end to end: FHRR core + data
  structures, splat fields (FPE / spectral / bands / cells / attributes
  / color), learning (ridge-fit holograms), closed-form X-ray
  rendering, CRDT replication with observed-remove deletion and live
  TCP sync, tagged wire/storage formats, GPU backend through every
  eval, docs-per-technique, gated CI. Tagged `v0.1.0`.
- **0.2 (in progress, findings in SDK.md log)** — codec
  rate-distortion (HP/HM/HG rules, measured on real captures),
  real-scene turntable, per-cell fitting of real scenes (spectral
  prior; honest sampling limit), Fourier-extension placement of the
  analytic-projection direction, fine-band reach split promoted to
  the capture default (reach follows the band cap; 33-44% slice-error
  cut on the saguaro at 1.5x storage — raising d instead bought 2-4%:
  dense-scene residual is coherent, not Monte-Carlo).

## 0.3 candidates (unclaimed unless noted)

- **Principled shrinkage denoiser** — HM-4's accidental truncation
  beat uncompressed decode on ground truth; deliberate soft/hard
  thresholding at the crosstalk noise level should do better, and
  composes with HG-8 for a denoise-then-persist pair.
- **Analytic L2 projection for per-cell fits** — the Fourier-extension
  route: closed-form region Gram, Tikhonov/TSVD from day one, spectral
  prior as frame regularizer. (CLAIMED: capture/spectral lane — "peer"
  is ambiguous in a shared doc; lanes are named by module ownership.)
- **Dense-scene coherent error** — what remains of train's top-down
  figure after the reach split (1.04 -> 0.98): not Monte-Carlo (d-boost
  measured ineffective), so it needs a different idea — connects to the
  shrinkage-denoiser item above.
- **Real-scene collaborative editing** — live_sync's OR strokes over
  capture-scale scenes; magnitude codecs for epoch blobs on the wire.
- **Occlusion research spike** — alpha compositing is outside linear
  superposition (documented failure mode); scope what a hybrid
  (holographic density + classical compositing pass) would look like.
- **Publication pass** — arXiv note on the strongest findings (codec
  split + accidental denoiser; observed-remove holographic CRDTs; the
  per-cell fit negative result and its Fourier-extension framing),
  then PyPI release. LICENSE: FSL-1.1-Apache-2.0, chosen and in-tree.
  Remaining blocker: flipping the repo public (user step).
- **Package name decision** — `holo` is the charter's working name;
  rename happens once, before PyPI.

## Support this work

Donation/sponsorship hooks are being set up (`.github/FUNDING.yml`
lands once accounts exist). If the roadmap above is useful to you —
research, robotics mapping, collaborative 3D, edge/analog hardware —
sponsoring specific items or funding issues is the most direct signal
for what gets built next.
