# Release Status

This candidate is assembled from the original `MPNN_SelfDistill` project.
The original project directory is not modified by the release build.

- The core model and training modules were compared to the original source at
  the Python AST level. Executable statements match after disregarding
  comments, docstrings, translated Chinese-only messages, and package import
  roots. Non-Chinese literal constants are retained.
- Original experiment notebooks with saved outputs are archived separately;
  they are not rewritten for this software repository.
- The included CEL-Seq2-to-10x Chromium V2 example completed on an RTX 4090
  using Python 3.9 / PyTorch 2.5.1. It reported 93.35% OT-common accuracy
  (0.7654 macro-F1) and 95.13% final distilled accuracy (0.7469 macro-F1).
- The executed notebook output has local home-directory and Conda paths
  removed; the selected notebook kernel is named `scPath`.
- The GitHub target is intended to be private. The Zenodo record is intended
  to have public metadata and restricted files.
- Software and data are all rights reserved. No open-source license is
  granted.

This status is updated after the example test and remote upload checks finish.
