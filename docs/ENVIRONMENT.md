# Environment

The packaged modules were prepared for the source project's Python 3.9 stack.
The demo and five full experiment notebooks require CUDA; CPU-only
installation is not a supported configuration for these training workflows.

Create an isolated Conda environment named `scPath`, install a PyTorch build
that matches the local CUDA runtime, and install the matching PyTorch Geometric
packages before the pinned requirements:

```bash
conda create -n scPath python=3.9 pip -y
conda activate scPath
# Follow the official PyTorch and PyG install instructions for your CUDA build.
python -m pip install -r requirements.txt
python -m pip install -e .
```

The dependency pins record the versions used in the source workstation, but
the generic `torch` wheel may not select the CUDA build desired by every
system. Install and verify PyTorch/PyG first; then install the other pinned
packages. Confirm availability with:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The notebooks use `SCPATH_DEVICE` to choose a visible CUDA device (default
`cuda:0`). This changes hardware placement, not the model or training
hyperparameters. Before starting Jupyter, set `PYTHONHASHSEED=42`; the source
preprocessing uses set-based gene intersections, so changing hash order can
change feature order and subsequent random-number consumption. Full study
runs require substantially more memory and time than the CEL-Seq2 quick start.
