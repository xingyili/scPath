# Environment

The packaged modules were prepared for the source project's Python 3.9 stack.
The demo requires CUDA; CPU-only installation is not a supported quick-start
configuration for the full training workflow.

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

The notebook uses a CUDA device selected in its configuration cell. Change
that device index only to one visible on the local machine; the training
algorithm and hyperparameters should otherwise remain unchanged when comparing
against the example run.
