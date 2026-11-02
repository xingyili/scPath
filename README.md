# scPath: Pathway-informed harmonization of single-cell atlases across biological contexts

![scPath workflow: from reference and query cells through pathway-informed graph construction, fusion, alignment, and distillation to harmonized cell annotations](docs/scpath_workflow.png)

*scPath workflow.*

## Overview

scPath aligns single-cell RNA-seq atlases while preserving differences that matter biologically, including cell identities and states that vary across conditions. It combines gene-expression and pathway-activity representations, uses unbalanced optimal transport to align shared populations without forcing atlas-specific cells to match, and refines uncertain cell-type assignments through knowledge distillation. The result is a harmonized cell representation with transferred annotations that can be used to investigate pathway programs across platforms, tissues, disease, aging, and treatment.

This repository provides the scPath implementation, a CEL-Seq2-to-10x Chromium V2 example with its prepared input data, and one core notebook for each of the five manuscript studies. The larger study datasets are provided separately as five archives in the [Zenodo data record](https://doi.org/10.5281/zenodo.23851863).

## Quick start

The example trains a model, so you need a CUDA-capable NVIDIA GPU, a compatible driver, [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), and access to this private GitHub repository. Run the commands from a terminal on the GPU machine.

### 1. Download the code

```bash
git clone https://github.com/xingyili/scPath.git
cd scPath
```

GitHub will require authentication from an account that has been granted access to the repository. The example input files and human pathway GMT files are already included; no Zenodo download is needed for this quick start.

### 2. Create the environment

```bash
conda create -n scPath python=3.9 pip -y
conda activate scPath
```

Install the CUDA build of PyTorch 2.5.1 that matches your system, then the remaining dependencies. The command below uses **CUDA 11.8**, matching the local environment used to check this repository; use the corresponding command in the [PyTorch 2.5.1 installation guide](https://pytorch.org/get-started/previous-versions/) for CUDA 12.1 or 12.4. See the [environment notes](docs/ENVIRONMENT.md) for PyTorch Geometric compatibility and other requirements.

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install jupyterlab
python -m ipykernel install --user --name scPath --display-name "Python (scPath)"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
```

`CUDA available` must print `True` before you run the notebook. Installing a CPU-only PyTorch build will not run this training example.

### 3. Run the included example

Start Jupyter from the repository root so the notebook can locate the bundled data:

```bash
PYTHONHASHSEED=42 jupyter lab examples/cel_seq2/demo.ipynb
```

Select the **Python (scPath)** kernel and run the cells in order. The notebook prepares the reference and query data, builds the human Reactome and KEGG graphs, trains scPath, and prints accuracy and macro-F1 for the resulting predictions. On a remote GPU server, open the Jupyter URL or connect your editor to that server's Jupyter session.

After it finishes, inspect `output/cel_seq2/metrics.csv` and `output/cel_seq2/logs/test_results_distilled.h5ad`. Generated files stay under `output/`; the included inputs in `data/RNA/cel_seq2/` are not overwritten. Training time and metrics depend on the GPU and run conditions.

## Reproduce a manuscript study

The five study notebooks require substantially more GPU memory and time than the quick-start example. Download the relevant `.tar.zst` archive from the [Zenodo data record](https://zenodo.org/records/23851863), then extract it under `data/experiments/`. For example, for the cross-disease pancreas study:

```bash
mkdir -p data/experiments
tar --zstd -xf /path/to/cross_disease_pancreas.tar.zst -C data/experiments
PYTHONHASHSEED=42 jupyter lab experiments/cross_disease_pancreas/core_experiment.ipynb
```

Replace `/path/to/` with the location of the downloaded archive. Select the **Python (scPath)** kernel and run the notebook in order. Each of the other four notebooks has a matching archive; [data and provenance](docs/DATA_AND_PROVENANCE.md) lists the archive names and contents. The notebooks write new graphs, checkpoints, and predictions under `output/experiments/`. Set `SCPATH_DEVICE=cuda:0` before starting Jupyter if you need to select a particular visible GPU.

The PBMC archive stores each sequencing platform's input once. Before running its notebook, reconstruct the selected pair's `train/` and `test/` directories:

```bash
tar --zstd -xf /path/to/cross_platform_pbmc.tar.zst -C data/experiments
python scripts/materialize_pbmc_pairs.py --pair 10x_Chromium_V2_A_to_10x_Chromium_V3
PYTHONHASHSEED=42 jupyter lab experiments/cross_platform_pbmc/core_experiment.ipynb
```

Use `--all` instead of `--pair ...` only when you need all 72 directed PBMC pairs.

## Repository layout

| Path | What it contains |
| --- | --- |
| `scpath/` | Model, training, evaluation, and data-processing code |
| `examples/cel_seq2/` | Runnable small-data notebook and its preparation function |
| `data/RNA/cel_seq2/` | Prepared example reference and query inputs |
| `data/pathway/` | Human and mouse Reactome/KEGG GMT files |
| `experiments/` | One core notebook per manuscript study |
| `scripts/materialize_pbmc_pairs.py` | Reconstructs a PBMC train/query pair from the archived platform inputs |
| `docs/` | Environment, data provenance, and the workflow image |

The full study RNA matrices and clinical PDAC inputs are not stored in this GitHub repository. See the [Zenodo data record](https://zenodo.org/records/23851863) for the study archives and [CITATION.cff](CITATION.cff) for citation information.
