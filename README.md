# scPath

scPath is a pathway-informed framework for harmonizing single-cell RNA-seq datasets across technical and biological contexts. The implementation combines expression and pathway-derived cell graphs, graph neural networks, unbalanced optimal transport, and confidence-aware student distillation.

This repository contains the core implementation, a small runnable CEL-Seq2-to-10x Chromium V2 example, and one core training notebook for each of the five manuscript studies. The study notebooks retain the source training parameters and the applicable metric calculations. Their data are distributed separately as five [Zenodo archives](https://doi.org/10.5281/zenodo.23851863).

## Install

Use Python 3.9. Install a PyTorch build matching the CUDA driver and the corresponding PyTorch Geometric dependencies first. Then install the remaining requirements and this package:

```bash
conda create -n scPath python=3.9 pip -y
conda activate scPath
# Install a CUDA-compatible PyTorch build and matching PyG packages first.
python -m pip install -r requirements.txt
python -m pip install -e .
python -m ipykernel install --user --name scPath --display-name "Python (scPath)"
```

Check that the selected environment sees the intended GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The example is a training workload and requires a CUDA GPU with enough memory for the selected reference/query pair. See [environment notes](docs/ENVIRONMENT.md).

## Run the example

Open `examples/cel_seq2/demo.ipynb` with the `scPath` kernel and run its cells in order. The included `data/RNA/cel_seq2/` directory contains the prepared reference and query files. The example uses the original human Reactome and KEGG GMT files in `data/pathway/human/` and writes generated graphs and model outputs under `output/cel_seq2/`.

The companion `examples/cel_seq2/process_data.py` contains the corresponding data preparation calls. The notebook reports measured metrics from the local example run.

## Run a manuscript experiment

Download the matching archive from the [public Zenodo data record](https://zenodo.org/records/23851863). Extract it into `data/experiments/` at the repository root, for example:

```bash
mkdir -p data/experiments
tar --zstd -xf cross_disease_pancreas.tar.zst -C data/experiments
```

Open `experiments/cross_disease_pancreas/core_experiment.ipynb` with the `scPath` kernel and execute its cells in order. The other study notebooks use their matching archives; PBMC requires the additional preparation step below. Set `PYTHONHASHSEED=42` before launching Jupyter and optionally `SCPATH_DEVICE=cuda:0` to select a visible GPU. New graphs, checkpoints, and predictions are written under `output/experiments/`. The experiment notebooks preserve the source training scale, so some runs need substantial GPU time and memory. See [data provenance](docs/DATA_AND_PROVENANCE.md) for archive names and experiment-specific inputs.

The PBMC archive stores each platform input once. To run the pair used by
`experiments/cross_platform_pbmc/core_experiment.ipynb`:

```bash
mkdir -p data/experiments
tar --zstd -xf cross_platform_pbmc.tar.zst -C data/experiments
python scripts/materialize_pbmc_pairs.py --pair 10x_Chromium_V2_A_to_10x_Chromium_V3
```

The script recreates the notebook's `train/` and `test/` input directories. Use `--all` only when all 72 pairs are needed.

## Repository contents

| Path | Contents |
| --- | --- |
| `scpath/` | Core source modules and required utilities |
| `examples/cel_seq2/` | Runnable example notebook and data preparation script |
| `experiments/` | Five source-aligned core study notebooks |
| `scripts/materialize_pbmc_pairs.py` | Rebuild PBMC pair directories from deduplicated inputs |
| `data/RNA/cel_seq2/` | Prepared small reference/query example |
| `data/pathway/` | Human and mouse Reactome/KEGG pathway GMT files |
| `docs/` | Environment and data provenance |

The repository does not include the five studies' RNA matrices, clinical PDAC data, intermediate files, or checkpoints. Selected study inputs and outputs are organized by experiment in the [Zenodo data record](https://zenodo.org/records/23851863); see [the data and provenance note](docs/DATA_AND_PROVENANCE.md).

## Citation

Please cite the scPath manuscript and the versioned Zenodo data record. See [CITATION.cff](CITATION.cff).
