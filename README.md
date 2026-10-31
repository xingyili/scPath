# scPath

scPath is a pathway-informed framework for harmonizing single-cell RNA-seq
datasets across technical and biological contexts. The implementation combines
expression and pathway-derived cell graphs, graph neural networks, unbalanced
optimal transport, and confidence-aware student distillation.

This repository contains the translated core implementation, a small runnable
CEL-Seq2-to-10x Chromium V2 example, and one core training notebook for each
of the five manuscript studies. The original `MPNN_SelfDistill` source and
this package use the historical GRU gate and batch-wise class-weighted
classification loss. Executable model logic matches the source after
package-import and English text adaptations. The study notebooks retain the
source training parameters and the applicable metric calculations;
figure-generation cells are omitted. Their data are distributed separately
as five restricted Zenodo archives, not in GitHub.

## Install

Use Python 3.9. Install a PyTorch build matching the CUDA driver and the
corresponding PyTorch Geometric dependencies first. Then install the remaining
requirements and this package:

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

The example is a training workload and requires a CUDA GPU with enough memory
for the selected reference/query pair. See [environment notes](docs/ENVIRONMENT.md).
For source-parity evidence, see [the parity note](docs/SOURCE_PARITY.md).

## Run the example

Open `examples/cel_seq2/demo.ipynb` with the `scPath` kernel and run its cells
in order. The included `data/RNA/cel_seq2/` directory contains the prepared
reference and query files. The example uses the original human Reactome and
KEGG GMT files in `data/pathway/human/` and writes generated graphs and model
outputs under `output/cel_seq2/`.

The companion `examples/cel_seq2/process_data.py` contains the corresponding
data preparation calls. The notebook reports measured metrics from the local
example run; they are example-specific and are not a guarantee of the scores
in the manuscript's 72-pair benchmark.

The recorded RTX 4090 run with the restored GRU/class-weight configuration
produced 94.77% accuracy / 0.9011 macro-F1 for OT common labels and 94.96%
accuracy / 0.9133 macro-F1 after distillation. The executed notebook contains
the associated output. These are example-specific metrics, not the five-study
reproduction scores.

## Run a manuscript experiment

Obtain access to the matching restricted Zenodo data archive. Extract it into
`data/experiments/` at the repository root, for example:

```bash
mkdir -p data/experiments
tar --zstd -xf cross_disease_pancreas.tar.zst -C data/experiments
```

Open `experiments/cross_disease_pancreas/core_experiment.ipynb` with the
`scPath` kernel and execute its cells in order. The same layout applies to
the four other experiments. Set `PYTHONHASHSEED=42` before launching Jupyter
and optionally `SCPATH_DEVICE=cuda:0` to select a visible GPU. New graphs,
checkpoints, and predictions are written under `output/experiments/`; the
historical files in the archive are read-only comparison artifacts. The
experiment notebooks preserve the source training scale, so some runs need
substantial GPU time and memory. See [data provenance](docs/DATA_AND_PROVENANCE.md)
for archive names and experiment-specific inputs.
These notebooks reproduce the core model run and its primary prediction
summary; they do not regenerate every downstream biological analysis table
retained as historical data in Zenodo.

## Repository contents

| Path | Contents |
| --- | --- |
| `scpath/` | Core source modules and required utilities |
| `examples/cel_seq2/` | Runnable example notebook and data preparation script |
| `experiments/` | Five source-aligned core study notebooks |
| `data/RNA/cel_seq2/` | Prepared small reference/query example |
| `data/pathway/` | Human and mouse Reactome/KEGG pathway GMT files |
| `docs/` | Environment, data provenance, source parity, and access notes |

The repository does not include the five studies' RNA matrices, clinical
PDAC data, intermediate files, or checkpoints. Selected original inputs and
historical data outputs are organized by experiment in the associated
restricted Zenodo deposit; see
[the data and provenance note](docs/DATA_AND_PROVENANCE.md).

## Access and rights

This is a private repository. Access is granted by repository invitation;
GitHub does not provide a public request-access workflow for private
repositories. The software and bundled data are all rights reserved. Inclusion
of upstream pathway or example data does not supersede the original sources'
terms. Access approval alone does not grant permission to redistribute,
modify, or reuse the files. See [access and publishing notes](docs/PUBLISHING.md).

## Citation

Please cite the scPath manuscript and the versioned restricted Zenodo data
record. See [CITATION.cff](CITATION.cff) and the access notes.
