# scPath

scPath is a pathway-informed framework for harmonizing single-cell RNA-seq
datasets across technical and biological contexts. The implementation combines
expression and pathway-derived cell graphs, graph neural networks, unbalanced
optimal transport, and confidence-aware student distillation.

This repository contains the translated core implementation and one small
CEL-Seq2-to-10x Chromium V2 example. The model and training calculations are
carried over from the original `MPNN_SelfDistill` source; only comments,
docstrings, user-facing Chinese text, and package import paths were changed.
The five manuscript studies are preserved separately as original research
archives in the restricted Zenodo record. Their saved outputs are historical
artifacts, not results regenerated for this software release.

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

The recorded RTX 4090 run produced 93.35% accuracy / 0.7654 macro-F1 for OT
common labels and 95.13% accuracy / 0.7469 macro-F1 after distillation. The
final accuracy improved, while macro-F1 was lower than the OT-common score;
both metrics are retained so the example is not presented using accuracy
alone. The executed notebook contains the associated output.

## Repository contents

| Path | Contents |
| --- | --- |
| `scpath/` | Core source modules and required utilities |
| `examples/cel_seq2/` | Runnable example notebook and data preparation script |
| `data/RNA/cel_seq2/` | Prepared small reference/query example |
| `data/pathway/` | Human and mouse Reactome/KEGG pathway GMT files |
| `docs/` | Environment, provenance, validation, and access notes |

The repository does not include the five studies' RNA matrices, clinical
PDAC data, intermediate files, or checkpoints. Those selected originals are
organized by experiment in the associated restricted Zenodo deposit; see
[the data and provenance note](docs/DATA_AND_PROVENANCE.md).

## Access and rights

This is a private repository. Access is granted by repository invitation;
GitHub does not provide a public request-access workflow for private
repositories. The software and bundled data are all rights reserved. Inclusion
of upstream pathway or example data does not supersede the original sources'
terms. Access approval alone does not grant permission to redistribute,
modify, or reuse the files. See [access and publishing notes](docs/PUBLISHING.md).

## Citation

Please cite the scPath manuscript, “Pathway-informed harmonization of
single-cell atlases across biological contexts,” and cite the Zenodo record
after its DOI is available. See [CITATION.cff](CITATION.cff).
