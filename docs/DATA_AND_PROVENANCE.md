# Data and Provenance

## Included in the repository

The `data/RNA/cel_seq2/` directory contains the prepared CEL-Seq2 reference and 10x Chromium V2 query used by the runnable example in `examples/cel_seq2/demo.ipynb`. The `data/pathway/` directory contains human and mouse Reactome and KEGG pathway files in GMT format.

## Study data archives

The five study datasets are available as separate archives in the [Zenodo data record](https://zenodo.org/records/23851863) (DOI: [10.5281/zenodo.23851863](https://doi.org/10.5281/zenodo.23851863)). Each archive contains the inputs used by its corresponding notebook in `experiments/`, selected saved results, a `DATA_SOURCE.md` describing the dataset, and a `MANIFEST.csv` listing the packaged files and their checksums.

| Study | Archive | Packaged inputs | Packaged results |
| --- | --- | --- | --- |
| Cross-platform PBMC | `cross_platform_pbmc.tar.zst` | Nine deduplicated platform inputs (matrix, barcodes, features, and metadata) and an index of 72 directed pairs | One final prediction AnnData file per pair |
| Cross-tissue | `cross_tissue.tar.zst` | Prepared Tabula Muris 10x and Smart-seq matrices and metadata | Prediction AnnData files |
| Cross-disease pancreas | `cross_disease_pancreas.tar.zst` | Mouse pancreas source AnnData, normal reference, and combined disease query | Prediction AnnData files and downstream analysis tables |
| Bone-marrow aging | `cross_time_bone_marrow.tar.zst` | Prepared older reference and younger query matrices and metadata | Prediction AnnData files and analysis tables |
| PDAC/NACT | `pdac_nact.tar.zst` | Processed T2, T3, and T338 count matrices, cell annotations, and prepared 10x exports | Prediction AnnData files and selected differential-expression, pathway, and cell-communication tables |

To run a study notebook, extract its archive into `data/experiments/` at the repository root, then open the matching `experiments/*/core_experiment.ipynb`.
