# CEL-Seq2 Example Data

This directory contains the prepared PBMC reference/query pair used by the
`examples/cel_seq2/demo.ipynb` workflow:

- `train/`: CEL-Seq2 PBMC reference
- `test/`: 10x Chromium V2 PBMC query
- Each split contains `matrix.mtx`, `features.tsv`, `barcodes.tsv`, and
  `metadata.csv`.

The files were copied from the original prepared benchmark pair. They are
included for the private project example only; upstream source terms still
apply. The corresponding source pair was checked for byte-identical files.
