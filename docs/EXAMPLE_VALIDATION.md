# CEL-Seq2 Example Validation

The packaged `examples/cel_seq2/demo.ipynb` was executed end-to-end on an
NVIDIA GeForce RTX 4090 with Python 3.9 and PyTorch 2.5.1. CUDA was available
to the runtime. No dependency installation or change to the existing Conda
environments was performed as part of this validation.

Data processing completed with 463 reference cells and 3,098 query cells, each
with 4,000 selected genes. The example generated Reactome and KEGG graphs,
completed the configured 500-step fusion stage, and completed 200 student
epochs. The resulting metrics were:

| Stage | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| OT common labels | 93.35% | 0.7654 |
| Final distilled labels | 95.13% | 0.7469 |

Accuracy increased after distillation, while macro-F1 decreased. Both values
are kept in the report because they measure different class behavior. The run
outputs are embedded in the notebook. Generated H5AD files, model weights,
graphs, TensorBoard logs, and the metrics CSV are local outputs and are not
tracked in the source repository.
