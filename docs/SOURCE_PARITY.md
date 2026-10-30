# Source Parity

The translated package includes the original active core modules:

`models.py`, `config.py`, `train.py`, `fusion_train.py`, `distill_train.py`,
`evaluate.py`, `plot_res.py`, `utils/data_preprocess.py`,
`utils/data_prepare.py`, `utils/model_utils.py`, `utils/lib.py`, and
`utils/pathway.py`.

The original and packaged Python ASTs were compared after ignoring comments,
docstrings, standalone Chinese prose expressions, Chinese user-facing text,
and the necessary `MPNN_SelfDistill` to `scpath` import-root change. Other
string constants and executable statements were retained. The only remaining
non-message literal difference is a translated Chinese full stop in a print
message.

A CUDA forward check compared original and packaged GAT, GCN, and GraphSAGE
models. Initialization state was bitwise identical. Source and target forward
outputs matched with maximum absolute error below `1e-6` in each configuration.
The end-to-end CEL-Seq2 example also completed through the packaged imports;
see [example validation](EXAMPLE_VALIDATION.md).

The original `MPNN_SelfDistill` files were read-only throughout release
preparation. The exact original core files are also preserved in the shared
Zenodo provenance archive.
