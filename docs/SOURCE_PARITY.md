# Source Parity

The translated package includes the original active core modules:

`models.py`, `config.py`, `train.py`, `fusion_train.py`, `distill_train.py`,
`evaluate.py`, `plot_res.py`, `utils/data_preprocess.py`,
`utils/data_prepare.py`, `utils/model_utils.py`, `utils/lib.py`, and
`utils/pathway.py`.

The source project was restored to its historical GRU gate and batch-wise
class-weighted classification loss before the current package was aligned.
The original and packaged Python ASTs were compared after ignoring comments,
docstrings, standalone Chinese prose expressions, Chinese user-facing text,
and the necessary `MPNN_SelfDistill` to `scpath` import-root change. Other
string constants and executable statements were retained. The `min_cells=4`
gene filter now also matches the source; an earlier package had used 10.

A CUDA forward check compared original and packaged GAT, GCN, and GraphSAGE
models. Initialization state was bitwise identical. Source and target forward
outputs matched with maximum absolute error below `1e-6` in each configuration.
The end-to-end CEL-Seq2 example completed through the packaged imports on a
CUDA GPU with the restored configuration. Its recorded metrics appear in
`examples/cel_seq2/demo.ipynb`.

The source project changed only in the two authorized historical restorations:
`models.py` (GRU gate) and `fusion_train.py` (class-weighted loss). The five
data-only Zenodo archives do not contain Python model modules or notebooks.
