# Data and Provenance

The GitHub package includes one prepared PBMC example, four pathway GMT files,
and five core study notebooks. The five study data archives are provided as
restricted Zenodo files, one per study. They contain original selected inputs
and historical data outputs, but no notebooks or model source. Files retain
original contents and are not recomputed by the archive build.

## Study lineages

| Study / archive | Original processing and experiment code | Input to core notebook | Historical output retained |
| --- | --- | --- | --- |
| Cross-platform PBMC / `cross_platform_pbmc.tar.zst` | `data/RNA/6_PBMC/data_solve*.ipynb`, `data_for_PML.ipynb`; `benchmark/cross_platform(small)/batch_batch/platform copy.ipynb` and `platform.ipynb` | `cross_platform_pbmc/original_paths/data/pairs/10x_Chromium_V2_A_to_10x_Chromium_V3/{train,test}` | Per-pair prediction AnnData files and metric tables |
| Cross-tissue / `cross_tissue.tar.zst` | `benchmark/cross_tissue/experiment/exp4_findnew/exp4_findnew.ipynb`, `exp4_final.ipynb` | `cross_tissue/original_paths/data/Tabula_Muris_{10x,smartseq}_Combined` | Prediction AnnData files and comparison metrics |
| Cross-disease / `cross_disease_pancreas.tar.zst` | `benchmark/cross_disease_states/mouse_pancreas/data_solve.ipynb`, `exp/train_mouse_pancreas.ipynb` | `cross_disease/original_paths/data/{normal,test_combined}`; parent atlas under `cross_disease/data/source/` | Prediction AnnData files and result tables |
| Bone marrow / `cross_time_bone_marrow.tar.zst` | `benchmark/cross_time/exp_TMS/train_bone_marrow.ipynb` | `cross_time_bone_marrow/original_paths/data/{20-month-old stage and over,test_combined}` | Prediction AnnData files and analysis tables |
| PDAC/NACT / `pdac_nact.tar.zst` | `benchmark/NACT/exp1_t2.ipynb`, `exp2_all.ipynb`, downstream analysis notebooks | `pdac_nact/original_paths/data/10x_export/{t2,t3}`; raw T2/T3/T338 matrices and annotations also retained | Prediction AnnData files and analysis tables |

The experiment archives use files selected from the original directories,
not new downloads or regenerated matrices. Their per-file manifests map each
archive member to the original path. Large model checkpoints, regenerable
graphs, redundant trial outputs, and raw sequencing reads are omitted unless
listed in an archive manifest. The PBMC core notebook runs one historical
pair, not the full 72-pair benchmark. The CEL-Seq2 quick start is a separate
small pair.

## Included example

`data/RNA/cel_seq2/` is the prepared CEL-Seq2 reference and 10x Chromium V2
query pair already used in the original PBMC benchmark. The eight data files
were copied from the prepared pair; the matching original pair was verified to
have identical file contents. `data/pathway/{human,mouse}/` contains the four
pathway GMT files from the source project.

For full upstream provenance, see each archive's `DATA_SOURCE.md` and
`MANIFEST.csv`. The original research notebooks remain in the source project
but are intentionally not distributed in the new data-only Zenodo version.
The archives do not claim that raw FASTQ or every public parent atlas is
included. The PDAC archive contains in-house human data and requires owner
approval for access; approval does not grant redistribution rights.

The original PDAC T3 query has only an unseen/private placeholder in the
model's `labels` column and `unknown` in `cell_type`. Accuracy and macro-F1
against that placeholder are not meaningful. The PDAC core notebook reports
the predicted-class distribution; local release validation compares its
predictions with the historical run on the same cells.
