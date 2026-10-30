"""Prepare the original CEL-Seq2 reference/query example without altering QC."""

from pathlib import Path
import random
import numpy as np
import torch
from scpath.utils.data_preprocess import data_processing, preprocess
from scpath.utils.pathway import built_pathway_graphs


def prepare(project_root=None):
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    data = root / "data/RNA/cel_seq2"
    output = root / "output/cel_seq2"
    graphs = output / "output"
    graphs.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    reference, query = data_processing(
        str(data / "train") + "/", str(data / "test") + "/", str(graphs) + "/",
        "human", "human", False, False,
    )
    pathways = ["Reactome", "KEGG"]
    built_pathway_graphs(str(root / "data/pathway/human") + "/", pathways,
                         reference, query, str(graphs) + "/", n_neighbors=5)
    for split in ["train", "test"]:
        for pathway in pathways:
            if not (graphs / f"{split}_knn_graph_{pathway}.npz").is_file():
                raise FileNotFoundError(f"Missing {split} {pathway} graph")
    reference, hvg = preprocess(reference)
    query = preprocess(query, hvg_genes=hvg)
    return reference, query, pathways, output
