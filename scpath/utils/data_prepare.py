from scipy.sparse import load_npz
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import from_scipy_sparse_matrix
import torch
import numpy as np

def prepare_data(path_output, adata_train, adata_test, pathway_names):
    train_cell_types = set(adata_train.obs['cell_type'].unique())
    test_cell_types = set(adata_test.obs['cell_type'].unique())
    common_types = sorted(list(train_cell_types & test_cell_types))
    train_only_types = sorted(list(train_cell_types - test_cell_types))
    test_only_types = sorted(list(test_cell_types - train_cell_types))
    label_mapping = {}
    current_label = 0
    for cell_type in common_types:
        label_mapping[cell_type] = current_label
        current_label += 1
    for cell_type in train_only_types:
        label_mapping[cell_type] = current_label
        current_label += 1
    for cell_type in test_only_types:
        label_mapping[cell_type] = current_label
        current_label += 1
    adata_train.obs['labels'] = adata_train.obs['cell_type'].map(label_mapping)
    adata_test.obs['labels'] = adata_test.obs['cell_type'].map(label_mapping)
    print(f'Shared classes: {len(common_types)}')
    print(f'Reference-only classes: {len(train_only_types)}')
    print(f'Query-only classes: {len(test_only_types)}')
    label_df = pd.DataFrame({'cell_type': list(label_mapping.keys()), 'label': list(label_mapping.values())}).sort_values('label')
    knn_graphs = {}
    for pathway in pathway_names:
        for dataset in ['train', 'test']:
            file_path = f'{path_output}{dataset}_knn_graph_{pathway}.npz'
            knn_graphs[f'{dataset}_{pathway}'] = load_npz(file_path)
    for adata in [adata_train, adata_test]:
        adata.obs['int_id'] = np.arange(len(adata), dtype=np.int64)
        assert adata.obs['int_id'].max() < len(adata), 'Failed to generate int_id values!'
    return (adata_train, adata_test, label_df, knn_graphs)
import scipy.sparse

def prepare_graph_data(adata, graph_matrix, device):
    (edge_index, _) = from_scipy_sparse_matrix(graph_matrix)
    if scipy.sparse.issparse(adata.X):
        feature_x = adata.X.toarray()
    else:
        feature_x = adata.X
    return (torch.FloatTensor(feature_x).to(device), edge_index.to(device))
