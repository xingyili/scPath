import os
import decoupler as dc
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, save_npz, load_npz

def read_gmt(file_path, lowercase_genes=True):
    pathways = []
    gene_sets = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            pathway_name = parts[0]
            genes = parts[2:]
            if lowercase_genes:
                genes = [gene.lower() for gene in genes]
            pathways.append(pathway_name)
            gene_sets[pathway_name] = genes
    return (pathways, gene_sets)

def compute_and_save_knn(adata_subset, path_output, pathway, dataset_name, n_neighbors):
    n_neighbors = 5
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(adata_subset.obsm[f'{pathway}_pathway_activity'])
    knn_graph = knn.kneighbors_graph(adata_subset.obsm[f'{pathway}_pathway_activity'], mode='connectivity', n_neighbors=n_neighbors)
    file_path = os.path.join(path_output, f'{dataset_name}_knn_graph_{pathway}.npz')
    save_npz(file_path, knn_graph)
    print(f'KNN graph saved to {file_path}.')

def built_pathway_graphs(path, pathway_names, adata_train_rna=None, adata_test_rna=None, path_output=None, n_neighbors=5, save=True):
    """Compute the original AUCell activities and pathway-specific cell graphs."""
    if adata_train_rna is None and adata_test_rna is None:
        raise ValueError('Provide at least one of adata_train_rna or adata_test_rna')
    pathway_activities = []
    if path_output is not None:
        os.makedirs(path_output, exist_ok=True)
    for pathway in pathway_names:
        print(f'\nProcessing pathway: {pathway}')
        try:
            file_path = f'{path}{pathway}.gmt'
            (pathways, gene_sets) = read_gmt(file_path, lowercase_genes=True)
            if not gene_sets:
                print(f'Warning: pathway {pathway} has no valid genes; skipping')
                continue
            print(f'Gene-set size: {len(next(iter(gene_sets.values())))} genes')
            if adata_train_rna is not None and adata_test_rna is not None:
                adata = adata_train_rna.concatenate(adata_test_rna, batch_key='dataset')
                has_batch = True
            else:
                adata = adata_train_rna if adata_train_rna is not None else adata_test_rna
                has_batch = False
            net = pd.DataFrame([(pathway, gene) for (pathway, genes) in gene_sets.items() for gene in genes], columns=['source', 'target']).drop_duplicates()
            dc.run_aucell(adata, net, source='source', target='target', use_raw=False, verbose=True)
            pathway_activity = adata.obsm['aucell_estimate']
            adata.obsm[f'{pathway}_pathway_activity'] = pathway_activity
            pathway_activities.append(adata)
            if has_batch:
                adata_train = adata[adata.obs['dataset'] == '0'].copy()
                adata_test = adata[adata.obs['dataset'] == '1'].copy()
            else:
                adata_train = adata if adata_train_rna is not None else None
                adata_test = adata if adata_test_rna is not None else None
            if save:

                def _save_knn(adata, prefix):
                    if adata is not None and path_output is not None:
                        compute_and_save_knn(adata, path_output, pathway, prefix, n_neighbors)
                _save_knn(adata_train, 'train')
                _save_knn(adata_test, 'test')
        except Exception as e:
            print(f'Processing pathway {pathway} failed: {str(e)}')
            continue
    if not save:
        return pathway_activities
    print('KNN graph processing completed for all pathways.')
