import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.io import mmread
import os
import matplotlib.pyplot as plt

def qc_filter(adata, path_output, name, species, plot=False, save_data=False):
    """Retain the original reference-cell QC and per-split gene filter."""
    if species == 'mouse':
        adata.var['mt'] = adata.var_names.str.startswith('mt-')
    elif species == 'human':
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    if plot:
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        sc.pl.violin(adata, 'n_genes_by_counts', show=False)
        plt.title('Genes per cell')
        plt.subplot(132)
        sc.pl.violin(adata, 'total_counts', show=False)
        plt.title('UMIs per cell')
        plt.subplot(133)
        sc.pl.violin(adata, 'pct_counts_mt', show=False)
        plt.title('MT%')
        plt.tight_layout()
        plt.savefig(os.path.join(path_output, 'QC_metrics.png'))
        plt.close()
    min_genes = 50
    max_genes = 5000
    max_mt = 5
    if name == 'train':
        print(f'Original cell count: {adata.n_obs}')
        adata = adata[adata.obs['n_genes_by_counts'] < max_genes, :]
        adata = adata[adata.obs['n_genes_by_counts'] > min_genes, :]
        adata = adata[adata.obs['pct_counts_mt'] < max_mt, :]
    sc.pp.filter_genes(adata, min_cells=10)
    print(f'Cell count after filtering: {adata.n_obs}')
    print(f'Gene count after filtering: {adata.n_vars}')
    if save_data:
        adata.write(os.path.join(path_output, f'{name}_processed.h5ad'))
    return adata

def data_processing(train_path, test_path, path_output, species_train, species_test, plot, save_data, qc_filter_bool=True):
    """Read cells-by-genes exports, preserve metadata row order, and align genes."""
    cell_names_train = pd.read_csv(train_path + 'barcodes.tsv', sep='\t', header=None, index_col=None)
    cell_names_train.columns = ['cells']
    X_train = csr_matrix(mmread(train_path + 'matrix.mtx').T).astype(float)
    gene_names_train = pd.read_csv(train_path + 'features.tsv', sep='\t', header=None, index_col=None)
    gene_names_train.columns = ['genes']
    gene_names_train['genes'] = gene_names_train['genes'].str.lower()
    adata_train_rna = sc.AnnData(X_train.T, obs=pd.DataFrame(index=cell_names_train.cells), var=pd.DataFrame(index=gene_names_train.genes))
    adata_train_rna.var_names_make_unique()
    metadata_path = os.path.join(train_path, 'metadata.csv')
    metadata = pd.read_csv(metadata_path)
    metadata.index = adata_train_rna.obs_names
    if 'CellType' in metadata.columns:
        adata_train_rna.obs['cell_type'] = metadata['CellType'].astype('category')
    elif 'cell_type' in metadata.columns:
        adata_train_rna.obs['cell_type'] = metadata['cell_type'].astype('category')
    else:
        print("Error:The train set doesn't has cell labels.")
    cell_names_test = pd.read_csv(test_path + 'barcodes.tsv', sep='\t', header=None, index_col=None)
    cell_names_test.columns = ['cells']
    X_test = csr_matrix(mmread(test_path + '/matrix.mtx').T).astype(float)
    gene_names_test = pd.read_csv(test_path + 'features.tsv', sep='\t', header=None, index_col=None)
    gene_names_test.columns = ['genes']
    gene_names_test['genes'] = gene_names_test['genes'].str.lower()
    adata_test_rna = sc.AnnData(X_test.T, obs=pd.DataFrame(index=cell_names_test.cells), var=pd.DataFrame(index=gene_names_test.genes))
    adata_test_rna.var_names_make_unique()
    metadata_path = os.path.join(test_path, 'metadata.csv')
    metadata = pd.read_csv(metadata_path)
    metadata.index = adata_test_rna.obs_names
    if 'CellType' in metadata.columns:
        adata_test_rna.obs['cell_type'] = metadata['CellType'].astype('category')
    elif 'cell_type' in metadata.columns:
        adata_test_rna.obs['cell_type'] = metadata['cell_type'].astype('category')
    else:
        adata_test_rna.obs['cell_type'] = 'unknown'
        print("The test set doesn't has cell labels.")
    if qc_filter_bool:
        adata_train_rna = qc_filter(adata_train_rna, path_output, 'train', species_train, plot, save_data)
        adata_test_rna = qc_filter(adata_test_rna, path_output, 'test', species_test, plot, save_data)
    common_genes = list(set(adata_train_rna.var_names) & set(adata_test_rna.var_names))
    print(f'Number of shared genes: {len(common_genes)}')
    adata_train_rna = adata_train_rna[:, common_genes]
    adata_test_rna = adata_test_rna[:, common_genes]
    return (adata_train_rna, adata_test_rna)
from scipy import sparse

def preprocess(adata, hvg_genes=None, needs_normalization=True, needs_log_transform=True, needs_scale=False):
    """Apply source transforms and use reference HVGs for both data splits."""
    is_training = hvg_genes is None
    if needs_normalization:
        print('Normalizing data...')
        sc.pp.normalize_total(adata, target_sum=10000.0)
    else:
        print('Skipping normalization')
    if needs_log_transform:
        print('Applying log1p transformation...')
        sc.pp.log1p(adata)
    else:
        print('Skipping log1p transformation')
    if is_training:
        sc.pp.highly_variable_genes(adata, n_top_genes=4000, flavor='seurat_v3', span=1.0)
        hvg_genes = adata.var_names[adata.var.highly_variable]
        adata = adata[:, hvg_genes]
    else:
        valid_genes = [g for g in hvg_genes if g in adata.var_names]
        adata = adata[:, valid_genes]
        adata = adata[:, hvg_genes]
    if needs_scale:
        print('Scaling data... ')
        sc.pp.scale(adata, max_value=10)
    if is_training:
        return (adata, hvg_genes)
    else:
        return adata
from scipy.sparse import issparse
import numpy as np

def preprocess_pca(adata, num=3000, fit_model=None):
    sc.pp.normalize_total(adata, target_sum=10000.0)
    sc.pp.log1p(adata)
    if fit_model is None:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=num, random_state=42)
        if issparse(adata.X):
            X_dense = adata.X.toarray()
        else:
            X_dense = adata.X
        adata.obsm['X_pca'] = np.ascontiguousarray(pca.fit_transform(X_dense))
        return (adata, pca)
    if issparse(adata.X):
        X_dense = adata.X.toarray()
    else:
        X_dense = adata.X
    adata.obsm['X_pca'] = np.ascontiguousarray(fit_model.transform(X_dense))
    return adata
