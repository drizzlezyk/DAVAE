import desc
import numpy as np
import pandas as pd
import scanpy.api as sc
import matplotlib
import os
os.environ['PYTHONHASHSEED'] = '0'
import matplotlib.pyplot as plt
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_versions()


base_path = '/Users/zhongyuanke/data/'
orig_merge_293t = base_path+'pbmc/zheng/293t_jurkat_merge.h5ad'
file1 = base_path+'pbmc/zheng/293t/hg19/'
file2 = base_path+'pbmc/zheng/jurkat/hg19/'
file3 = base_path + 'pbmc/zheng/293t_jurkat_50_50/hg19/'
# adata1 = sc.read_10x_mtx(file1)
# adata2 = sc.read_10x_mtx(file2)
# adata3 = sc.read_10x_mtx(file3)
adata = sc.read_h5ad(orig_merge_293t)

mito_genes = adata.var_names.str.startswith('MT-')
# for each cell compute fraction of counts in mito genes vs. all genes
# the `.A1` is only necessary as X is sparse (to transform to a dense array after summing)
adata.obs['percent_mito'] = np.sum(
    adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
# add the total counts per cell as observations-annotation to adata
adata.obs['n_counts'] = adata.X.sum(axis=1).A1

adata = adata[adata.obs['percent_mito'] < 0.05, :]

desc.normalize_per_cell(adata, counts_per_cell_after=1e4)
# sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, subset=True)
# adata = adata[:, adata.var['highly_variable']]
adata = desc.train(adata, dims=[adata.shape[1], 32, 16], tol=0.005, n_neighbors=10,
                   batch_size=256, louvain_resolution=[0.8],
                   save_dir="result_pbmc3k", do_tsne=True, learning_rate=300,
                   do_umap=True, num_Cores_tsne=4,
                   save_encoder_weights=True)

print(adata.obsm['X_umap0.8'])

