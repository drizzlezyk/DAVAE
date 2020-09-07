import scanpy as sc
from utils import tools, plot
import csv
import pandas as pd
import numpy as np
import argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import umap
from utils import tools as pp
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()
parser.add_argument("--dim_red", type=int, default=1, help="method for dimension reduction, 1 for umap, 0 for tsne, 2 "
                                                           "for pca")
parser.add_argument("--x_filetype", type=str, default='csv', help="file type of x: 10x_mtx/h5ad/matrix.mtx")
parser.add_argument("--y_filetype", type=str, default='csv', help="file type of x: 10x_mtx/h5ad/matrix.mtx")
opt = parser.parse_args()

base_path = '/Users/zhongyuanke/data/'
file_orig_x = base_path+'sc_atac/pbmc_10k_v3_scale.csv'
file_orig_y = base_path+'sc_atac/atac_gene_activity.csv'
celltype_x_path = base_path + 'sc_atac/pbmc_10k_v3_celltype.csv'
orig_path = base_path + 'vcca/atac/pbmc_atac_orig.png'
c_map = 'Set2'
title = 'VCCA'

orig_adata_x = pp.read_sc_data(file_orig_x, fmt=opt.x_filetype)
orig_adata_y = pp.read_sc_data(file_orig_y, fmt=opt.y_filetype)

celltype_x = pd.read_csv(celltype_x_path, index_col=0)
celltype_x = celltype_x.values
celltype_x = tools.text_label_to_number(celltype_x)
print(celltype_x)
celltype_y = np.zeros([orig_adata_y.shape[0],])
for i in range(orig_adata_y.shape[0]):
    celltype_y[i] = 20

label = np.concatenate([celltype_x, celltype_y])


# orig_x = data_x.X
# orig_y = data_y.X
#
# if opt.dim_red == 1:
#     orig_x_emb = umap.UMAP().fit_transform(orig_x)
#     orig_y_emb = umap.UMAP().fit_transform(orig_y)
#     xy_label = 'umap'
# elif opt.dim_red == 0:
#     orig_x_emb = TSNE().fit_transform(orig_x)
#     orig_y_emb = TSNE().fit_transform(orig_y)
#     xy_label = 'tsne'
# else:
#     orig_x_emb = PCA(n_components=2).fit_transform(orig_x)
#     orig_y_emb = PCA(n_components=2).fit_transform(orig_y)
#     xy_label = 'pca'
# plot.plot_integration_with_cluster(orig_x_emb, orig_y_emb, label, orig_path, c_map, xy_label, title='Original')


method = 'VCCA'
file_vcca_x = base_path + 'vcca/atac/pbmc_atac_vcca_01.h5ad'
file_vcca_y = base_path + 'vcca/atac/pbmc_rna_vcca_01.h5ad'

# file_vcca_x = base_path + 'vcca/panc8/' + tech1 + tech2 + '_' + tech1 + '_scanorama_01.h5ad'
# file_vcca_y = base_path + 'vcca/panc8/' + tech1 + tech2 + '_' + tech2 + '_scanorama_01.h5ad'
vcca_path = base_path + 'vcca/atac/pbmc_rna_atac_vcca_01.png'

adata_vcca_x = sc.read_h5ad(file_vcca_x)
adata_vcca_y = sc.read_h5ad(file_vcca_y)
vcca_x = adata_vcca_x.X
vcca_y = adata_vcca_y.X
vcca_x = normalize(vcca_x, norm='l2', axis=1, copy=True, return_norm=False)
vcca_y = normalize(vcca_y, norm='l2', axis=1, copy=True, return_norm=False)
print(vcca_x.shape)
print(vcca_y.shape)
if opt.dim_red == 1:
    vcca_x_emb = umap.UMAP().fit_transform(vcca_x)
    vcca_y_emb = umap.UMAP().fit_transform(vcca_y)
    xy_label = 'umap'
elif opt.dim_red == 0:
    vcca_x_emb = TSNE().fit_transform(vcca_x)
    vcca_y_emb = TSNE().fit_transform(vcca_y)
    xy_label = 'tsne'
else:
    vcca_x_emb = PCA().fit_transform(vcca_x)
    vcca_y_emb = PCA().fit_transform(vcca_y)
    xy_label = 'pca'
merge_vcca = np.concatenate([vcca_x_emb, vcca_y_emb])
plot.plot_integration_with_cluster(vcca_x_emb, vcca_y_emb, label, vcca_path, c_map, xy_label, title=method)