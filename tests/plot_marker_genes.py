import scanpy as sc
from utils import plot
import umap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


c_map = 'Purples'
base_path = '/Users/zhongyuanke/data/'
fig_size = (28, 5)
title = ['CD3', 'CD4', 'CD14', 'CD8a', 'CD19']

bar = ['silver', 'r']

file_orig = base_path+'dann_vae/hca/orig.h5ad'

file_davae = base_path+'dann_vae/hca/davae_01.h5ad'

fig_path = base_path+'dann_vae/hca/marker_01.png'
label_fig_path = base_path+'vcca/atac/chro_label.png'
orig_umap = base_path+'result/merge5_umap.h5ad'
adata = sc.read_h5ad(file_orig)
print(adata.shape)
# genes = adata.var
# adata_orig = sc.read_h5ad(orig_umap)
adata_davae = sc.read_h5ad(file_davae)
genes = adata.var
# adata_dca = sc.read_h5ad(file_dca)
# adata_mse = sc.read_h5ad(file_mse)
# adata_scxx = sc.read_h5ad(file_scxx)
# adata_scan = sc.read_h5ad(file_scan)
# x_orig = adata_orig.obsm['umap']
# x_mse = adata_mse.obsm['mid']
# x_dca = adata_dca.obsm['mid']
# x_scvi = adata_scxx.obsm['mid']
# x_scan = adata_scan.obsm['umap']
data_emb = adata_davae.obsm['umap']
xy_label = 'umap'

all_markers = ['LYZ', 'CD14', 'CST3', 'IL3RA', 'CD79A', 'MS4A1', 'MME', 'SDC1', 'CD34', 'HBB',
               'NKG7', 'PPBP', 'CD8B', 'CD4', 'PTPRC', 'CCR7', 'IL7R']

gene_names = ['LYZ', 'CD14', 'CD79A', 'MS4A1']
# gene_names = ['NKG7', 'CD8B', 'CST3', 'IL7R']
thresholds = [7, 4, 7, 4]
plot.plot_gradient_marker_gene(data_emb, adata, gene_names, thresholds, 'UMAP', fig_size, 14, fig_path)

#-------------------------------------------------------------------------
# celltype_path = '/Users/zhongyuanke/data/atac/p0_BrainCortex/label.csv'
# cell_type = pd.read_csv(celltype_path, index_col=0)
# cell_type = np.array(cell_type)
# label = []
# for i in range(5081):
#     label.append(cell_type[i, 0])
# label = np.array(label)
#
# fig = plt.figure(figsize=(10, 6))
# scatter = plt.scatter(vcca_emb[:, 0], vcca_emb[:, 1], c=label, cmap='tab20c', s=3, linewidth=0)
# plt.xlabel(xy_label + '1')
# plt.ylabel(xy_label + '2')
# # plt.title('VCCA')
# handles, labels = scatter.legend_elements(num=19)
# celltype = ['Ex23_Cux1', 'Ex6_Tle4', 'Ex345_Foxp1', 'IP_Eomes', 'RG', 'Ex4_Tenm3', 'Ex5_Crmp1', 'Ex45_Galntl6', 'In_1',
#             'In_2', 'IP_Hmgn2', 'Ex56_Epha6', 'Ex23_Cntn2', 'IP_Gadd45g', 'CR', 'Endo', 'Peri', 'OPC', 'Mic']
#
# legend2 = plt.legend(handles, celltype, loc="upper center", ncol=4)
# plt.savefig(label_fig_path)

