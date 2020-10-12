import scanpy as sc
from utils import tools, plot
import csv
import numpy as np
import argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import umap
from utils import tools as pp
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument("--dim_red", type=int, default=1, help="method for dimension reduction, 1 for umap, 0 for tsne, 2 "
                                                           "for pca")
parser.add_argument("--x_filetype", type=str, default='10x_mtx', help="file type of x: 10x_mtx/h5ad/matrix.mtx")
parser.add_argument("--y_filetype", type=str, default='10x_mtx', help="file type of x: 10x_mtx/h5ad/matrix.mtx")
opt = parser.parse_args()

base_path = '/Users/zhongyuanke/data/'
# x_cluster = base_path+'clustering/frozen_pbmc_b_c_50_50/kmeans/7_clusters/clusters.csv'
# y_cluster = base_path+'clustering/frozen_pbmc_b_c_90_10/kmeans/7_clusters/clusters.csv'
# file_orig_x = base_path+'pbmc/frozen_pbmc_b_c_50_50/hg19'
# file_orig_y = base_path+'pbmc/frozen_pbmc_b_c_90_10/hg19'
# file_vcca_x = base_path+'vcca/doner/doner_bc_5050_vcca_02.h5ad'
# file_vcca_y = base_path+'vcca/doner/doner_bc_9010_vcca_02.h5ad'
# file_vcca_x = base_path+'vcca/panc8/celseq2smartseq2_celseq2_scanorama_01.h5ad'
# file_vcca_y = base_path+'vcca/panc8/celseq2smartseq2_smartseq2_scanorama_01.h5ad'
# orig_path = base_path+'vcca/panc8/celseq2smartseq2_orig.png'
# vcca_path = base_path+'vcca/panc8/celseq2smartseq2_scanorama_umap_01.png'


c_map = 'Set2'
title = 'VCCA'
k = 9


# print('start original dimension reduction')
# orig_adata_x = pp.read_sc_data(file_orig_x, fmt=opt.x_filetype)
# orig_adata_y = pp.read_sc_data(file_orig_y, fmt=opt.y_filetype)
# sc.pp.filter_genes(orig_adata_x, min_cells=50)
# sc.pp.filter_genes(orig_adata_y, min_cells=50)
# orig_y = orig_adata_y.X
# orig_x = orig_adata_x.X
# print(orig_y.shape)
# print(orig_x.shape)
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
# merge_data = np.concatenate([orig_x_emb, orig_y_emb])
# orig_label = KMeans(n_clusters=k, random_state=0).fit_predict(merge_data)
# plot.plot_integration_with_cluster(orig_x_emb, orig_y_emb, orig_label, orig_path, c_map, xy_label,
#                                    title='Original')

tech1 = 'celseq2'
tech2 = 'celseq'
tech3 = 'indrop'
tech4 = 'smartseq2'
tech5 = 'fluidigmc1'
adata1, celltype1 = tools.get_panc8(tech1)
adata2, celltype2 = tools.get_panc8(tech2)
adata3, celltype3 = tools.get_panc8(tech3)
adata4, celltype4 = tools.get_panc8(tech4)
adata5, celltype5 = tools.get_panc8(tech5)
orig_path = base_path + 'vcca/panc8/' + tech1 + tech2 + '_orig.png'

celltypes = [celltype1, celltype2, celltype3, celltype4, celltype5]
celltype = np.concatenate(celltypes)
label = tools.text_label_to_number(celltype)

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
file_vcca_x = base_path + 'vcca/panc8/' + tech1 + tech2 + '_' + tech1 + '_vcca_01.h5ad'
file_vcca_y = base_path + 'vcca/panc8/' + tech1 + tech2 + '_' + tech2 + '_vcca_01.h5ad'

file_scxx = base_path + 'scxx/vae_panc8_01.h5ad'
file_dann_vae = base_path + 'dann_vae/panc8_01.h5ad'
file_vcca = base_path + 'vcca/panc8_merge5_01.h5ad'
# file_scan_1 = base_path + 'scanorama/'+tech1+'_01.h5ad'
# file_scan_2 = base_path + 'scanorama/'+tech2+'_01.h5ad'
# file_scan_3 = base_path + 'scanorama/'+tech3+'_01.h5ad'
# file_scan_4 = base_path + 'scanorama/'+tech4+'_01.h5ad'
# file_scan_5 = base_path + 'scanorama/'+tech5+'_01.h5ad'

file_scan_1 = base_path + 'vcca/panc8/merge5_'+tech1+'_01.h5ad'
file_scan_2 = base_path + 'vcca/panc8/merge5_'+tech2+'_01.h5ad'
file_scan_3 = base_path + 'vcca/panc8/merge5_'+tech3+'_01.h5ad'
file_scan_4 = base_path + 'vcca/panc8/merge5_'+tech4+'_01.h5ad'
file_scan_5 = base_path + 'vcca/panc8/merge5_'+tech5+'_01.h5ad'

vcca_path = base_path + 'vcca/merge5_vcca_l2_01.png'
scan_path = base_path + 'scanorama/merge5_01.png'
scxx_path = base_path + 'scxx/vae_merge5_01.png'
dann_vae_path = base_path + 'dann_vae/dann_vae_merge5_01.png'
# adata_vcca_x = sc.read_h5ad(file_vcca_x)
# adata_vcca_y = sc.read_h5ad(file_vcca_y)

scan_adata_1 = sc.read_h5ad(file_scan_1)
scan_adata_2 = sc.read_h5ad(file_scan_2)
scan_adata_3 = sc.read_h5ad(file_scan_3)
scan_adata_4 = sc.read_h5ad(file_scan_4)
scan_adata_5 = sc.read_h5ad(file_scan_5)

# vcca_x = adata_vcca_x.X
# vcca_y = adata_vcca_y.X
# vcca_x = normalize(vcca_x, norm='l2', axis=1, copy=True, return_norm=False)
# vcca_y = normalize(vcca_y, norm='l2', axis=1, copy=True, return_norm=False)
# print(vcca_x.shape)
# print(vcca_y.shape)

scan1 = scan_adata_1.X
scan2 = scan_adata_2.X
scan3 = scan_adata_3.X
scan4 = scan_adata_4.X
scan5 = scan_adata_5.X

len1 = scan1.shape[0]
len2 = scan2.shape[0]
len3 = scan3.shape[0]
len4 = scan4.shape[0]
len5 = scan5.shape[0]
print(len1, len2, len3, len4, len5)
datasets = []
# scan = np.concatenate([scan1, scan2, scan3, scan4, scan5], axis=0)
scan = sc.read_h5ad(file_dann_vae)
scan = scan.X
if opt.dim_red == 1:
    scan_emb = umap.UMAP().fit_transform(scan)
    xy_label = 'umap'
elif opt.dim_red == 0:
    scan_emb = TSNE().fit_transform(scan)
    xy_label = 'tsne'
else:
    scan_emb = PCA().fit_transform(scan)
    xy_label = 'pca'
scan_emb1 = scan_emb[0:len1]
scan_emb2 = scan_emb[len1:len1+len2]
scan_emb3 = scan_emb[len1+len2:len1+len2+len3]
scan_emb4 = scan_emb[len1+len2+len3:len1+len2+len3+len4]
scan_emb5 = scan_emb[len1+len2+len3+len4:len1+len2+len3+len4+len5]
scan_datasets = [scan_emb1, scan_emb2, scan_emb3, scan_emb4, scan_emb5]
plot.plot_integration_with_cluster(scan_datasets, label, dann_vae_path, c_map, xy_label, title=method)