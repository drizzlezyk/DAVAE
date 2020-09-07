import scanpy as sc
from utils import tools, plot
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder
import anndata
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import umap
import pandas as pd
from utils import tools as pp
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans

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

# --------------------293t-------------
outlabel_path = 'dann_vae/outlabel_01.csv'
outbatch_path = 'dann_vae/outbatch_01.csv'

davae_path = 'dann_vae/pbmc/293t_save05.h5ad'
orig_path = 'pbmc/zheng/293t_jurkat_merge.h5ad'
# jurkat_path = 'dann_vae/pbmc/293t_01.h5ad'
# scan_path = 'scanorama/293t_jurkat.h5ad'
# seurat_path = 'seurat_result/293t.csv'
# desc_path = 'desc/desc_jurkat.h5ad'
label_path = base_path + 'pbmc/zheng/293t_jurkat_cluster.txt'


# --------------------panc8-------------

# davae_path = 'dann_vae/panc8/panc8_merge5.h5ad'
# desc_path = 'desc/desc_panc8.h5ad'
# scan_path = 'scanorama/panc8_celseq2celseq.h5ad'
# seurat_path = 'seurat_result/panc8_cel2_cel.csv'
# tech2 = 'celseq2'
# tech3 = 'celseq'
# type = '_2000'
# adata2_orig, celltype2 = tools.get_panc8(tech2, type)
# adata3_orig, celltype3 = tools.get_panc8(tech3, type)
# celltype = np.concatenate([celltype2, celltype3])

# --------------------ifnb-------------
# davae_path = 'dann_vae/ifnb/ifnb_01.h5ad'
# desc_path = 'desc/desc_panc8.h5ad'
# scan_path = 'scanorama/panc8_celseq2celseq.h5ad'
# seurat_path = 'seurat_result/panc8_cel2_cel.csv'
# tech1 = 'ctrl'
# tech2 = 'stim'
# type = '_8000'
# adata2_orig, celltype2 = tools.get_ifnb(tech1, type)
# adata3_orig, celltype3 = tools.get_ifnb(tech2, type)
# celltype = np.concatenate([celltype2, celltype3])

# ----------------human two batch----------------
# davae_path = 'dann_vae/human/davae_01.h5ad'
# orig_path = 'pbmc/human_two_batch/PBMC.merged.h5ad'

# ---------------smartseq --------------------
# davae_path = 'dann_vae/benchmark1/davae_01.h5ad'
# orig_path = 'dann_vae/benchmark1/orig.h5ad'
# scan_path = 'scanorama/smart_seq.h5ad'

# ---------------atac --------------------
# davae_path = 'dann_vae/atac/davae_01.h5ad'
# orig_path = 'dann_vae/atac/orig.h5ad'
# celltype_x_path = base_path + 'seurat_data/sc_atac/pbmc_10k_v3_celltype.csv'
# celltype_x = pd.read_csv(celltype_x_path, index_col=0)
# celltype_x = celltype_x.values
#
# ---------------hca --------------------
# davae_path = 'dann_vae/hca/davae_01.h5ad'
# orig_path = 'dann_vae/hca/orig.h5ad'

# encoder = LabelEncoder()
# label = encoder.fit_transform(celltype_x)

adata_davae = sc.read_h5ad(base_path+davae_path)
# adata_scan = sc.read_h5ad(base_path+scan_path)
# adata_orig = sc.read_h5ad(base_path+orig_path)
# adata_seurat = sc.read_csv(base_path+seurat_path)
# adata_desc = sc.read_h5ad(base_path + desc_path)
# orig_batch = adata_orig.obs['batch']
# orig_label = adata_orig.obs['label']
# label = adata_davae.obs['label']
data = adata_davae.X
label = adata_davae.obs['label']
batches = adata_davae.obs['batch'].values
print(label[1])
# celltype = tools.get_label_by_txt(label_path)
# orig_label = tools.text_label_to_number(celltype)

figure_davae = base_path + 'dann_vae/pbmc/test.png'
figure_scan = base_path + 'scanorama/smart_seq.png'
figure_desc = base_path + 'desc/panc8.png'
figure_seurat = base_path+'seurat_result/pan8_cel2_cel.png'
figure_orig = base_path+'dann_vae/hca/orig.png'

if opt.dim_red == 1:
    data_emb = umap.UMAP().fit_transform(data)
    # data_emb = adata_davae.obsm['umap']
    # data_emb = data
    xy_label = 'umap'
elif opt.dim_red == 0:
    data_emb = TSNE().fit_transform(data)
    # data_emb = data
    xy_label = 'tsne'
else:
    # data_emb = PCA().fit_transform(data)
    data_emb = data
    xy_label = 'pca'

color_bar1 = ['plum', 'lightskyblue', 'mediumpurple', 'b', 'c']
# color = []
# for i in range(len(batches)):
#     color.append(color_bar1[batches[i]])

# color_bar2 = ['c', 'r', 'salmon']
# color_label = []
# for i in range(len(label)):
#     color_label.append(color_bar2[label[i]])

# data_emb = adata_desc.obsm['X_umap']
title = 'DAVAE'
fig = plt.figure(figsize=(18, 8))
ax = fig.add_subplot(121)
plt.title(title)
ax.scatter(data_emb[:, 0], data_emb[:, 1], cmap='Set2', c=batches, s=1, linewidth=0)
plt.xlabel(xy_label + '1')
plt.ylabel(xy_label + '2')

len1 = 7057
len2 = 9432
# data_emb = data_emb[len1:len1+len2, ]
ax = fig.add_subplot(122)
plt.title(title)
plt.xlabel(xy_label + '1')
plt.ylabel(xy_label + '2')
ax.scatter(data_emb[:, 0], data_emb[:, 1], c=label, cmap='Set2', s=1, linewidth=0)
plt.savefig(figure_davae)

# adata_davae.obsm['umap']=data_emb
# adata_davae.write_h5ad(base_path+davae_path)
np.savetxt("/Users/zhongyuanke/data/dann_vae/k_bet/293t/davae.csv", data, delimiter=",")
# np.savetxt("/Users/zhongyuanke/data/dann_vae/k_bet/davae_batch.csv", batches, delimiter=",")
# kmeans = KMeans(n_clusters=4).fit(data_emb)
#
#
# ari = adjusted_rand_score(label, kmeans.labels_)
# sh = silhouette_score(data_emb, kmeans.labels_)
# print(ari, sh)
# #
# ari_scan = adjusted_rand_score(label, kmeans.labels_)
# print(ari_scan)