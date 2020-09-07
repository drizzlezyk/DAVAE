import scanpy as sc
from utils import tools, plot
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.preprocessing import normalize
import umap
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


# -----------------panc8-------------------------------------------
# orig_path = 'dann_vae/panc8/orig_cel2_cel.h5ad'
# desc_path = 'desc/desc_panc8.h5ad'
# davae_path = 'dann_vae/panc8/panc8_merge_01.h5ad'
# seurat_path = 'seurat_result/panc8_cel2_cel.csv'
# scan_path = 'scanorama/panc8_celseq2celseq.h5ad'
# -----------------293t---------------------------------------------
pbmc_path = 'dann_vae/pbmc/davae_save01.h5ad'
orig_path = 'dann_vae/pbmc/orig.h5ad'
davae_path = 'dann_vae/pbmc/293t_save04.h5ad'
scan_path = 'scanorama/293t_jurkat.h5ad'
seurat_path = 'seurat_result/293t.csv'
desc_path = 'desc/desc_jurkat.h5ad'
label_path = base_path + 'pbmc/zheng/293t_jurkat_cluster.txt'
# -----------------ifnb----------------------------------------------
# orig_path = 'dann_vae/ifnb/ifnb_orig.h5ad'
# davae_path = 'dann_vae/ifnb/ifnb_01.h5ad'
# scan_path = 'scanorama/ifnb.h5ad'
# seurat_path = 'seurat_result/ifnb.csv'
# desc_path = 'desc/ifnb.h5ad'
# -----------------smartseq----------------------------------------------
# orig_path = 'dann_vae/benchmark1/orig.h5ad'
# davae_path = 'dann_vae/benchmark1/davae_save01.h5ad'
# scan_path = 'scanorama/smart_seq.h5ad'
# seurat_path = 'seurat_result/smartseq.csv'
# desc_path = 'desc/smartseq.h5ad'



# label = np.loadtxt(base_path+outlabel_path, delimiter=",")
# batches = np.loadtxt(base_path+outbatch_path, delimiter=',')
adata_davae = sc.read_h5ad(base_path+davae_path)
adata_scan = sc.read_h5ad(base_path+scan_path)
adata_orig = sc.read_h5ad(base_path+orig_path)
adata_seurat = sc.read_csv(base_path+seurat_path)
adata_desc = sc.read_h5ad(base_path + desc_path)

sc.pp.filter_genes(adata_orig, min_cells=1)

data_scan = adata_scan.X
data_davae = adata_davae.X
data_desc = adata_desc.X
data_desc_emb = adata_desc.obsm['X_umap0.8']
# data_orig_emb = adata_orig.obsm['umap']
data_orig = adata_orig.X
data_seurat = adata_seurat.X

# data_orig_emb = umap.UMAP().fit_transform(data_orig)
# adata_orig.obsm['umap'] = data_orig_emb
# adata_orig.write_h5ad(base_path+orig_path)
print('finish write')

label = adata_davae.obs['label']
batches = adata_davae.obs['batch']
# np.savetxt('/Users/zhongyuanke/data/dann_vae/k_bet/davae_batch.csv', batches)

orig_batches = adata_orig.obs['batch']

# celltype = tools.get_label_by_txt(label_path)
# orig_label = tools.text_label_to_number(celltype)
# adata_orig.obs['label'] = orig_label
# adata_orig.write_h5ad(base_path+orig_path)
orig_label = adata_orig.obs['label']

figure_merge = base_path + 'dann_vae/pbmc/compare02.png'

print(data_orig.shape)
if opt.dim_red == 1:
    data_orig_emb = adata_orig.obsm['umap']
    # data_orig_emb = umap.UMAP().fit_transform(data_orig)
    data_davae_emb = umap.UMAP().fit_transform(data_davae)
    data_seurat_emb = umap.UMAP().fit_transform(data_seurat)
    data_scan_emb = umap.UMAP().fit_transform(data_scan)
    # data_emb = data
    xy_label = 'umap'
    adata_orig.obsm['umap'] = data_orig_emb

elif opt.dim_red == 0:
    data_orig_emb = TSNE().fit_transform(data_orig)
    data_davae_emb = TSNE().fit_transform(data_davae)
    data_seurat_emb = TSNE().fit_transform(data_seurat)
    data_scan_emb = TSNE().fit_transform(data_scan)
    # data_emb = data
    xy_label = 'tsne'
else:
    data_orig_emb = umap.UMAP().fit_transform(data_orig)
    data_davae_emb = umap.UMAP().fit_transform(data_davae)
    data_seurat_emb = umap.UMAP().fit_transform(data_seurat)
    data_scan_emb = umap.UMAP().fit_transform(data_scan)
    xy_label = 'pca'
# adata_orig.write_h5ad(base_path+orig_path)
batches = np.array(batches, dtype=int)
orig_batches = np.array(orig_batches, dtype=int)


batch_cmap = 'tab20b'
c_map = 'Set3'

fig = plt.figure(figsize=(34, 13))
ax = fig.add_subplot(251)
plt.title('Raw', fontsize=20)
ax.scatter(data_orig_emb[:, 0], data_orig_emb[:, 1], c=orig_batches, cmap=batch_cmap, s=3)
plt.xlabel(xy_label + '1')
plt.ylabel(xy_label + '2')

ax = fig.add_subplot(256)
ax.scatter(data_orig_emb[:, 0], data_orig_emb[:, 1], c=orig_label, cmap=c_map, s=3)
plt.xlabel(xy_label + '1')
plt.ylabel(xy_label + '2')

# ---------------------------------------

ax = fig.add_subplot(252)
plt.title('DAVAE', fontsize=20)
ax.scatter(data_davae_emb[:, 0], data_davae_emb[:, 1], c=batches, cmap=batch_cmap, s=3)

plt.xlabel(xy_label + '1')
plt.ylabel(xy_label + '2')

ax = fig.add_subplot(257)
ax.scatter(data_davae_emb[:, 0], data_davae_emb[:, 1], c=label, cmap=c_map, s=3)

plt.xlabel(xy_label + '1')
plt.ylabel(xy_label + '2')

# ---------------------------------------

ax = fig.add_subplot(253)
plt.title('Scanorama', fontsize=20)
ax.scatter(data_scan_emb[:, 0], data_scan_emb[:, 1], c=orig_batches, cmap=batch_cmap, s=3)
plt.xlabel(xy_label + '1')
plt.ylabel(xy_label + '2')

ax = fig.add_subplot(258)
ax.scatter(data_scan_emb[:, 0], data_scan_emb[:, 1], c=orig_label, cmap=c_map,  s=3)
plt.xlabel(xy_label + '1')
plt.ylabel(xy_label + '2')

# ---------------------------------------

ax = fig.add_subplot(254)
plt.title('DESC', fontsize=20)
ax.scatter(data_desc_emb[:, 0], data_desc_emb[:, 1], c=orig_batches, cmap=batch_cmap, s=3)
plt.xlabel(xy_label + '1')
plt.ylabel(xy_label + '2')

ax = fig.add_subplot(259)
ax.scatter(data_desc_emb[:, 0], data_desc_emb[:, 1], c=orig_label, cmap=c_map, s=3)
plt.xlabel(xy_label + '1')
plt.ylabel(xy_label + '2')

# ---------------------------------------

ax = fig.add_subplot(255)
plt.title('Seurat 3.0', fontsize=20)
ax.scatter(data_seurat_emb[:, 0], data_seurat_emb[:, 1], c=orig_batches, cmap=batch_cmap, s=3)
plt.xlabel(xy_label + '1')
plt.ylabel(xy_label + '2')

ax = fig.add_subplot(2, 5, 10)
ax.scatter(data_seurat_emb[:, 0], data_seurat_emb[:, 1], c=orig_label, cmap=c_map, s=3)
plt.xlabel(xy_label + '1')
plt.ylabel(xy_label + '2')

plt.savefig(figure_merge)
plt.close(fig)

cluster_count = 2

kmeans_orig = KMeans(n_clusters=cluster_count).fit(data_orig_emb)
# kmeans_orig = KMeans().fit(data_orig_emb)
ari_orig = adjusted_rand_score(orig_label, kmeans_orig.labels_)
sh_orig = silhouette_score(data_orig_emb, kmeans_orig.labels_)
print('original', ari_orig, sh_orig)

kmeans_davae = KMeans(n_clusters=cluster_count).fit(data_davae_emb)
# kmeans_davae = KMeans().fit(data_davae_emb)
ari_davae = adjusted_rand_score(label, kmeans_davae.labels_)
sh_davae = silhouette_score(data_davae_emb, kmeans_davae.labels_)
print('davae', ari_davae, sh_davae)

kmeans_scan = KMeans(n_clusters=cluster_count).fit(data_scan_emb)
# kmeans_scan = KMeans().fit(data_scan_emb)
ari_scan = adjusted_rand_score(orig_label, kmeans_scan.labels_)
sh_scan = silhouette_score(data_scan_emb, kmeans_scan.labels_)
print('scanorama', ari_scan, sh_scan)

kmeans_desc = KMeans(n_clusters=cluster_count).fit(data_desc_emb)
# kmeans_desc = KMeans().fit(data_desc_emb)
ari_desc = adjusted_rand_score(orig_label, kmeans_desc.labels_)
sh_desc = silhouette_score(data_desc_emb, kmeans_desc.labels_)
print('desc', ari_desc, sh_desc)

kmeans_seurat = KMeans(n_clusters=cluster_count).fit(data_seurat_emb)
# kmeans_seurat = KMeans().fit(data_seurat_emb)
ari_seurat = adjusted_rand_score(orig_label, kmeans_seurat.labels_)
sh_seurat = silhouette_score(data_seurat_emb, kmeans_seurat.labels_)
print('seurat', ari_seurat, sh_seurat)


name_list = ['RAW', 'DAVAE', 'Scanorama', 'DESC', 'Seurat 3.0']
num_list = [ari_orig, ari_davae, ari_scan, ari_desc, ari_seurat]
num_list1 = [sh_orig, sh_davae, sh_scan, sh_desc, sh_desc]
print(num_list)


num_list = np.array(num_list)
num_list1 = np.array(num_list1)
# np.savetxt(base_path+'dann_vae/benchmark1/ari.csv', num_list, delimiter=',')
#
num = np.concatenate([num_list, num_list1], axis=-1)
np.savetxt(base_path+'dann_vae/benchmark1/ari_sh_02.csv', num, delimiter=',')
#

# np.savetxt("/Users/zhongyuanke/data/dann_vae/k_bet/davae.csv", data, delimiter=",")
# np.savetxt("/Users/zhongyuanke/data/dann_vae/k_bet/davae_batch.csv", batches, delimiter=",")
# kmeans_scan = KMeans(n_clusters=2).fit(data_scan_emb)
# kmeans_seurat = KMeans(n_clusters=2).fit(data_scan_emb)
# kmeans_scan = KMeans(n_clusters=2).fit(data_scan_emb)
# kmeans_scan = KMeans(n_clusters=2).fit(data_scan_emb)
# #
# #
# ari = adjusted_rand_score(orig_label, kmeans.labels_)
# sh = silhouette_score(data_emb, kmeans.labels_)
# print(ari, sh)
# #
# ari_scan = adjusted_rand_score(label, kmeans.labels_)
# print(ari_scan)