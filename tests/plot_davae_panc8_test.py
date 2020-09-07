import scanpy as sc
from utils import tools, plot
import matplotlib.pyplot as plt
import numpy as np
import argparse
import anndata
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import umap
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
c_map = 'Set2'


file1 = base_path+'dann_vae/panc8/indrop_out.h5ad'
file2 = base_path+'dann_vae/panc8/celseq2_out.h5ad'
file3 = base_path+'dann_vae/panc8/celseq_out.h5ad'
file4 = base_path+'dann_vae/panc8/smartseq2_out.h5ad'
file5 = base_path+'dann_vae/panc8/fluidigmc1_out.h5ad'

adata1 = sc.read_h5ad(file1)
adata2 = sc.read_h5ad(file2)
adata3 = sc.read_h5ad(file3)
adata4 = sc.read_h5ad(file4)
adata5 = sc.read_h5ad(file5)
adata = adata1.concatenate(adata2)

data = adata.X
label = adata.obs['label']
batches = adata.obs['batch']

figure_davae = base_path + 'dann_vae/panc8/merge5.png'

if opt.dim_red == 1:
    data_emb = umap.UMAP().fit_transform(data)
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

batches = np.array(batches, dtype=int)

title = 'DAVAE'
fig = plt.figure(figsize=(18, 8))
ax = fig.add_subplot(121)
plt.title(title)
ax.scatter(data_emb[:, 0], data_emb[:, 1], c=batches, cmap='Set3', s=1)
plt.xlabel(xy_label + '1')
plt.ylabel(xy_label + '2')

ax = fig.add_subplot(122)
plt.title(title)
plt.xlabel(xy_label + '1')
plt.ylabel(xy_label + '2')
ax.scatter(data_emb[:, 0], data_emb[:, 1], c=label, cmap='Set2', s=1)
plt.savefig(figure_davae)

# np.savetxt("/Users/zhongyuanke/data/dann_vae/k_bet/davae.csv", data, delimiter=",")
# np.savetxt("/Users/zhongyuanke/data/dann_vae/k_bet/davae_batch.csv", batches, delimiter=",")
kmeans = KMeans().fit(data_emb)
#
#
ari = adjusted_rand_score(label, kmeans.labels_)
sh = silhouette_score(data_emb, kmeans.labels_)
print(ari, sh)
#
# ari_scan = adjusted_rand_score(label, kmeans.labels_)
# print(ari_scan)