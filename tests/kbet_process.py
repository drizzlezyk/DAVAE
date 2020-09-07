import scanpy as sc
import numpy as np
import umap

base_path = '/Users/zhongyuanke/data/'

davae_path = base_path + 'dann_vae/pbmc/293t_01.h5ad'
pbmc_path = base_path + 'dann_vae/pbmc/davae_save01.h5ad'
orig_path = base_path + 'pbmc/zheng/293t_jurkat_merge_filt.csv'
scan_path = base_path + 'scanorama/293t_jurkat.h5ad'
seurat_path = base_path + 'seurat_result/293t.csv'
desc_path = base_path + 'desc/desc_jurkat.h5ad'
label_path = base_path + 'pbmc/zheng/293t_jurkat_cluster.txt'

davae_kbet = base_path + 'dann_vae/k_bet/293t/davae.csv'
orig_kbet = base_path + 'dann_vae/k_bet/293t/orig.csv'
scan_kbet = base_path + 'dann_vae/k_bet/293t/scan.csv'
desc_kbet = base_path + 'dann_vae/k_bet/293t/desc.csv'
seurat_kbet = base_path + 'dann_vae/k_bet/293t/seurat.csv'


adata_davae = sc.read_h5ad(davae_path)
adata_scan = sc.read_h5ad(scan_path)
adata_orig = sc.read_csv(orig_path)
adata_seurat = sc.read_csv(seurat_path)
adata_desc = sc.read_h5ad(desc_path)


print(adata_orig.shape)
data_scan = adata_scan.X
data_davae = adata_davae.X
data_desc = adata_desc.X
# data_orig_emb = adata_orig.obsm['umap']
data_orig = adata_orig.X
data_seurat = adata_seurat.X


data_orig_emb = umap.UMAP(n_components=6).fit_transform(data_orig)
data_seurat_emb = umap.UMAP(n_components=6).fit_transform(data_seurat)
data_scan_emb = umap.UMAP(n_components=6).fit_transform(data_scan)
data_desc_emb = umap.UMAP(n_components=6).fit_transform(data_desc)

print(data_orig_emb.shape)
# data_orig_emb = np.array(data_orig_emb)
np.savetxt(davae_kbet, data_davae, delimiter=',')
np.savetxt(orig_kbet, data_orig_emb, delimiter=',')
np.savetxt(scan_kbet, data_scan_emb, delimiter=',')
np.savetxt(desc_kbet, data_desc_emb, delimiter=',')
np.savetxt(seurat_kbet, data_seurat_emb, delimiter=',')
