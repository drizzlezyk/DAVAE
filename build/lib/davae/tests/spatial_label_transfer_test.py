from sklearn.metrics.pairwise import cosine_distances
import scanpy as sc
import anndata as an
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def label_transfer(dist, labels):
    lab = pd.get_dummies(labels).to_numpy().T
    class_prob = lab @ dist
    norm = np.linalg.norm(class_prob, 2, axis=0)
    class_prob = class_prob / norm
    class_prob = (class_prob.T - class_prob.min(1)) / class_prob.ptp(1)
    return class_prob


base_path = '/Users/zhongyuanke/data/'
davae_anterior_path = base_path + 'dann_vae/spatial/rna_anterior_davae_01.h5ad'
davae_posterior_path = base_path + 'dann_vae/spatial/rna_posterior_davae_01.h5ad'
file1_spatial = base_path+'spatial/mouse_brain/10x_mouse_brain_Anterior/'
file2_spatial = base_path+'spatial/mouse_brain/10x_mouse_brain_Posterior/'
file1 = base_path+'spatial/mouse_brain/10x_mouse_brain_Anterior/V1_Mouse_Brain_Sagittal_Anterior_filtered_feature_bc_matrix.h5'
file2 = base_path+'spatial/mouse_brain/10x_mouse_brain_Posterior/V1_Mouse_Brain_Sagittal_Posterior_filtered_feature_bc_matrix.h5'
file_rna = base_path+'spatial/mouse_brain/adata_processed_sc.h5ad'

# adata_anterior = sc.read_visium(file1_spatial,count_file=file1)
adata_anterior = sc.read_visium(file2_spatial,count_file=file2)

adata_davae = sc.read_h5ad(davae_posterior_path)
adata_rna = sc.read_h5ad(file_rna)

len_anterior = adata_anterior.shape[0]
# len_posterior = adata_posterior.shape[0]
len_rna = adata_rna.shape[0]
davae_emb = adata_davae.X

adata_rna.obsm["davae_embedding"] = davae_emb[0:len_rna, :]
print(adata_anterior)
# adata_posterior.obsm["davae_embedding"] = davae_emb[len_rna:len_rna+len_posterior, :]
adata_anterior.obsm['davae_embedding'] = davae_emb[len_rna:len_rna+len_anterior, :]

distances_anterior = 1 - cosine_distances(
    adata_rna.obsm["davae_embedding"],
    adata_anterior.obsm['davae_embedding'],
)
# distances_posterior = 1 - cosine_distances(
#     adata_rna.obsm["davae_embedding"],
#     adata_anterior.obsm['davae_embedding'],
# )
class_prob_anterior = label_transfer(distances_anterior, adata_rna.obs.cell_subclass)

# class_prob_posterior = label_transfer(
#     distances_posterior, adata_rna.obs.cell_subclass
# )

cp_anterior_df = pd.DataFrame(
    class_prob_anterior, columns=np.sort(adata_rna.obs.cell_subclass.unique())
)
# cp_posterior_df = pd.DataFrame(
#     class_prob_posterior, columns=np.sort(adata_cortex.obs.cell_subclass.unique())
# )

cp_anterior_df.index = adata_anterior.obs.index
adata_anterior_transfer = adata_anterior.copy()
adata_anterior_transfer.obs = pd.concat(
    [adata_anterior.obs, cp_anterior_df], axis=1
)
print(adata_anterior_transfer)
sc.pl.spatial(
    adata_anterior_transfer,
    img_key="hires",
    color=["L2/3 IT", "L4", "L5 PT", "L6 CT"],
    size=1.5,
)
# adata_posterior_subset_transfer = adata_posterior_subset.copy()
# adata_posterior_subset_transfer.obs = pd.concat(
#     [adata_posterior_subset.obs, cp_posterior_df], axis=1
# )


