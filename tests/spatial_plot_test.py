from utils import tools
import numpy as np
from network.dann_cvae import DANN_VAE
import anndata
from sklearn.utils import shuffle
from keras.utils import to_categorical
import scanpy as sc
import matplotlib.pyplot as plt

base_path = '/Users/zhongyuanke/data/'
out_path = 'dann_vae/spatial/davae_01.h5ad'
file1_spatial = base_path+'spatial/10x_mouse_brain_Anterior/'
file2_spatial = base_path+'spatial/10x_mouse_brain_Posterior/'
file1 = base_path+'spatial/10x_mouse_brain_Anterior/V1_Mouse_Brain_Sagittal_Anterior_filtered_feature_bc_matrix.h5'
file2 = base_path+'spatial/10x_mouse_brain_Posterior/V1_Mouse_Brain_Sagittal_Posterior_filtered_feature_bc_matrix.h5'
figure_umap = base_path+'dann_vae/spatial/umap.png'

adata_spatial_anterior = sc.read_visium(file1_spatial,count_file=file1)
adata_spatial_posterior = sc.read_visium(file2_spatial,count_file=file2)
adata_spatial_anterior.var_names_make_unique()
adata_spatial_posterior.var_names_make_unique()
for adata in [
    adata_spatial_anterior,
    adata_spatial_posterior,
]:
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=4000, inplace=True)

adata_spatial = adata_spatial_anterior.concatenate(
    adata_spatial_posterior,
    batch_key="library_id",
    uns_merge="unique",
    batch_categories=[
        k
        for d in [
            adata_spatial_anterior.uns["spatial"],
            adata_spatial_posterior.uns["spatial"],
        ]
        for k, v in d.items()
    ],
)

embedding_adata = sc.read_h5ad(base_path+'dann_vae/spatial/davae_01.h5ad')
embedding = embedding_adata.X
adata_spatial.obsm["davae_embedding"] = embedding
sc.pp.neighbors(adata_spatial,use_rep='davae_embedding')
sc.tl.umap(adata_spatial)
sc.tl.leiden(adata_spatial, key_added="clusters")
sc.pl.umap(
    adata_spatial, color=["clusters", "library_id"], palette=sc.pl.palettes.default_20
)



sc.set_figure_params(facecolor="white", figsize=(8, 8))
clusters_colors = dict(
    zip([str(i) for i in range(18)], adata_spatial.uns["clusters_colors"])
)
fig, axs = plt.subplots(1, 2, figsize=(15, 10))

for i, library in enumerate(
    ["V1_Mouse_Brain_Sagittal_Anterior", "V1_Mouse_Brain_Sagittal_Posterior"]
):
    ad = adata_spatial[adata_spatial.obs.library_id == library, :].copy()
    sc.pl.spatial(
        ad,
        img_key="hires",
        library_id=library,
        color="clusters",
        size=1.5,
        palette=[
            v
            for k, v in clusters_colors.items()
            if k in ad.obs.clusters.unique().tolist()
        ],
        legend_loc=None,
        show=False,
        ax=axs[i],
    )

plt.tight_layout()
plt.savefig('/Users/zhongyuanke/data/dann_vae/spatial/visual.png')