from network import da_cvae
from utils import tools, plot
import numpy as np
import umap

tech1 = 'ctrl'
tech2 = 'stim'
type = '_8000'
data1, celltype1 = tools.get_ifnb(tech1, type)
data2, celltype2 = tools.get_ifnb(tech2, type)

data_list = [data1, data2]
adata_out = da_cvae.integration(data_list, epoch=19, latent_size=6)

celltype = np.concatenate([celltype1, celltype2])
label = tools.text_label_to_number(celltype)
adata_out.obs['label'] = label

embedding = adata_out.obsm['davae']
data_umap = umap.UMAP().fit_transform(embedding)
adata_out.obsm['umap'] = data_umap
plot.plot_davae_result_labeled(adata_out, cmap_batch='tab20b', cmap_label='Set3')
