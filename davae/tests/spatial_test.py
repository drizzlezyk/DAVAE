from utils import tools
import numpy as np
from network.da_cvae import DAVAE
import anndata
from sklearn.utils import shuffle
from keras.utils import to_categorical
import scanpy as sc


base_path = '/Users/zhongyuanke/data/'
out_path = 'dann_vae/spatial/davae_01.h5ad'
file1 = base_path+'spatial/mouse_brain/10x_mouse_brain_Anterior/V1_Mouse_Brain_Sagittal_Anterior_filtered_feature_bc_matrix.h5'
file2 = base_path+'spatial/mouse_brain/10x_mouse_brain_Posterior/V1_Mouse_Brain_Sagittal_Posterior_filtered_feature_bc_matrix.h5'

batch_size = 256
epochs = 25

adata1 = sc.read_10x_h5(file1)
adata2 = sc.read_10x_h5(file2)
print(adata1)
# adata1.var_names_make_unique()
# adata2.var_names_make_unique()
len1 = adata1.shape[0]
len2 = adata2.shape[0]

# sc.pp.filter_genes(adata, min_cells=300)
sc.pp.log1p(adata1)
sc.pp.log1p(adata2)
print(adata1)
print(adata2)

sc.pp.highly_variable_genes(adata1, n_top_genes=6000)
sc.pp.highly_variable_genes(adata2, n_top_genes=6000)
adata1 = adata1[:, adata1.var.highly_variable]
adata2 = adata2[:, adata2.var.highly_variable]
adata1.write_h5ad('/Users/zhongyuanke/data/spatial/10x_mouse_brain_Anterior/anterior.h5ad')
adata2.write_h5ad('/Users/zhongyuanke/data/spatial/10x_mouse_brain_Posterior/posterior.h5ad')
# del adata1.var['highly_variable']
# del adata2.var['highly_variable']
# del adata1.var['means']
# del adata2.var['means']
# del adata1.var['dispersions']
# del adata2.var['dispersions']
# del adata1.var['dispersions_norm']
# del adata2.var['dispersions_norm']
print(adata1)
print(adata2)
adata = adata1.concatenate(adata2)
print(adata.shape)

orig_data = adata.X.A
orig_batch_label = adata.obs['batch']
orig_batch = to_categorical(orig_batch_label)
data, batches, batch_label = shuffle(orig_data, orig_batch, orig_batch_label, random_state=0)

net_x = DAVAE(input_size=data.shape[1], batches=2)
net_x.build()
net_x.compile()
his = net_x.train(data, batches, batch_label, epochs=epochs, batch_size=batch_size)

mid = net_x.embedding(orig_data, orig_batch)

adata_mid = anndata.AnnData(X=mid)
print(adata_mid.shape)
adata_mid.obs['batch'] = orig_batch_label
adata_mid.write_h5ad(base_path+out_path)
