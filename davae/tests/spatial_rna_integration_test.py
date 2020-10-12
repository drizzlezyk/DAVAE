from utils import tools
import numpy as np
from network.da_cvae import DAVAE
import anndata
from sklearn.utils import shuffle
from keras.utils import to_categorical
import scanpy as sc


base_path = '/Users/zhongyuanke/data/'
anterior_out_path = 'dann_vae/spatial/rna_anterior_davae_01.h5ad'
posterior_out_path = 'dann_vae/spatial/rna_posterior_davae_01.h5ad'
file1 = base_path+'spatial/mouse_brain/10x_mouse_brain_Anterior/V1_Mouse_Brain_Sagittal_Anterior_filtered_feature_bc_matrix.h5'
file2 = base_path+'spatial/mouse_brain/10x_mouse_brain_Posterior/V1_Mouse_Brain_Sagittal_Posterior_filtered_feature_bc_matrix.h5'
file_rna = base_path+'spatial/mouse_brain/adata_processed_sc.h5ad'
rna_anterior_orig = base_path+'dann_vae/spatial/rna_anterior_orig.h5ad'
batch_size = 256
epochs = 25

adata1 = sc.read_10x_h5(file1)
adata2 = sc.read_10x_h5(file2)
adata_rna = sc.read_h5ad(file_rna)
len1 = adata1.shape[0]
len2 = adata2.shape[0]
len_rna = adata_rna.shape[0]

adata1.var_names_make_unique()
adata2.var_names_make_unique()
adata_rna.var_names_make_unique()

sc.pp.log1p(adata1)
sc.pp.log1p(adata2)
sc.pp.log1p(adata_rna)


sc.pp.highly_variable_genes(adata1, n_top_genes=8000)
sc.pp.highly_variable_genes(adata2, n_top_genes=8000)
sc.pp.highly_variable_genes(adata_rna, n_top_genes=8000)
adata1 = adata1[:, adata1.var.highly_variable]
adata2 = adata2[:, adata2.var.highly_variable]
adata_rna = adata_rna[:, adata_rna.var.highly_variable]

genes1 = adata1.var.index.values
genes2 = adata2.var.index.values
genes_rna = adata_rna.var.index.values
genes_rna = genes_rna.tolist()
print(genes1)
print(genes_rna)
# adata = adata1.concatenate(adata_rna)

data1, data_rna, genes = tools.intersection_genes(adata2.X.A, adata_rna.X, genes2, genes_rna)
print(len(genes))
orig_data = np.concatenate([data_rna, data1])
print(orig_data.shape)


orig_batch_label = []
for i in range(len_rna):
    orig_batch_label.append(1)
for i in range(len2):
    orig_batch_label.append(0)
orig_batch_label = np.array(orig_batch_label)
orig_batch = to_categorical(orig_batch_label)

data, batches, batch_label = shuffle(orig_data, orig_batch, orig_batch_label, random_state=0)

net_x = DAVAE(input_size=data.shape[1], batches=2)
net_x.build()
net_x.compile()
his = net_x.train(data, batches, batch_label, epochs=epochs, batch_size=batch_size)

mid = net_x.embedding(orig_data, orig_batch)

adata_davae = anndata.AnnData(X=mid)
adata_davae.obs['batch'] = orig_batch_label
adata_davae.write_h5ad(base_path+posterior_out_path)
