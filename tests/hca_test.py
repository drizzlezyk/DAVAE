from utils import tools
import numpy as np
from network.dann_cvae import DANN_VAE
import anndata
from sklearn.utils import shuffle
import scanpy as sc


base_path = '/Users/zhongyuanke/data/'
out_path = 'dann_vae/hca/davae_01.h5ad'
file1 = base_path+'HCA/ica_cord_blood_h5.h5'
file2 = base_path+'HCA/ica_bone_marrow_h5.h5'

batch_size = 5000
epochs = 6

adata1 = sc.read_10x_h5(file1)
sc.pp.filter_cells(adata1, min_genes=300)
sc.pp.filter_genes(adata1, min_cells=6000)
print(adata1.shape)
adata2 = sc.read_10x_h5(file2)
sc.pp.filter_cells(adata2, min_genes=300)
sc.pp.filter_genes(adata2, min_cells=6000)
adata1.var_names_make_unique()
adata2.var_names_make_unique()
len1 = adata1.shape[0]
len2 = adata2.shape[0]

print(adata1.shape)
print(adata2.shape)
adata = adata1.concatenate(adata2)
adata.write_h5ad('/Users/zhongyuanke/data/dann_vae/hca/orig.h5ad')
print('1')
sc.pp.filter_genes(adata, min_cells=300)
sc.pp.log1p(adata)
orig_data = adata.X.A
# sc.pp.filter_genes(data1, min_cells=3)
# sc.pp.filter_genes(data2, min_cells=3)
b1 = np.array([0, 1])
b2 = np.array([1, 0])
orig_batches = []
orig_batch_label = []
loss_weight = []
for i in range(len1):
    orig_batches.append(b1)
    orig_batch_label.append(0)
    loss_weight.append(0)
for i in range(len2):
    orig_batches.append(b2)
    orig_batch_label.append(1)
    loss_weight.append(1)

orig_batches = np.array(orig_batches)
loss_weight = np.array(loss_weight)
print(adata1.shape)
print(adata2.shape)
print(orig_data.shape)


data, batches, batch_label, loss_weight = shuffle(orig_data, orig_batches, orig_batch_label, loss_weight,
                                                         random_state=0)


net_x = DANN_VAE(input_size=data.shape[1], batches=2)
net_x.build()
net_x.compile()
his = net_x.train(data, batches, loss_weight, epochs=epochs, batch_size=batch_size)

mid = net_x.integrate(orig_data, orig_batches)

adata_mid = anndata.AnnData(X=mid)
adata_mid.obs['batch'] = orig_batch_label
adata_mid.write_h5ad(base_path+out_path)
