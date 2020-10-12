from utils import tools
import numpy as np
from network.da_cvae import DAVAE
import anndata
from sklearn.utils import shuffle
from keras.utils import to_categorical
import scanpy as sc


tech1 = 'ctrl'
tech2 = 'stim'
type = '_8000'
data1, celltype1 = tools.get_ifnb(tech1, type)
data2, celltype2 = tools.get_ifnb(tech2, type)

base_path = '/Users/zhongyuanke/data/'
out_path = 'dann_vae/ifnb/davae_01.h5ad'

batch_size = 256
epochs = 20

orig_len_x = data1.shape[0]
orig_len_y = data2.shape[0]
adata = data1.concatenate(data2)
orig_data = adata.X
# sc.pp.filter_genes(data1, min_cells=3)
# sc.pp.filter_genes(data2, min_cells=3)
# b1 = np.array([0, 1])
# b2 = np.array([1, 0])
# orig_batches = []
# batch_label = []
# loss_weight = []
# for i in range(orig_len_x):
#     orig_batches.append(b1)
#     batch_label.append(0)
#     loss_weight.append(1)
# for i in range(orig_len_y):
#     orig_batches.append(b2)
#     batch_label.append(1)
#     loss_weight.append(0)
orig_batch_label = adata.obs['batch']
orig_batch = to_categorical(orig_batch_label)
loss_weight = orig_batch_label
print(data1.shape)
print(data2.shape)
print(orig_data.shape)

celltype = np.concatenate([celltype1, celltype2])
label = tools.text_label_to_number(celltype)
adata.obs['label'] = label
# adata.write_h5ad(base_path+'dann_vae/ifnb/orig.h5ad')

data, batches, loss_weight = shuffle(orig_data, orig_batch, loss_weight,
                                                         random_state=0)


net_x = DAVAE(input_size=data.shape[1], batches=2)
net_x.build()
net_x.compile()
his = net_x.train(data, batches, loss_weight, epochs=epochs, batch_size=batch_size)

mid = net_x.embedding(orig_data, batches)

adata_mid = anndata.AnnData(X=mid)
adata_mid.obs['batch'] = orig_batch_label
adata_mid.obs['label'] = label
adata_mid.write_h5ad(base_path+out_path)
