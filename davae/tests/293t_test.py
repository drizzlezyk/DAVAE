import numpy as np
from utils import tools
from network.da_vae import DAVAE
import anndata
import scanpy as sc
import pandas as pd
from keras.utils import to_categorical
from sklearn.utils import shuffle
from pandas.core.frame import DataFrame


epochs = 25

base_path = '/Users/zhongyuanke/data/'

file1 = base_path+'pbmc/zheng/293t/hg19/'
file2 = base_path+'pbmc/zheng/jurkat/hg19/'
file3 = base_path + 'pbmc/zheng/293t_jurkat_50_50/hg19/'
label_path = base_path + 'pbmc/zheng/293t_jurkat_cluster.txt'
out_path = 'dann_vae/pbmc/293t_01.h5ad'
orig_path = base_path + 'dann_vae/pbmc/orig.h5ad'
result_path = 'da_vae/scxx_result/'

adata1 = sc.read_10x_mtx(file1)
adata2 = sc.read_10x_mtx(file2)
adata3 = sc.read_10x_mtx(file3)


print('finish read')
b1 = np.array([0, 0, 1, 1])
b2 = np.array([0, 1, 1, 0])
b3 = np.array([1, 1, 0, 1])

batches = []
batch_label = []
loss_weight = []
for i in range(adata1.shape[0]):
    batches.append(b1)
    batch_label.append(0)
    loss_weight.append(0)
for i in range(adata2.shape[0]):
    batches.append(b2)
    batch_label.append(1)
    loss_weight.append(0)
for i in range(adata3.shape[0]):
    batches.append(b3)
    batch_label.append(2)
    loss_weight.append(1)
loss_weight = np.array(loss_weight)
orig_batches = np.array(batches)
orig_batch_label = np.array(batch_label)
celltype = tools.get_label_by_txt(label_path)
orig_label = tools.text_label_to_number(celltype)

# print(celltype[9530])
# print(orig_label[9530])
# np.savetxt('/Users/zhongyuanke/data/dann_vae/k_bet/293t/orig_label.csv', orig_label)
# print('finish write')

# adata = adata1.concatenate(adata2, adata3)
# adata = sc.read_csv('/Users/zhongyuanke/data/pbmc/zheng/293t_jurkat_merge_filt.csv')
# adata.obs['batch'] = orig_batch_label
# adata.obs['label'] = orig_label
# adata.obs['celltype'] = celltype
# adata.write_h5ad(orig_path)
# sc.pp.filter_genes(adata, min_cells=1000)
adata = sc.read_h5ad(orig_path)
orig_data = adata.X
data, batch, loss_weight = shuffle(orig_data, orig_batches, loss_weight, random_state=0)

batch_label = np.array(batch_label)
print(batch_label)
print(data.shape)
print(batch.shape)
#
# np.savetxt(base_path+outlabel_path, label, delimiter=",")

net = DAVAE(input_size=data.shape[1], batches=4, path=result_path)
net.build()
net.compile()
history = net.train(data, batch, loss_weight, batch_size=256, epochs=epochs)
x = net.embedding(orig_data, orig_batches)

adata_mid = anndata.AnnData(X=x)
adata_mid.obs['batch'] = orig_batch_label
adata_mid.obs['label'] = orig_label
adata_mid.write_h5ad(base_path+out_path)







