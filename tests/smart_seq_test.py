import numpy as np
from utils import tools
from network.dann_vae import DANN_VAE
import anndata
import scanpy as sc
import pandas as pd
from keras.utils import to_categorical
from sklearn.utils import shuffle


epochs = 40
base_path = '/Users/zhongyuanke/data/'

orig_path = 'dann_vae/benchmark1/orig.h5ad'
file = base_path+'dann_vae/benchmark1/GSE94820_raw.txt'
out_path = 'dann_vae/benchmark1/davae_01.h5ad'

# adata = sc.read_text(file, delimiter='\t')
# adata = adata.transpose()
# data = adata.X
# barcodes = list(adata.obs.index)
# label = []
# plates = []
# for i in range(len(barcodes)):
#     para_list = barcodes[i].split('_')
#     label.append(para_list[0])
#     plates.append(para_list[1])
# print(label)
# print(plates)
# batch_label = []
# filt_label = []
# filt_data = []
# filt_barcodes = []
# barcodes1 = []
# barcodes2 = []
# batches = []
# batch1_data = []
# batch2_data = []
# for i in range(len(barcodes)):
#     if plates[i] == 'P7' or plates[i] == 'P8' or plates[i] == 'P9' or plates[i] == 'P10':
#         batch_label.append(0)
#         batches.append([0, 1])
#         filt_label.append(label[i])
#         filt_data.append(data[i, ])
#         filt_barcodes.append(barcodes[i])
#         batch1_data.append(data[i])
#         barcodes1.append(barcodes[i])
#     elif plates[i] == 'P3' or plates[i] == 'P4' or plates[i] == 'P13' or plates[i] == 'P14':
#         batch_label.append(1)
#         batches.append([1, 0])
#         filt_label.append(label[i])
#         filt_data.append(data[i, ])
#         filt_barcodes.append(barcodes[i])
#         batch2_data.append(data[i])
#         barcodes2.append(barcodes[i])
# filt_data = np.array(filt_data)
# batch_label = np.array(batch_label)
# batches = np.array(batches)
# filt_label = np.array(filt_label)
#
# adata_orig = anndata.AnnData(X=filt_data)
# filt_barcodes = pd.DataFrame(index=filt_barcodes)
# adata_orig.obs = filt_barcodes
# adata_orig.var = adata.var
# adata_orig.obs['batch'] = batch_label
# filt_label = tools.text_label_to_number(filt_label)
# print(filt_label.shape)
# adata_orig.obs['label'] = filt_label
# # adata_orig.write_h5ad(base_path+'dann_vae/benchmark1/orig.h5ad')
#
# batch1_data = np.array(batch1_data).transpose()
# batch2_data = np.array(batch2_data).transpose()
# var = list(adata.var.index)
#
#
# batch1 = pd.DataFrame(columns=barcodes1, data=batch1_data, index=var)
# batch1.to_csv(base_path+'dann_vae/benchmark1/batch1.csv')
# batch2 = pd.DataFrame(columns=barcodes2, data=batch2_data, index=var)
# batch2.to_csv(base_path+'dann_vae/benchmark1/batch2.csv')

adata = sc.read_h5ad(base_path+orig_path)
sc.pp.filter_genes(adata, min_cells=60)
sc.pp.log1p(adata)
# sc.pp.highly_variable_genes(adata,min_mean=0.001, max_mean=5, min_disp=0.5)
# adata = adata[:, adata.var.highly_variable]
label = adata.obs['label']
label = tools.text_label_to_number(label)
batch_label = adata.obs['batch']
loss_weights = []
for i in range(384):
    loss_weights.append(5)
for i in range(384):
    loss_weights.append(0)
loss_weights = np.array(loss_weights)
data = adata.X
batches = to_categorical(batch_label)

data, batch, label, batch_label, loss_weights = shuffle(data, batches, label, batch_label, loss_weights, random_state=0)
#
# batch_label = np.array(batch_label)
print(adata.shape)
# #
# # np.savetxt(base_path+outlabel_path, label, delimiter=",")
#
net = DANN_VAE(input_size=data.shape[1], batches=2)
net.build()
net.compile()
net.train(data, batch, loss_weights, batch_size=256, epochs=epochs)
x = net.integrate(data, batch)

adata_mid = anndata.AnnData(X=x)
adata_mid.obs = adata.obs
adata_mid.obs['batch'] = batch_label
adata_mid.obs['label'] = label

adata_mid.write_h5ad(base_path+out_path)







