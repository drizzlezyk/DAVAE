from utils import tools
import numpy as np
from network.deep_classifier import CLASSIFIER
import anndata
from keras.utils import to_categorical
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.neighbors import NearestNeighbors

base_path = '/Users/zhongyuanke/data/'
pred_label_path = base_path + 'dann_vae/atac/pred_label.csv'
pred_type_path = base_path + 'dann_vae/atac/pred_type.csv'

file1 = base_path+'seurat_data/sc_atac/atac_v1_pbmc_10k_filtered_peak_bc_matrix_8000.csv'
file2 = base_path+'seurat_data/sc_atac/pbmc_10k_v3_8000.csv'
davae_path = base_path+'dann_vae/atac/davae_01.h5ad'

celltype_x_path = base_path + 'seurat_data/sc_atac/pbmc_10k_v3_celltype.csv'
celltype_x = pd.read_csv(celltype_x_path, index_col=0)
celltype_x = celltype_x.values

encoder = LabelEncoder()
orig_label = encoder.fit_transform(celltype_x)

batch_size = 100
epochs = 25

adata1 = sc.read_csv(file1)
adata2 = sc.read_csv(file2)
adata_davae = sc.read_h5ad(davae_path)
data = adata_davae.X

len1 = adata1.shape[0]
len2 = adata2.shape[0]

test_set = data[0:len1, ]
train_set = data[len1:len1+len2, ]

label = to_categorical(orig_label)
class_num = label.shape[1]
print(label.dtype)
print(test_set.shape)
print(train_set.shape)
net_x = CLASSIFIER(input_size=train_set.shape[1], class_num=class_num)
net_x.build()
net_x.compile()
his = net_x.train(x=train_set, label=label, epochs=epochs, batch_size=batch_size)

pred_label = net_x.prediction(test_set)
print(pred_label.shape)

pred_type = encoder.inverse_transform(pred_label)

df = pd.DataFrame(pred_type)
df.to_csv(pred_type_path)
np.savetxt(pred_label_path, pred_label, delimiter=',')

list1 = [0,1,2,3,4,5,6,7,8,9,10,11,12]
label1 = encoder.inverse_transform(list1)
print(label1)

all_label = np.concatenate([pred_label, orig_label])
adata_orig = sc.read_h5ad('/Users/zhongyuanke/data/dann_vae/atac/orig.h5ad')
adata_orig.obs['label'] = all_label
adata_orig.write_h5ad('/Users/zhongyuanke/data/dann_vae/atac/orig.h5ad')
