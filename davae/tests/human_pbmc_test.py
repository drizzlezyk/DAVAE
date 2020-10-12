import numpy as np
from utils import tools
from network.da_cvae import DAVAE
import anndata
import scanpy as sc
from keras.utils import to_categorical
from sklearn.utils import shuffle


epochs = 20
base_path = '/Users/zhongyuanke/data/'

file = base_path+'pbmc/human_two_batch/PBMC.merged.h5ad'

inter_file1 = base_path + 'dann_vae/pbmc/human_5.h5ad'
inter_file2 = base_path + 'dann_vae/pbmc/human_3.h5ad'
# inter_adata1 = sc.read_h5ad(inter_file1)
# inter_adata2 = sc.read_h5ad(inter_file2)
# inter_data1 = inter_adata1.X
# inter_data2 = inter_adata2.X

out_path = 'dann_vae/human/davae_01.h5ad'

adata = sc.read(file)
print(adata.shape)
data = adata.X
# data = np.concatenate([inter_data1, inter_data2], axis=0)
print(adata.obs_keys())
batch_label = adata.obs['batch']
celltype = adata.obs['Cell type']
batch = to_categorical(batch_label)
print(batch)
label = tools.text_label_to_number(celltype)

data, batch, label, batch_label = shuffle(data, batch, label, batch_label, random_state=0)

batch_label = np.array(batch_label)
print(batch_label)
print(data.shape)
print(batch.shape)
#
# np.savetxt(base_path+outlabel_path, label, delimiter=",")

net = DAVAE(input_size=data.shape[1], batches=2)
net.build()
net.compile()
net.train(data, batch,batch_label, batch_size=256, epochs=epochs)
x = net.embedding(data, batch)

adata_mid = anndata.AnnData(X=x)
adata_mid.obs['batch'] = batch_label
adata_mid.obs['label'] = label
adata_mid.write_h5ad(base_path+out_path)







