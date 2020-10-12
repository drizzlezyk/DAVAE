import numpy as np
from utils import tools
from network.da_cvae import DAVAE
import anndata
import scanpy as sc
import pandas as pd
from keras.utils import to_categorical
from sklearn.utils import shuffle
from pandas.core.frame import DataFrame


epochs = 20
base_path = '/Users/zhongyuanke/data/'

# file1 = base_path+'trajectory/monocytes/mcsf_day3_1.txt'
# file2 = base_path+'trajectory/monocytes/mcsf_day3_2.txt'
# file3 = base_path + 'trajectory/monocytes/mcsf_day6_1.txt'
# file4 = base_path + 'trajectory/monocytes/mcsf_day6_2.txt'
# file5 =
# file1 = base_path + 'trajectory/DESC_GSE146974/mh001/'
# file2 = base_path + 'trajectory/DESC_GSE146974/rp002/'
# file3 = base_path + 'trajectory/DESC_GSE146974/rp009/'

file1 = base_path + 'trajectory/LSP_PAM/lps_local.csv'
file2 = base_path + 'trajectory/LSP_PAM/pam_local.csv'
traj1_path = base_path + 'trajectory/LSP_PAM/pam_local_traj.csv'
traj2_path = base_path + 'trajectory/LSP_PAM/lps_local_traj.csv'
out_path = 'dann_vae/trajectory/293t_01.h5ad'
orig_path = base_path + 'dann_vae/trajectory/orig.h5ad'

adata1 = sc.read_csv(file1)
adata2 = sc.read_csv(file2)

traj1 = np.genfromtxt(traj1_path, delimiter=",", skip_header=True)
traj2 = np.genfromtxt(traj2_path, delimiter=",", skip_header=True)
traj = np.concatenate([traj1, traj2])
traj = traj[:, 1]
print('finish read')

adata = adata1.concatenate(adata2)
print(adata.shape)
print(traj.shape)
adata.obs['traj'] = traj
batch_label = adata.obs['batch']
orig_batch = to_categorical(batch_label)
# adata = sc.read_csv('/Users/zhongyuanke/data/pbmc/zheng/293t_jurkat_merge_filt.csv')
# adata.obs['batch'] = orig_batch_label
# adata.obs['label'] = orig_label
# adata.obs['celltype'] = celltype
# adata.write_h5ad(orig_path)
# sc.pp.filter_genes(adata, min_cells=1000)
adata.write_h5ad(orig_path)

orig_data = adata.X
data, batch, loss_weight = shuffle(orig_data, orig_batch, batch_label, random_state=0)
batch_label = np.array(batch_label)
print(batch_label)
print(data.shape)
print(batch.shape)
#
# np.savetxt(base_path+outlabel_path, label, delimiter=",")

net = DAVAE(input_size=data.shape[1], batches=2)
net.build()
net.compile()
history = net.train(data, batch, loss_weight, batch_size=64, epochs=epochs)
x = net.embedding(orig_data, orig_batch)

adata.obsm['davae']=x

adata.write_h5ad(base_path+out_path)







