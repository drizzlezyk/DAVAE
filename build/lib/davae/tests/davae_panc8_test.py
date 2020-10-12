import numpy as np
from utils import tools
from network.da_cvae import DAVAE
import anndata
import scanpy as sc
from sklearn.utils import shuffle


epochs = 18
base_path = '/Users/zhongyuanke/data/'

result_path = '../scxx_result/'

out_path = 'dann_vae/panc8/panc8_merge5.h5ad'

merge_path = 'dann_vae/panc8/inter2_merge.h5ad'
# file = 'merge_result/293t_jurkat.h5ad'

tech1 = 'indrop'
tech2 = 'celseq2'
tech3 = 'celseq'
tech4 = 'smartseq2'
tech5 = 'fluidigmc1'
type = '_2000'
file_path1 = base_path + 'dann_vae/panc8/inter5_indrop.h5ad'
file_path2 = base_path + 'dann_vae/panc8/inter5_celseq2.h5ad'
file_path3 = base_path + 'dann_vae/panc8/inter5_celseq.h5ad'
file_path4 = base_path + 'dann_vae/panc8/inter5_smartseq2.h5ad'
file_path5 = base_path + 'dann_vae/panc8/inter5_fluidigmc1.h5ad'

# adata2 = tools.get_intersection_panc8(tech2)
# adata3 = tools.get_intersection_panc8(tech3)

adata1, celltype1 = tools.get_panc8(tech1, type)
adata2, celltype2 = tools.get_panc8(tech2, type)
adata3, celltype3 = tools.get_panc8(tech3, type)
adata4, celltype4 = tools.get_panc8(tech4, type)
adata5, celltype5 = tools.get_panc8(tech5, type)

#
adata_all = sc.read_h5ad(base_path + merge_path)

data1 = sc.read_h5ad(file_path1).X
data2 = sc.read_h5ad(file_path2).X
data3 = sc.read_h5ad(file_path3).X
data4 = sc.read_h5ad(file_path4).X
data5 = sc.read_h5ad(file_path5).X
b1 = np.array([0, 1])
b2 = np.array([1, 0])
# b1 = np.array([1, 0, 0, 0, 0, 1])
# b2 = np.array([0, 1, 0, 0, 0, 0])
# b3 = np.array([0, 0, 1, 0, 0, 0])
# b4 = np.array([0, 0, 0, 1, 0, 0])
# b5 = np.array([0, 0, 0, 0, 1, 0])

# b1 = np.array([1, 0, 0, 0, 1])
# b3 = np.array([0, 1, 0, 0, 0])
# b4 = np.array([0, 0, 1, 0, 0])
# b5 = np.array([0, 0, 0, 1, 0])
# b1 = np.array([0, 0, 1, 1])
# b4 = np.array([0, 1, 1, 0])
# b5 = np.array([1, 1, 0, 1])

# data1 = adata1.X
# data2 = adata2.X
# data3 = adata3.X
# data4 = adata4.X
# data5 = adata5.X

orig_batches = []
batch_label = []
loss_weight = []
for i in range(adata1.shape[0]):
    orig_batches.append(b1)
    batch_label.append(0)
    loss_weight.append(1)
# for i in range(adata2.shape[0]):
#     orig_batches.append(b2)
#     batch_label.append(1)
# for i in range(adata3.shape[0]):
#     orig_batches.append(b3)
#     batch_label.append(2)
#     loss_weight.append(1)
for i in range(adata4.shape[0]):
    orig_batches.append(b2)
    batch_label.append(3)
    loss_weight.append(0)
# for i in range(adata5.shape[0]):
#     orig_batches.append(b5)
#     batch_label.append(4)
#     loss_weight.append(1)
orig_batches = np.array(orig_batches)
loss_weight = np.array(loss_weight)


# celltype = np.concatenate([celltype2, celltype3])
celltype = np.concatenate([celltype1, celltype4])
# celltype = np.concatenate([celltype1, celltype2])
label = tools.text_label_to_number(celltype)

# adata_all = adata2.concatenate(adata3)
# adata_all.obs['label'] = label
# adata_all.write_h5ad('/Users/zhongyuanke/data/dann_vae/panc8/orig_cel2_cel.h5ad')
# print('end')
# data = np.concatenate([data1, data3, data4, data5], axis=0)
data = adata_all.X
total_len = data.shape[0]
data, batches, label, batch_label, loss_weight = shuffle(data, orig_batches, label, batch_label, loss_weight,
                                                         random_state=0)

# add = data[0:11, ]
# add_batch = []
# for i in range(11):
#     add_batch.append([0, 0])
# add_batch = np.array(add_batch)

# data = np.concatenate([data, add], axis=0)
# batches = np.concatenate([batches, add_batch], axis=0)

# batch_label = np.array(batch_label)
print(data.shape)
print(batches.shape)

# b3 = np.array([0, 0, 1])
# x = [b1, b2, b3]


# x = [b1, b2, b3, b4, b5]

net = DAVAE(input_size=adata_all.shape[1], batches=2, path=result_path)
net.build()
net.compile()
history = net.train(data, batches, loss_weight, batch_size=256, epochs=epochs)
x = net.embedding(data, batches)

data_out = []
label_out = []
data_out_indrop = []
label_indrop = []
for i in range(len(batch_label)):
    if batch_label[i]!=0:
        data_out.append(x[i, ])
        label_out.append(label[i])
    else:
        data_out_indrop.append(x[i, ])
        label_indrop.append(label[i])
data_out_indrop = np.array(data_out_indrop)
data_out = np.array(data_out)
label_indrop = np.array(label_indrop)
label_out = np.array(label_out)
print(data_out.shape, len(label_out))

adata_mid = anndata.AnnData(X=x)
adata_mid.obs['batch'] = batch_label
adata_mid.obs['label'] = label
adata_mid.write_h5ad(base_path+out_path)
#
# adata_indrop = anndata.AnnData(X=data_out_indrop)
# adata_indrop.obs['label'] = label_indrop
# adata_indrop.write_h5ad(base_path+'dann_vae/panc8/indrop_out.h5ad')
#
# adata_other = anndata.AnnData(X=data_out)
# adata_other.obs['label'] = label_out
# adata_other.write_h5ad(base_path+'dann_vae/panc8/celseq2_out.h5ad')




