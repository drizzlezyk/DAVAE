from utils import tools
import numpy as np
from network.vcca_private import VCCA
import anndata
import scanpy as sc


tech1 = 'celseq2'
tech2 = 'celseq'
data1, celltype_indrop = tools.get_panc8(tech1)
data2, celltype_celseq = tools.get_panc8(tech2)

# base_path = '/Users/zhongyuanke/data/'
base_path = '../data/'
out_path_x = 'vcca_result/panc8/'+tech1+tech2+'_'+tech1+'_output_vcca_01.h5ad'
out_path_y = 'vcca_result/panc8/'+tech1+tech2+'_'+tech2+'_output_vcca_01.h5ad'

mid_path_x = 'vcca_result/panc8/'+tech1+tech2+'_'+tech1+'_vcca_01.h5ad'
mid_path_y = 'vcca_result/panc8/'+tech1+tech2+'_'+tech2+'_vcca_01.h5ad'
result_path = '../vcca/'
batch_size = 120
epochs = 35
# x = x[0:y.shape[0], ]
orig_len_x = data1.shape[0]
orig_len_y = data2.shape[0]
# orig_len_x = 1000
# orig_len_y = 1000
# sc.pp.filter_genes(data1, min_cells=3)
# sc.pp.filter_genes(data2, min_cells=3)
x = data1.X
y = data2.X
# x = x[:1000, ]
# y = y[:1000, ]
print(data1.shape)
print(data2.shape)
x, y = tools.up_sampling(x, y, batch_size)

net_x = VCCA(input_size_x=x.shape[1], inputs_size_y=y.shape[1], batch_size=batch_size, path=result_path)
net_x.build()
net_x.compile()
his = net_x.train(x, y, epochs=epochs, batch_size=batch_size)
z, hx, hy = net_x.integrate_compose(x, y)
# x_mid = np.concatenate([hx, z], axis=1)
# y_mid = np.concatenate([hy, z], axis=1)
y_mid = z
x_mid = z
print(x_mid.shape)

net_y = VCCA(input_size_x=x.shape[1], inputs_size_y=y.shape[1], batch_size=batch_size, path=result_path)
net_y.build()
net_y.compile()
his_y = net_y.train(y, x, epochs=epochs, batch_size=batch_size)
z_y, hx, hy = net_y.integrate_compose(y, x)

y_mid = z_y

x_mid = x_mid[0:orig_len_x, ]
y_mid = y_mid[0:orig_len_y, ]
x_mid = anndata.AnnData(X=x_mid)
x_mid.write_h5ad(base_path+mid_path_x)

y_mid = anndata.AnnData(X=y_mid)
y_mid.write_h5ad(base_path+mid_path_y)

# output_x, output_y = net_x.get_output(x)
# output_x = output_x[0:orig_len_x, ]
# output_y = output_y[0:orig_len_y, ]
#
# output_x = anndata.AnnData(X=output_x)
# output_x.write_h5ad(base_path+out_path_x)
#
# output_y = anndata.AnnData(X=output_y)
# output_y.write_h5ad(base_path+out_path_y)