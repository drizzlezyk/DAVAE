
import scanpy as sc

from pandas.core.frame import DataFrame

base_path = '/Users/zhongyuanke/data/'

file1 = base_path+'pbmc/zheng/293t/hg19/'
file2 = base_path+'pbmc/zheng/jurkat/hg19/'
file3 = base_path + 'pbmc/zheng/293t_jurkat_50_50/hg19/'

# adata1 = sc.read_10x_mtx(file1)
# adata2 = sc.read_10x_mtx(file2)
adata3 = sc.read_10x_mtx(file3)

obs = list(adata3.obs.index)
adata3.obs['barcodes'] = obs
print(adata3)

# adata2.write_h5ad('/Users/zhongyuanke/data/pbmc/zheng/293t.h5ad')
# adata2.write_h5ad('/Users/zhongyuanke/data/pbmc/zheng/jurkat.h5ad')
adata3.write_h5ad('/Users/zhongyuanke/data/pbmc/zheng/293t_jurkat_50_50.h5ad')