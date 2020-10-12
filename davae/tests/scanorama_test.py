import scanorama
import scanpy as sc
from utils import tools
import anndata


tech1 = 'celseq2'
tech2 = 'celseq'

base_path = '/Users/zhongyuanke/data/'
out_path_x = base_path + 'vcca/panc8/'+tech1+tech2+'_'+tech1+'_scanorama_01.h5ad'
out_path_y = base_path + 'vcca/panc8/'+tech1+tech2+'_'+tech2+'_scanorama_01.h5ad'
out_merge_path = base_path + 'scanorama/panc8'+'_'+tech1+tech2+'.h5ad'

# file1 = base_path+'pbmc/zheng/293t/hg19/'
# file2 = base_path+'pbmc/zheng/jurkat/hg19/'
# file3 = base_path + 'pbmc/zheng/293t_jurkat_50_50/hg19/'

orig_path = 'dann_vae/benchmark1/orig.h5ad'
file1 = base_path+'dann_vae/benchmark1/batch1.h5ad'
file2 = base_path+'dann_vae/benchmark1/batch2.h5ad'
adata1 = sc.read_h5ad(file1)
adata2 = sc.read_h5ad(file2)
# adata3 = sc.read_10x_mtx(file3)

# adata1, celltype_indrop = tools.get_panc8(tech1)
# adata2, celltype_celseq = tools.get_panc8(tech2)

tech1 = 'ctrl'
tech2 = 'stim'
type = '_8000'
# adata1, celltype1 = tools.get_ifnb(tech1, type)
# adata2, celltype2 = tools.get_ifnb(tech2, type)

base_path = '/Users/zhongyuanke/data/'
out_path = 'scanorama/smart_seq.h5ad'
# sc.pp.filter_genes(adata1, min_cells=10)
# sc.pp.filter_genes(adata2, min_cells=10)

datas = [adata1, adata2]
# datas = [adata1, adata2, adata3]
print(adata1.shape)
print(adata2.shape)
# print(adata3.shape)
integrated, corrected = scanorama.correct_scanpy(datas, return_dimred=True)

out1 = anndata.AnnData(integrated[0])
out2 = anndata.AnnData(integrated[1])
# out3 = anndata.AnnData(integrated[2])

adata = out1.concatenate(out2)
print(adata.shape)
adata.write_h5ad(base_path+out_path)
