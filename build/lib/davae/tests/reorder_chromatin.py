import utils.tools as pre
import anndata
import argparse
import csv
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument("--base_path", type=str, default='/Users/zhongyuanke/data/', help="base path")
parser.add_argument("--y_filetype", type=str, default='csv', help="file type of x: 10x_mtx/h5ad/txt/csv")
opt = parser.parse_args()

ref_barcodes_path = 'atac/p0_BrainCortex/dna/norm_count_matrix.csv'
file_x = 'atac/p0_BrainCortex/chromatin/orig_norm_count_matrix.csv'
barcodes_path = 'atac/p0_BrainCortex/chromatin/barcodes.csv'
reorder_outpath = 'atac/p0_BrainCortex/chromatin/orig_norm_count_matrix_reorder.h5ad'

adata_x = pre.read_sc_data(opt.base_path+file_x, fmt=opt.y_filetype)

with open(opt.base_path+barcodes_path,'r') as csvfile:
    reader = csv.reader(csvfile)
    barcodes = [row[1] for row in reader]
del(barcodes[0])
print(barcodes)

with open(opt.base_path+ref_barcodes_path,'r') as csvfile:
    reader = csv.reader(csvfile)
    ref_barcodes = [row[0] for row in reader]
del(ref_barcodes[0])
print(ref_barcodes)


x = adata_x.X


chromatin = []
for i in range(adata_x.shape[0]):
    index = pre.locate_sample(barcodes, ref_barcodes[i])
    chromatin.append(x[index, ])
chromatin = np.array(chromatin)

print(chromatin[0])
print(x[174])
adata_out = anndata.AnnData(X=chromatin)
adata_out.var = adata_x.var
adata_out.obs = adata_x.obs
adata_out.write_h5ad(opt.base_path + reorder_outpath)


