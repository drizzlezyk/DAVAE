import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams


base_path = '/Users/zhongyuanke/data/'
orig_path = base_path + 'dann_vae/trajectory/orig.h5ad'
file = base_path + 'dann_vae/trajectory/293t_01.h5ad'
figure_gene_variable = base_path + 'dann_vae/trajectory/high_vari_gene.png'
figure_paga = base_path + 'dann_vae/trajectory/paga_davae.png'
figure_graph = base_path + 'dann_vae/trajectory/graph_davae.png'

adata = sc.read_h5ad(file)
sc.pp.scale(adata)
# sc.pp.highly_variable_genes(adata, n_top_genes=10)
# sc.pl.highest_expr_genes(adata, n_top=10)
# sc.pp.neighbors(adata, n_neighbors=4, n_pcs=20

# sc.pp.neighbors(adata)
sc.pp.neighbors(adata, use_rep='davae')
sc.tl.draw_graph(adata)
sc.pl.draw_graph(adata, color='traj', legend_loc='on data', size=18, title='trajectory')
sc.pl.draw_graph(adata, color='batch', legend_loc='on data', size=18, title='batch')
#
sc.tl.louvain(adata, resolution=1.0)
sc.tl.paga(adata, groups='louvain')
# sc.pl.paga(adata, color=['louvain', 'NFKBIA', 'CXCL2', 'TNFAIP2','CD44','IL1A','TNF'])
# sc.pl.paga(adata, color=['FPR1', 'TREM1', 'CLCN7', 'MALT1','PSTPIP2','TSHZ1','RELA'])
# sc.pl.paga(adata, color=['RALGDS', 'SLC25A25', 'ZEB2', 'BCL2L11','IL1A','FLRT3','PLAGL2'])
# sc.pl.paga(adata, color=['TNFAIP3', 'TGM2', 'TOP1', 'PTX3','PIP5K1A','TLR2','SGMS2'])
# sc.pl.paga(adata, color=['PDE4B', 'TLR2', 'CLEC4D', 'PILRA','INSIG1', 'SLC16A10', 'MCOLN2'])
# sc.pl.paga(adata, color=['PTAFR', 'INSIG1', 'CXCL1', 'CXCL2','NUP54', 'RASGEF1B', 'NIACR1'])
sc.pl.paga(adata, color=['ORAI2', 'PILRA', 'FAM20C', 'OSBPL3','ICOSL', 'IRAK-2', 'CLEC4D'])

# sc.tl.draw_graph(adata, init_pos='paga')
# sc.pl.draw_graph(adata, color=['louvain',  'FPR1'], legend_loc='on data')