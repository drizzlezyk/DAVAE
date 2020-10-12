import scanpy as sc
from utils import tools, plot
import csv
import numpy as np
import argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import umap
from utils import tools as pp
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--dim_red", type=int, default=1, help="method for dimension reduction, 1 for umap, 0 for tsne, 2 "
                                                           "for pca")
parser.add_argument("--x_filetype", type=str, default='10x_mtx', help="file type of x: 10x_mtx/h5ad/matrix.mtx")
parser.add_argument("--y_filetype", type=str, default='10x_mtx', help="file type of x: 10x_mtx/h5ad/matrix.mtx")
opt = parser.parse_args()

base_path = '../data/'
tech1 = 'indrop'
tech2 = 'celseq2'
data_x, celltype_x = tools.get_panc8(tech1)
data_y, celltype_y = tools.get_panc8(tech2)

orig_x = data_x.X
orig_y = data_y.X

file_vcca_x = base_path + 'vcca_result/panc8/'+tech1+tech2+'_'+tech1+'_output_vcca_01.h5ad'
file_vcca_y = base_path + 'vcca_result/panc8/'+tech1+tech2+'_'+tech2+'_output_vcca_01.h5ad'

adata_vcca_x = sc.read_h5ad(file_vcca_x)
adata_vcca_y = sc.read_h5ad(file_vcca_y)
vcca_x = adata_vcca_x.X
vcca_y = adata_vcca_y.X

for i in range(0, 1000, 100):
    save_path = base_path + 'vcca_result/panc8/' + tech1 + tech2 + '_loss_' + str(i) + '.png'
    fig = plt.figure()
    plt.scatter(orig_x[i], vcca_x[i], c='darkseagreen', s=3, linewidth=0)
    plt.plot([-1, 3], [-1, 3], c='r')
    plt.xlabel('orig')
    plt.ylabel('vcca')
    plt.savefig(save_path)