from davae.network import da_cvae
import numpy as np
import umap
import time
import scanpy as sc
import desc

import os
import psutil
import argparse
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_versions()

parser = argparse.ArgumentParser()
parser.add_argument("--base_path", type=str, default='/Users/zhongyuanke/data/dann_vae/scalability/', help="base path")

opt = parser.parse_args()

base_path = opt.base_path
file1 = base_path+'blood_5w.h5ad'
file2 = base_path+'bone_5w.h5ad'

time_list = []

adata1 = sc.read_h5ad(base_path+'blood_5w.h5ad')
adata2 = sc.read_h5ad(base_path+'bone_5w.h5ad')
adata = adata1.concatenate(adata2)
t0 = time.time()
desc.normalize_per_cell(adata, counts_per_cell_after=1e4)
adata = desc.train(adata, dims=[adata.shape[1], 32, 16], tol=0.005, n_neighbors=10,
                   batch_size=256, louvain_resolution=[0.8],
                   save_dir="result_pbmc3k", do_tsne=True, learning_rate=300,
                   do_umap=True, num_Cores_tsne=4,
                   save_encoder_weights=False)
t1 = time.time()
print("Total time running DAVAE 10w cells: %s seconds" % (str(t1-t0)))
time_list.append(t1-t0)

info = psutil.virtual_memory()
print('内存使用：', psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024, 'GB')
print('总内存：', info.total/1024/1024/1024, 'GB')
print('内存占比：', info.percent)
print('cpu个数：', psutil.cpu_count())
