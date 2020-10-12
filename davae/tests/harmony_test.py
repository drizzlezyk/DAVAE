import harmonypy as hm
import numpy as np
import umap
import time
import scanpy as sc
import os
import pandas as pd
import psutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_path", type=str, default='/Users/zhongyuanke/data/dann_vae/scalability/', help="base path")

opt = parser.parse_args()
vars_use = ['batch']
base_path = opt.base_path
file1 = base_path+'blood_5w.h5ad'
file2 = base_path+'bone_5w.h5ad'
adata1 = sc.read_h5ad(base_path+'blood_5w.h5ad')
adata2 = sc.read_h5ad(base_path+'bone_5w.h5ad')
adata = adata1.concatenate(adata2)
data = adata.X.A
sc.pp.log1p(adata)
print(adata.obs)
meta_data = adata.obs
ho = hm.run_harmony(data, meta_data, vars_use)