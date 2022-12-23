

# A versatile and scalable single-cell data integration algorithm based on domain-adversarial and variational approximation

[![Build Status](https://travis-ci.com/drizzlezyk/DAVAE.svg?token=xjQszw5hMG8ZGQ7oscBw&branch=master)](https://travis-ci.com/drizzlezyk/DAVAE)
[![Documentation Status](https://readthedocs.org/projects/davae/badge/?version=latest)](https://davae.readthedocs.io/en/latest/?badge=latest)

Single-cell transcriptomics allows us to observe cell status and identify cell types at the level of individual cells. With the development of single cell multi omics, a key analytical challenge is to integrate these data sets from different technologies and experiments to uncover biological insights. We present DAVAE, an unsupervised deep embedding algorithm that incorporate domain adaptation into the variational autoencoder to learn batch-invariant features, enabling us to integrate single-cell measurements across technologies and different modalities.

***



## Installation 

```python
$ pip install davae
```

**Note**: you need to make sure that the `pip` is for python3. Our package is suitable for both tensorflow1 and tensorflow2.



## Tutorial 1:  Human Cell Atlas(HCA)

#### Import module

```python
import davae
import scanpy as sc
```

#### Load data

we use two HCA darasets, cord blood can be download: https://s3.amazonaws.com/preview-ica-expression-data/ica_cord_blood_h5.h5; bone marrow: https://s3.amazonaws.com/preview-ica-expression-data/ica_bone_marrow_h5.h5. We will analyze ~780000 cells. 

```python
base_path = '/Users/zyk/data/'
file1 = base_path+'HCA/ica_cord_blood_h5.h5'
file2 = base_path+'HCA/ica_bone_marrow_h5.h5'

adata1 = sc.read_10x_h5(file1)
adata2 = sc.read_10x_h5(file2)

adata1.var_names_make_unique()
adata2.var_names_make_unique()
```

#### Preprocessing

```python
# filer genes that only less than 1% cell expressed
sc.pp.filter_genes(adata1, min_cells=3000)
sc.pp.filter_genes(adata2, min_cells=3000)
adata_list = [adata1, adata2]
```

#### Integration with DAVAE

```python
adata_davae = DAVAE_Integration(adata_list)
```

#### Result visualization 

```python
# using umap to realize dimensionality reduction
sc.pp.neighbors(adata_davae, use_rep='davae')
sc.tl.umap(adata_davae)
sc.tl.louvain(adata_davae)
# visualization
sc.pl.umap(adata_davae, color=['batch', 'louvain'])

```



<img src="https://github.com/drizzlezyk/DAVAE/blob/master/result/hca/batch.png" width="50%">

<img src="https://github.com/drizzlezyk/DAVAE/blob/master/result/hca/cluster33.png" width="50%">

```python 
# marker visualization
sc.pl.umap(adata_davae, color=['CD79A', 'S100A8', 'HPRT1', 'GNLY', 'CST3', 'CD3D'],
           s=1, frameon=False, ncols=3, vmax='p99')

```

<img src="https://github.com/drizzlezyk/DAVAE/blob/master/result/hca/marker6.png" width="80%">



# Tutorial 2  Control & stimulation data from PBMC



### Packages need to import

```
from davae.network import dann_cvae
from davae.utils import tools, plot
import numpy as np
import umap
import davae
import scanpy as sc
```

### Load datasets

```python
adata1 = sc.read_h5ad(base_path + 'ifnb/ctrl.h5ad')
adata1 = sc.read_h5ad(base_path + 'ifnb/stim.h5ad')
```

### Use DAVAE for integration

```python
data_list = [adata1, adata2]
adata_davae = dann_cvae.integration(data_list, epoch=30)
```

### Dimension reduction and visualization

```python
sc.pp.neighbors(adata_davae, use_rep='davae')
sc.tl.umap(adata_davae)

# visualization
adata.obs['cell type'] = adata.obs['label'].map(cluster2annotation_ifnb).astype('category')
cluster2annotation_ifnb = {
     0: 'CD14 Mono',
     1: 'CD4 Naive T',
     2: 'CD4 Memory T',
     3: 'CD16 Mono',
     4: 'B',
     5: 'CD8 T',
     6: 'NK',
     7: 'T activated',
     8: 'DC',
     9: 'B activated',
     10: 'MK',
     11: 'pDC',
     12: 'Eryth',
}
sc.pl.umap(adata_davae, color=['batch', 'cell type'])
```
<img src='https://github.com/drizzlezyk/DAVAE/blob/master/result/ifnb/seurat_label.png' width="80%">



### Observe gene expression changes after stimulation

```
sc.pl.umap(adata2, color=['TNFSF10', 'APOBEC3A', "CCL8", 'CXCL10',  'ISG15', 'LAG3'],
           s=3, frameon=False, ncols=6, vmax='p99')
sc.pl.umap(adata1, color=['TNFSF10', 'APOBEC3A', "CCL8", 'CXCL10', 'ISG15', 'LAG3'],
           s=3, frameon=False, ncols=6, vmax='p99')
```

<img src='https://github.com/drizzlezyk/DAVAE/blob/master/result/ifnb/control_marker.png'>

<img src='https://github.com/drizzlezyk/DAVAE/blob/master/result/ifnb/stim_markers.png'>







