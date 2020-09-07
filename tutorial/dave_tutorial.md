# Integration of scRNA-seq data using Domain adaption deep variational autoencoder(DAVAE)



Abstract

***

## Installation 

```python
$ pip install davae
```

**Note**: you need to make sure that the `pip` is for python3



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
import umap
import davae.utils.plot as plot

# using umap to realize dimensionality reduction
data = adata_davae.obsm['davae_result']
data_emb = umap.UMAP().fit_transform(data)
adata_davae.obsm['umap'] = data_emb

# visualization
plot.plot_davae_result(adata_davae)
```

![avatar](/Users/zhongyuanke/Documents/graduated/integration/tutorial/label_01.png)

```python 
# plot marker genes
gene_names = ['LYZ', 'CD14', 'CD79A', 'MS4A1']
thresholds = [7, 4, 7, 4]
plot.plot_gradient_marker_gene(data_emb, adata, gene_names, thresholds, 'UMAP', fig_size, 14, fig_path)
```

<img src="/Users/zhongyuanke/data/dann_vae/hca/marker_save01.png" alt="avatar" style="zoom:68%;" />



`











