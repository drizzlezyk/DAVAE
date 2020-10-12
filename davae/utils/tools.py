import numpy as np
import csv
import scanpy as sc
from scipy.sparse import csr_matrix
from sklearn import metrics
import anndata
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def read_sc_data(input_file, fmt='h5ad', backed=None, transpose=False, sparse=False, delimiter=" "):
    if fmt == '10x_h5':
        adata = sc.read_10x_h5(input_file)
    elif fmt == '10x_mtx':
        adata = sc.read_10x_mtx(input_file)
    elif fmt == "mtx":
        adata = sc.read_mtx(input_file)
    elif fmt == 'h5ad':
        adata = sc.read_h5ad(input_file, backed=backed)
    elif fmt == "csv":
        adata = sc.read_csv(input_file)
    elif fmt == "txt":
        adata = sc.read_text(input_file, delimiter=delimiter)
    elif fmt == "tsv":
        adata = sc.read_text(input_file, delimiter="\t")
    else:
        raise ValueError('`format` needs to be \'10x_h5\' or \'10x_mtx\'')
    if transpose:
        adata = adata.transpose()
    if sparse:
        adata.X = csr_matrix(adata.X, dtype='float32')
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    return adata


def cacu_size_factor(adata):
    return adata.X.sum(axis=1)


def get_label_by_count(label_path):
    label = []
    with open(label_path)as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            count = row
    for i in range(len(count)):
        for j in range(int(count[i])):
            label.append(i)
    return label


def merge(files, type='h5ad'):
    adata_all = read_sc_data(files[0], type)
    sample_count = []
    sample_count.append(adata_all.shape[0])
    for i in range(len(files)-1):
        adata = read_sc_data(files[i+1], type)
        sample_count.append(adata.shape[0])
        adata_all = adata_all.concatenate(adata)
    return adata_all, sample_count


def write_count(count, out_path):
    with open(out_path, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(count)


def filt(adata, min_c, min_g):
    sc.pp.filter_genes(adata, min_cells=min_c)
    sc.pp.filter_cells(adata, min_genes=min_g)
    return adata


def write_merge_label(count, labels, path):
    res=[]
    for i in range(len(count)):
        for j in range(count[i]):
          res.append(labels[i])
    print(res[1])
    with open(path, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(res)
    f.close()


def find_gene_pos(genes, gene):
    genes['i'] = range(genes.shape[0])
    return genes.loc[gene]['i']


def cacu_color(X, i):
    return X[:, i]


def get_label_by_txt(txtpath):
    label = []
    with open(txtpath) as label_f:
        f11 = label_f.read().splitlines()
    for x in f11:
        label.append(x)
    return label


def cacu_clustering_metrics(x_list, label, metrics_list):
    result = []
    for m in metrics_list:
        m_list = []
        if m == 'sh':
            for x in x_list:
                m_list.append(metrics.silhouette_score(x, label))
        elif m == 'ch':
            for x in x_list:
                m_list.append(metrics.calinski_harabasz_score(x, label))
        result.append(m_list)
    return result


def txt_to_adata(txt_path ):
    f = open(txt_path)
    lines = f.readlines()
    gene_count = len(lines)-1
    first_line = lines[0].split()
    cell_count = len(first_line)-1

    data = []
    for i in range(1, gene_count+1):

        para_line = lines[i].strip().split(' ')
        del para_line[0]
        data.append(para_line)
    f.close()
    data = np.array(data)
    adata = anndata.AnnData(X=data)
    return adata


def seurat_preprocession(adata, min_cells, min_genes):
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def joint_cluster_feature(dataset, k):
    x_nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    x_nbrs.fit(dataset)
    print(x_nbrs)
    dist, ind = x_nbrs.kneighbors()
    feature = np.zeros([dataset.shape[0], k])
    f = np.zeros([dataset.shape[0], 1])
    for i in range(0, dataset.shape[0]):
        f[i] = max(dataset[i])
    for i in range(0, dataset.shape[0]):
        for j in range(0, k):
            feature[i, j] = f[ind[i, j]]
    return feature


def cacu_Manhattan_dist(dataset):
    d2 = pdist(dataset, 'cityblock')
    y = squareform(d2)
    print(y.shape)
    return y


def get_panc8(tech, type=''):
    # panc8_path = '../data/panc_8/'
    panc8_path = '/Users/zhongyuanke/data/seurat_data/panc_8/'
    data_path = panc8_path + tech + '_data'+type+'.csv'
    celltype_path = panc8_path+tech+'_cell_type.csv'
    cell_type = pd.read_csv(celltype_path, index_col=0)
    cell_type = cell_type.values
    adata = sc.read_csv(data_path)
    print(adata.shape)
    return adata, cell_type


def get_intersection_panc8(tech, norm=True):
    panc8_path = '/Users/zhongyuanke/data/dann_vae/panc8/'
    data_path = panc8_path+'inter_'+tech+'.h5ad'
    adata = sc.read_h5ad(data_path)
    return adata


def get_ifnb(tech='stim',type=''):
    ifnb_path = '/Users/zhongyuanke/data/seurat_data/ifnb/'
    data_path = ifnb_path + tech + type +'.csv'
    celltype_path = ifnb_path + tech + '_cell_type.csv'
    cell_type = pd.read_csv(celltype_path, index_col=0)
    cell_type = cell_type.values
    adata = sc.read_csv(data_path)
    return adata, cell_type


def text_label_to_number(txt_label):
    encoder = LabelEncoder()
    label = encoder.fit_transform(txt_label)
    return label


def filt_genes(x, min_cells=0):
    result = []
    x = x.transpose()
    for row in x:
        count = 0
        for i in range(x.shape[1]):
            if row[i] > 0:
                count += 1
        if count >= min_cells:
            result.append(row)
    result = np.array(result)
    result = result.transpose()
    return result


def up_sampling(x, y, batch_size):
    orig_len_x = x.shape[0]
    orig_len_y = y.shape[0]
    # sc.pp.filter_genes(data1, min_cells=3)
    # sc.pp.filter_genes(data2, min_cells=3)
    len_x = orig_len_x
    len_y = orig_len_y

    n = orig_len_x // len_y if orig_len_x > len_y else len_y // orig_len_x

    if orig_len_x > len_y:
        if orig_len_x % batch_size != 0:
            add_len_x = batch_size - orig_len_x % batch_size
        else:
            add_len_x = 0
        len_x = orig_len_x + add_len_x
        add_x = x[:add_len_x, ]
        x = np.vstack((x, add_x))
        up_y = np.zeros([len_x, y.shape[1]])
        for i in range(n):
            up_y[i * len_y:(i + 1) * len_y, ] = y
        up_y[n * len_y:len_x, ] = y[0:len_x - n * len_y, ]
        y = up_y
    else:
        if orig_len_y % batch_size != 0:
            add_len_y = batch_size - orig_len_y % batch_size
        else:
            add_len_y = 0
        len_y = orig_len_y + add_len_y
        add_y = y[:add_len_y, ]
        y = np.vstack((y, add_y))

        up_x = np.zeros([len_y, x.shape[1]])
        for i in range(n):
            up_x[i * len_x:(i + 1) * len_x, ] = x
        up_x[n * len_x:len_y, ] = x[0:len_y - n * len_x, ]
        x = up_x
    return x, y


def multi_up_sampling(x, batch_size):
    orig_lens = []
    for i in range(len(x)):
        orig_lens.append(x[i].shape[0])
    lens = orig_lens
    max_len = max(orig_lens)
    max_index = orig_lens.index(max(orig_lens))
    print(max_len)
    if max_len % batch_size != 0:
        add_len = batch_size - max_len % batch_size
        max_len = max_len + add_len
    up_datasets = []
    for i in range(len(x)):
        n = max_len//x[i].shape[0]
        up_x = np.zeros([max_len, x[0].shape[1]])
        for j in range(n):
            up_x[j * x[i].shape[0]:(j + 1) * x[i].shape[0], ] = x[i]
        up_x[n * x[i].shape[0]:max_len, ] = x[i][0:max_len - n * x[i].shape[0], ]
        up_datasets.append(up_x)
    return up_datasets, max_index


def locate_sample(barcodes, barcode):
    for i in range(len(barcodes)):
        if barcode == barcodes[i]:
            return i
    return -1


def intersection_genes(data1, data2, genes1, genes2):
    data1 = data1.transpose()
    data2 = data2.transpose()
    inter_genes = []
    filt_data1 = []
    filt_data2 = []
    for i in range(len(genes1)):
        if genes1[i] in genes2:
            index2 = genes2.index(genes1[i])
            inter_genes.append(genes1[i])
            filt_data1.append(data1[i, ])
            filt_data2.append(data2[index2, ])
    filt_data1 = np.array(filt_data1).transpose()
    filt_data2 = np.array(filt_data2).transpose()
    return filt_data1, filt_data2, inter_genes


def multi_intersection_genes(data_list, gene_list):
    filt_data_list = []

    for i in range(len(data_list)):
        data_list[i] = data_list[i].transpose()

    ref_genes = list(set(gene_list[0]).intersection(gene_list[1]))
    for i in range(2, len(gene_list)):
        ref_genes = list(set(ref_genes).intersection(gene_list[i]))

    for i in range(len(data_list)):
        para_genes = gene_list[i]
        para_data = data_list[i]
        filt_data = []
        for j in range(len(para_genes)):
            if para_genes[j] in ref_genes:
                filt_data.append(para_data[j])

        filt_data = np.array(filt_data)
        filt_data = filt_data.transpose()
        print(filt_data.shape)
        filt_data_list.append(filt_data)

    return filt_data_list, ref_genes


def sample_by_reference(ref_barcodes, barcodes, data):
    sample_data = []
    for i in range(len(ref_barcodes)):
        if ref_barcodes[i] in barcodes:
            index = barcodes.index(ref_barcodes[i])
            sample_data.append(data[index, ])
    sample_data = np.array(sample_data)
    print(sample_data.shape)
    return 0

