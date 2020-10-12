import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import tools
from sklearn import preprocessing
import scanpy as sc
import umap


sns.set(style='white', rc={'figure.figsize': (8, 6), 'figure.dpi': 150})
matplotlib.use('TkAgg')
bar = ['c', 'violet', 'darkorange', 'darkseagreen', 'r', 'silver']


def plot_gradient_marker_gene(x, adata, gene_names, max_list, xy_label, figsize, sub_range, save_path, font_size=14, cmap='Blues'):
    all_gene = adata.var

    index_list, gene_value, color_list, x_list, xy_label_list = [], [], [], [], []
    for name in gene_names:
        index_list.append(tools.find_gene_pos(all_gene, name))
        x_list.append(x)
        xy_label_list.append(xy_label)
    for i in index_list:
        gene_value.append(tools.cacu_color(adata.X.A, i))

    fig = plt.figure(figsize=figsize)
    for i in range(len(x_list)):
        para_sub = sub_range * 10 + i + 1
        ax = fig.add_subplot(para_sub)
        sc = ax.scatter(x_list[i][:, 0], x_list[i][:, 1], c=gene_value[i], vmin=0, vmax=max_list[i], cmap=cmap, s=1, linewidth=0)
        plt.xlabel(xy_label[i] + '1')
        plt.ylabel(xy_label[i] + '2')
        plt.title(gene_names[i], fontsize=font_size)
        cbar = fig.colorbar(sc)
    plt.savefig(save_path)


def plot_mark_gene(x, adata, gene_names, thresholds, xy_label, figsize, sub_range, save_path, font_size=14):
    all_gene = adata.var
    index_list, gene_value, color_list, x_list, xy_label_list = [], [], [], [], []
    for name in gene_names:
        index_list.append(tools.find_gene_pos(all_gene, name))
        x_list.append(x)
        xy_label_list.append(xy_label)
    for i in index_list:
        gene_value.append(preprocessing.minmax_scale(tools.cacu_color(adata.X.A, i)))
    for k in range(len(gene_value)):
        para_color = []
        for i in range(len(gene_value[k])):
            if gene_value[k][i] < thresholds[k]:
                para_color.append(bar[5])
            else:
                para_color.append(bar[4])
        color_list.append(para_color)
    plot_multi(x_list, color_list, xy_label_list, gene_names, figsize, sub_range, 1, save_path, font_size=font_size)


def plot_multi(x, color, xy_label, title, fig_size, sub_range, size, sava_path, c_map='Set2', font_size=14):
    fig = plt.figure(figsize=fig_size)

    for i in range(len(x)):
        para_sub = sub_range*10+i+1
        ax = fig.add_subplot(para_sub)
        ax.scatter(x[i][:, 0], x[i][:, 1], c=color[i], cmap=c_map, s=size, linewidth=0, alpha=0.7)
        plt.xlabel(xy_label[i] + '1')
        plt.ylabel(xy_label[i] + '2')
        plt.title(title[i], fontsize=font_size)
    plt.savefig(sava_path)


# given expression matrix, gene_list(marker_gene) , plot a heat map
def plot_expression_matrix(x, gene_index_list, title, save_path, normalize='minmax', cmap = 'Set2'):
    expression = []
    for i in range(len(gene_index_list)):
        expression.append(x[:, gene_index_list[i]])
    expression = np.array(expression)
    if normalize == 'minmax':
        expression = preprocessing.scale(expression)
    elif normalize == 'zscore':
        expression = preprocessing.minmax_scale(expression)
    plt.figure()
    plt.title(title)
    plt.imshow(expression, interpolation='nearest', cmap=cmap, origin='lower')
    plt.savefig(save_path)


def plot_integration(datasets, title, out_path, xy_label='umap'):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    plt.title(title)
    for i in range(len(datasets)):
        ax.scatter(datasets[i][:, 0], datasets[i][:, 1], c=bar[i], s=2.5, linewidth=0)
    plt.xlabel(xy_label+'1')
    plt.ylabel(xy_label+'2')
    plt.savefig(out_path)


def plot_cluster(x, label, save_path, cmap='Set1', xy_label='tsne'):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.scatter(x[:, 0], x[:, 1], c=label, cmap=cmap, s=2, linewidth=0)
    plt.xlabel(xy_label + '1')
    plt.ylabel(xy_label + '2')
    plt.savefig(save_path)


def plot_integration_with_cluster(datasets, label, out_path, c_map='Set1', xy_label='tsne', title='Original'):
    fig = plt.figure(figsize=(18, 8))
    data = np.concatenate(datasets)
    ax = fig.add_subplot(121)
    plt.title(title)

    scatter = ax.scatter(datasets[0][:, 0], datasets[0][:, 1], c=bar[0], s=2.5, linewidth=0, label='DNA')
    scatter = ax.scatter(datasets[1][:, 0], datasets[1][:, 1], c=bar[1], s=2.5, linewidth=0, label='Chromatin')
    plt.xlabel(xy_label + '1')
    plt.ylabel(xy_label + '2')
    plt.legend(loc="upper right", title="Batch")

    ax = fig.add_subplot(122)
    plt.title(title)
    plt.xlabel(xy_label + '1')
    plt.ylabel(xy_label + '2')
    scatter = ax.scatter(data[:, 0], data[:, 1], c=label, cmap=c_map, s=3, linewidth=0)
    handles, labels = scatter.legend_elements(num=19, alpha=0.6)
    celltype = ['Ex23_Cux1', 'Ex6_Tle4', 'Ex345_Foxp1', 'IP_Eomes','RG','Ex4_Tenm3','Ex5_Crmp1','Ex45_Galntl6','In_1','In_2',
                'IP_Hmgn2','Ex56_Epha6','Ex23_Cntn2','IP_Gadd45g','CR','Endo','Peri','OPC','Mic']

    legend2 = plt.legend(handles, celltype, loc="upper center", title="Cell Type", ncol=4)
    plt.savefig(out_path)


def plot_davae_result_labeled(adata, title='DAVAE', cmap_batch='Set2', cmap_label='Set3', save=None):
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(121)
    plt.title(title)
    batch = adata.obs['batch'].values
    batch = np.array(batch, dtype=int)
    label = adata.obs['label']
    data_emb = adata.obsm['umap']
    xy_label = 'umap'
    ax.scatter(data_emb[:, 0], data_emb[:, 1], cmap=cmap_batch, c=batch, s=1, linewidth=0)
    plt.xlabel(xy_label + '1')
    plt.ylabel(xy_label + '2')
    ax = fig.add_subplot(122)
    plt.title(title)
    plt.xlabel(xy_label + '1')
    plt.ylabel(xy_label + '2')
    ax.scatter(data_emb[:, 0], data_emb[:, 1], c=label, cmap=cmap_label, s=1, linewidth=0)
    plt.show()
    if save != None:
        plt.savefig(save)


def plot_davae_result(adata, cmap='Set2'):
    title = 'DAVAE'
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    plt.title(title)
    batch = adata.obs['batch']
    data_emb = adata.obsm['umap']
    xy_label = 'umap'
    ax.scatter(data_emb[:, 0], data_emb[:, 1], cmap=cmap, c=batch, s=1, linewidth=0)
    plt.xlabel(xy_label + '1')
    plt.ylabel(xy_label + '2')
    plt.show()


