import anndata as ad
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
from tqdm import tqdm
import math
import scanpy as sc
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import sys

def subgraph(graph, seed, n_neighbors, node_sele_prob):
    total_matrix_size = 1 + np.cumprod(n_neighbors).sum()  
    picked_nodes = {seed}  
    last_layer_nodes = {seed}
    
    to_pick = 1
    for n_neighbors_current in n_neighbors:  
        to_pick = to_pick * n_neighbors_current
        neighbors = graph[list(last_layer_nodes), :].nonzero()[1]  
        
        neighbors_prob = node_sele_prob[list(neighbors)]
        neighbors = list(set(neighbors))  
        n_neigbors_real = min(
            to_pick,
            len(neighbors))  
        if len(neighbors_prob) == 0:
            continue
        last_layer_nodes = set(
            np.random.choice(neighbors, n_neigbors_real, replace=False,
                             p=softmax(neighbors_prob)))  # Select non-repeated nodes from neighbors
        
        picked_nodes |= last_layer_nodes  # Update picked_nodes as last_layer_nodes ∪ picked_nodes
    indices = list(sorted(picked_nodes - {seed}))
    return indices



def softmax(x):

    x = np.array(x, dtype=float)
    if x.size == 0:
        return x
    e = np.exp(x - np.max(x))
    return e / e.sum()

def random_walk_subgraph(graph, seed,rna=True,i=0,node_list=None,walk_length=10,cell_size=25):
    count = 0
    max_attempts = 500
    attempts = 0
    if node_list is not None:
        graph[node_list,:] = 0
    if not rna:
        cell_nodes = {seed}
        count += 1
    else:
        cell_nodes = set()


    while count < cell_size and attempts < max_attempts:
        attempts += 1
        current = seed
        neighbors = graph[current, :].nonzero()[1]
        for i in range(walk_length):
            neighbors = list(set(neighbors))
            if i % 2 == 0: 
                graph_matrix = graph.transpose()[:,current].todense()
            else:
                graph_matrix = graph.transpose()[current,:].todense()
            rna_ = graph_matrix
            if rna:
                rna_[rna_ < 5] = 0

            neighbors_prob = np.squeeze(np.array(np.log(rna_ + 1)))[list(neighbors)]
            
            if len(neighbors_prob) == 0:
                break  
            probs = softmax(neighbors_prob)
            next_node = np.random.choice(neighbors, p=probs)
            current = next_node  
            if i % 2 == 0: 
                neighbors = graph[:,current].nonzero()[0]
            else:
                neighbors = graph[current, :].nonzero()[1]
                cell_nodes.add(next_node)
                if len(cell_nodes) > count :
                    count += 1
            if count >= cell_size:
                break
    return sorted(cell_nodes)

def subgraph_extract(RNA_matrix, ATAC_matrix, neighbor=[20], cell_size=30):
    print('Using random walk to partition the data into batches...')
    node_ids = np.random.choice(RNA_matrix.shape[1], size=RNA_matrix.shape[1], replace=False) #500
    n_batch = math.ceil(node_ids.shape[0] / cell_size)
    indices_ss = []
    RNA_matrix1 = RNA_matrix.copy()
    dic = {}
    selected_nodes = set()
    seed_list = set([node_ids[i] for i in range(n_batch)])
    random_seeds = set()

    for i in tqdm(range(n_batch)):
        gene_indices_all = []
        peak_indices_all = []
        if i < n_batch - 1:
            seed = node_ids[i]
            RNA_matrix2 = RNA_matrix.copy()
            RNA_matrix2[:,np.array(list(selected_nodes))] = 0
            RNA_matrix2[:,np.array(list(seed_list - {seed}))] = 0
            node_list1 =  random_walk_subgraph(RNA_matrix2.transpose(), seed,rna=True)
            node_list1 = list(set(node_list1)- {seed})

            if len(node_list1) < 24:
                random_seeds.add(seed)
                continue

            selected_nodes = selected_nodes.union(node_list1)
            ATAC_matrix2 = ATAC_matrix.copy()
            ATAC_matrix2[:,np.array(list(selected_nodes))] = 0
            ATAC_matrix2[:,np.array(list(seed_list - {seed}))] = 0
            
            node_list2 = random_walk_subgraph(ATAC_matrix2.transpose(), seed,rna=False,node_list=node_list1,walk_length=10,cell_size=50-len(node_list1))

            if len(node_list2) < 50-len(node_list1):
                random_seeds.add(seed)
                selected_nodes = selected_nodes - set(node_list1)
                continue
            selected_nodes = selected_nodes.union(node_list2)

            node_list = np.concatenate([node_list1 , node_list2])
        

            for index, node in enumerate(node_list):
                rna_ = RNA_matrix1[:, node].todense()
                rna_[rna_ < 5] = 0
                gene_indices = subgraph(RNA_matrix.transpose(), node, neighbor, np.squeeze(np.array(np.log(rna_ + 1))))
                peak_indices = subgraph(ATAC_matrix.transpose(), node, neighbor,
                                        np.squeeze(np.array(np.log(ATAC_matrix[:, node].todense() + 1))))
                
                dic[node] = {'g': gene_indices, 'p': peak_indices}
                gene_indices_all = gene_indices_all + gene_indices
                peak_indices_all = peak_indices_all + peak_indices
                
            gene_indices_all = list(set(gene_indices_all))
            peak_indices_all = list(set(peak_indices_all))
            node_list = np.array(list(node_list),dtype=int)

            h = dict()
            h['gene_index'] = gene_indices_all
            h['peak_index'] = peak_indices_all
            h["cell_index"] = node_list
            indices_ss.append(h)
            
        else:
            seed = node_ids[i]
            random_seeds.add(seed)
            random_seeds = list(random_seeds)
            RNA_matrix2 = RNA_matrix.copy()
            RNA_matrix2[:,np.array(list(selected_nodes))] = 0
            RNA_matrix2[:,np.array(list(seed_list))] = 0
            nodes = list(set(RNA_matrix2.nonzero()[1]))
            for i in range(len(random_seeds)):
                seed = random_seeds[i]

                node_list = set(nodes[i * 49:(i+1) * 49])
                node_list.add(seed)
                
                selected_nodes = selected_nodes.union(node_list)
                node_list = list(node_list)
            
                for index, node in enumerate(node_list):
                    rna_ = RNA_matrix1[:, node].todense()
                    rna_[rna_ < 5] = 0
                    gene_indices = subgraph(RNA_matrix.transpose(), node, neighbor, np.squeeze(np.array(np.log(rna_ + 1))))
                    peak_indices = subgraph(ATAC_matrix.transpose(), node, neighbor,
                                            np.squeeze(np.array(np.log(ATAC_matrix[:, node].todense() + 1))))
                    
                    dic[node] = {'g': gene_indices, 'p': peak_indices}
                    gene_indices_all = gene_indices_all + gene_indices
                    peak_indices_all = peak_indices_all + peak_indices

        
                gene_indices_all = list(set(gene_indices_all))
                peak_indices_all = list(set(peak_indices_all))
                node_list = np.array(list(node_list),dtype=int)

                h = dict()
                h['gene_index'] = gene_indices_all
                h['peak_index'] = peak_indices_all
                h["cell_index"] = node_list

                indices_ss.append(h)

    return indices_ss, node_ids, dic


def softmax(x):
    return (np.exp(x) / np.exp(x).sum())

def segment_function(x):
    if x <= 500:
        return 0.2, 5
    elif x <= 5000:
        return 0.5, 10
    else:
        return 0.8, 15
        

class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """Constructor for LabelSmoothing module.
        :param smoothing: Label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def init_cluster(RNA_matrix, custom_n_neighbors=None, n_pcs=40, custom_resolution=None, use_rep=None):
    # print(
    #     '\tWhen the number of cells is less than or equal to 500, it is recommended to set the resolution value to 0.2.')
    # print('\tWhen the number of cells is within the range of 500 to 5000, the resolution value should be set to 0.5.')
    # print('\tWhen the number of cells is greater than 5000, the resolution value should be set to 0.8.')

    adata = ad.AnnData(RNA_matrix.transpose(), dtype='int32')
    if custom_resolution is None or custom_n_neighbors is None:
        resolution, n_neighbors = segment_function(adata.shape[0])
    else:
        resolution = custom_resolution
        n_neighbors = custom_n_neighbors

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if use_rep is not None:
        adata.obsm['use_rep']=use_rep
        sc.pp.neighbors(adata, use_rep='use_rep', n_neighbors=n_neighbors)
    else:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    sc.tl.leiden(adata, resolution)
    return adata.obs['leiden']


def purity_score(y_true, y_pred):
    """Purity score

    Args:
        y_true (np.ndarray): n*1 matrix, true labels
        y_pred (np.ndarray): n*1 matrix, predicted clusters

    Returns:
        float: Purity score
    """
    # Create a matrix to store the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)

    # Sort the labels
    # Some labels might be missing, e.g., a set {0,2} where 1 is missing
    # First, find the unique labels and then map them to an ordered set
    # E.g., {0,2} should be mapped to {0,1}
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    y_true = np.array(y_true, dtype='int64')

    # Update the unique labels
    labels = np.unique(y_true)

    # Set the number of bins to n_classes + 2 so that we can compute the actual
    # class occurrences between two consecutive bins
    # The larger bin is excluded: [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        # Find the most frequent label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    y_true = np.array(y_true, dtype='int8')
    y_voted_labels = np.array(y_voted_labels, dtype='int8')
    return accuracy_score(y_true, y_voted_labels), y_true


def Entropy(pred_label, true_label):
    e = 0
    for k in set(pred_label):
        en = 0
        pred_k = Counter(pred_label)[k]
        index_pred_k = pred_label == k
        for j in set(true_label):
            true_j = Counter(true_label)[j]
            intersection_kj = (true_label[index_pred_k] == j).sum()
            p = np.array(intersection_kj) / np.array(pred_k)
            if p != 0:
                en += np.log(p) * p
        e = e + en * pred_k / true_label.shape[0]
    return abs(e)
