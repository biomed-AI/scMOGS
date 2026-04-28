import os
import numpy as np
import pandas as pd
import anndata as ad
import argparse
import torch
import torch.cuda as cuda
from torch import nn
from torch.autograd import Variable
import torch.distributions as D
import torch.nn.functional as F
from scipy.io import mmread, mmwrite
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def compute_CGS(data_dir, result_dir, save_dir, species, homologs_path):
    indices_path = os.path.join(result_dir, 'indices.npy')
    indices = np.load(indices_path, allow_pickle=True)
    gene_index_list = []
    cell_index_list = []
    peak_index_list = []
    for a in range(len(indices)):
        gene_index_list = gene_index_list + list(indices[a]['gene_index'])
        cell_index_list = cell_index_list + list(indices[a]['cell_index'])
        peak_index_list = peak_index_list + list(indices[a]['peak_index'])

    gene_cell_embedding_path = os.path.join(result_dir, 'gene_cell_embedding.npy')
    peak_cell_embedding_path = os.path.join(result_dir, 'peak_cell_embedding.npy') 
    pred_label_path = os.path.join(result_dir, 'pred.npy') 
    
    gene_name_path = os.path.join(data_dir, 'Gene_names.tsv') 
    cell_name_path = os.path.join(data_dir, 'Cell_names.tsv') 
    peak_name_path = os.path.join(data_dir, 'Peak_names.tsv') 
    gene_peak_path = os.path.join(data_dir, 'Gene_Peak.mtx') 
    print("Reading file...")
    gene_names = pd.read_csv(gene_name_path, header=None)[0].tolist()
    cell_names = pd.read_csv(cell_name_path, header=None)[0].tolist()
    peak_names = pd.read_csv(peak_name_path, header=None)[0].tolist()
    gene_peak = mmread(gene_peak_path).toarray()

    pred_label = np.load(pred_label_path)

    gene_cell_embedding = np.load(gene_cell_embedding_path)
    peak_cell_embedding = np.load(peak_cell_embedding_path)

    gene_cell_embedding = F.log_softmax(torch.tensor(gene_cell_embedding), -1).numpy()
    peak_cell_embedding = F.log_softmax(torch.tensor(peak_cell_embedding), -1).numpy()
    print("Computing score...")
    gene_to_indices = defaultdict(list)
    for idx, gene in enumerate(gene_index_list):
        gene_to_indices[gene].append(idx)

    peak_to_indices = defaultdict(list)
    for idx, peak in enumerate(peak_index_list):
        peak_to_indices[peak].append(idx)
        
    unique_genes = list(gene_to_indices.keys())
    unique_peaks = list(peak_to_indices.keys())

    unique_gene_cell_embedding = np.zeros((len(unique_genes), gene_cell_embedding.shape[1]))
    unique_peak_cell_embedding = np.zeros((len(unique_peaks), peak_cell_embedding.shape[1]))
    all_peak_cell_embedding = np.zeros((len(peak_names), peak_cell_embedding.shape[1]))

    for i, gene in enumerate(unique_genes):
        rows = gene_cell_embedding[gene_to_indices[gene]]  
        unique_gene_cell_embedding[i] = rows.mean(axis=0)  

    for i, peak in enumerate(unique_peaks):
        rows = peak_cell_embedding[peak_to_indices[peak]]  
        unique_peak_cell_embedding[i] = rows.mean(axis=0)
        all_peak_cell_embedding[peak] = rows.mean(axis=0)
    
    GeneaggPeak_cell_embed = np.matmul(all_peak_cell_embedding.T, gene_peak.T).T
    unique_GeneaggPeak_cell_embedding = np.zeros((len(unique_genes), gene_cell_embedding.shape[1]))
    for i, gene in enumerate(unique_genes):
        unique_GeneaggPeak_cell_embedding[i] = GeneaggPeak_cell_embed[gene]

    scaler = MinMaxScaler(feature_range=(0, 1))  
    zscore_scaler = StandardScaler()

    unique_gene_cell_embedding = zscore_scaler.fit_transform(unique_gene_cell_embedding.T).T  
    unique_GeneaggPeak_cell_embedding = zscore_scaler.fit_transform(unique_GeneaggPeak_cell_embedding.T).T 

    sum_unique_gene_cell_embedding = 0.8*unique_gene_cell_embedding + 0.2*unique_GeneaggPeak_cell_embedding

    unique_labels = np.unique(pred_label)  
    num_classes = len(unique_labels)       
    num_genes = unique_gene_cell_embedding.shape[0]

    gene_names2 = [gene_names[index] for index in unique_genes]
    cell_names2 = [cell_names[index] for index in cell_index_list]

    save_dir1 = os.path.join(save_dir, os.path.basename(result_dir))
    mkdir(save_dir1)
    mkdir(os.path.join(save_dir1, 'latent_to_gene'))
    sum_unique_gene_cell_embedding_save_path = os.path.join(save_dir1, 'latent_to_gene', os.path.basename(save_dir1)+'_gene_marker_score.feather')

    if species=='human':
        mk_score_df = pd.DataFrame(sum_unique_gene_cell_embedding, index=gene_names2, columns=cell_names2)
        mk_score_df.reset_index(inplace=True)
        mk_score_df.rename(columns={"index": "HUMAN_GENE_SYM"}, inplace=True)
        mk_score_df.to_feather(sum_unique_gene_cell_embedding_save_path)
    elif species=='mouse':
        homologs_df = pd.read_csv(homologs_path, sep='\t')
        homologs_dict = dict(zip(homologs_df['MOUSE_GENE_SYM'], homologs_df['HUMAN_GENE_SYM']))
        gene_names3 = []
        abandon_gene_name_index = []
        use_gene_name_index = []
        for gene in gene_names2:
            if gene in list(homologs_dict.keys()):
                human_gene = homologs_dict[gene]
                gene_names3.append(human_gene)
                use_gene_name_index.append(gene_names2.index(gene))
            else:
                abandon_gene_name_index.append(gene_names2.index(gene))
        
        mk_score_df = pd.DataFrame(sum_unique_gene_cell_embedding[use_gene_name_index,:], index=gene_names3, columns=cell_names2)
        mk_score_df.reset_index(inplace=True)
        mk_score_df.rename(columns={"index": "HUMAN_GENE_SYM"}, inplace=True)
        
    print("Finish")

def main():
    parser = argparse.ArgumentParser(description='Computing the cell-gene interaction score')
    parser.add_argument('--input_file', type=str) #Path to the input dataset directory
    parser.add_argument('--embedding_file', type=str) #Path to the directory where the embeddings is saved
    parser.add_argument('--output_file', type=str) #Path to the directory where the score matrix will be saved
    parser.add_argument('--species', type=str, default='human') #human or mouse
    parser.add_argument('--homologs_path', type=str, default='./scMOGS/mouse_human_homologs.txt') #Path to the a homologous transformation file for converting the gene names
    args = parser.parse_args()

    input_file = args.input_file
    embedding_file = args.embedding_file
    output_file = args.output_file
    species = args.species
    homologs_path = args.homologs_path
    compute_CGS(input_file, embedding_file, output_file, species, homologs_path)


if __name__ == "__main__":
    main()