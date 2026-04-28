import os
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import multiprocessing as mp
from collections import Counter
from operator import itemgetter
import random
from functools import partial
import scipy.sparse as sp
from scipy.io import mmread
from scipy.sparse import hstack, vstack, coo_matrix
import seaborn as sb
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import SparsePCA
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.cuda as cuda
from torch import nn
from torch.autograd import Variable
import torch.distributions as D
import torch.nn.functional as F
import torch_geometric.data as Data
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax as Softmax
from torchmetrics.functional import pairwise_cosine_similarity
import warnings
from warnings import filterwarnings
import argparse
from tqdm import tqdm
import scanpy as sc
from scipy import sparse

from scMOGS.conv import *
#from scMOGS.egrn import *
from scMOGS.model import *
from scMOGS.tools import *

def main():
    filterwarnings("ignore")
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description='Training HGNN on the heterogeneous graph')
    parser.add_argument('--lr', type=float, default=0.0005) # Learning rate
    parser.add_argument('--labsm', type=float, default=0.1) # The rate of LabelSmoothing
    parser.add_argument('--wd', type=float, default=0.1) # Weight decay

    parser.add_argument('--nlayers', type=int, default=3) # The number of graph convolution layers
    parser.add_argument('--n_hid', type=int, default=104) # The dimensionality of embeddings
    parser.add_argument('--nheads', type=int, default=8) # The number of parallel attention heads within the multi-head attention module

    parser.add_argument('--neighbor', type=int, default=20) # The number of neighboring nodes to be selected for each cell in the subgraph
    parser.add_argument('--cell_size', type=int, default=50) # The number of cells per subgraph (batch)
    parser.add_argument('--epochs_p1', type=int, default=10) # The epoch number of MultimodalFeatureEncoder
    parser.add_argument('--epochs_p2', type=int, default=5) # The epoch number of IntegratedOmicTrainer
    parser.add_argument('--device', type=int, default=0) # GPU device
    parser.add_argument('--input_file', type=str) #Path to the input dataset directory
    parser.add_argument('--output_file', type=str) #Path to the directory where the embeddings will be saved
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    labsm = args.labsm
    lr = args.lr
    wd = args.wd
    n_hid = args.n_hid
    nheads = args.nheads
    nlayers = args.nlayers
    neighbor = args.neighbor
    cell_size = args.cell_size
    epochs_p1 = args.epochs_p1
    epochs_p2 = args.epochs_p2
    device_indice = args.device

    torch.cuda.set_device(device_indice)
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    print('You will use:', device, str(device_indice))

    os.chdir(input_file)
    gene_peak = mmread("Gene_Peak.mtx").tocsr()
    peak_cell = mmread("Peak_Cell.mtx").tocsr()
    gene_cell = mmread("Gene_Cell.mtx").tocsr()
    gene_names = pd.read_csv('Gene_names.tsv', header=None)
    peak_names = pd.read_csv('Peak_names.tsv', header=None)

    peak_cell.obs_names = peak_names[0]
    gene_cell.obs_names = gene_names[0]
    gene_peak.obs_names = gene_names[0]
    gene_peak.var_names = peak_names[0]

    RNA_matrix = gene_cell
    ATAC_matrix = peak_cell
    RP_matrix = gene_peak
    Gene_Peak = gene_peak

    cell_num = RNA_matrix.shape[1]
    gene_num = RNA_matrix.shape[0]
    peak_num = ATAC_matrix.shape[0]

    initial_pre = init_cluster(RNA_matrix)
    ini_p1 = [int(i) for i in initial_pre]

    indices, Node_Ids, dic = subgraph_extract(RNA_matrix, ATAC_matrix, neighbor = [neighbor], cell_size=cell_size)
    np.save(output_file + "/indices.npy", indices, allow_pickle=True)
    np.save(output_file + "/Node_Ids.npy", Node_Ids)
    # indices = np.load(output_file + "/indices.npy",allow_pickle=True)
    # Node_Ids = np.load(output_file + "/Node_Ids.npy")
    n_batch = len(indices)

    node_model = MultimodalFeatureEncoder(RNA_matrix, ATAC_matrix, indices, ini_p1, hidden_dim=n_hid, num_heads=nheads, 
                                        num_layers=nlayers, label_smooth_rate=labsm, lr_rate=lr, weight_decay=wd, device=device, num_types=3, num_relations=2, epochs=epochs_p1)
    hgnn, cell_emb, gene_emb, peak_emb, h = node_model.train_model(num_batches=n_batch)

    OmicTrainer = IntegratedOmicTrainer(gnn_backbone=hgnn, h=h, label_smooth_rate=labsm, hidden_dim=n_hid, num_batches=n_batch, 
                                        device=device, lr_rate=lr, weight_decay=wd, num_epochs=epochs_p2, save_path=output_file)
    OmicTrainer_gnn = OmicTrainer.train_model(indices, RNA_matrix, ATAC_matrix, Gene_Peak, ini_p1)
    result = OmicTrainerPred(RNA_matrix, ATAC_matrix, RP_matrix, MarsGT_gnn=OmicTrainer_gnn, indices=indices, 
                        nodes_id=Node_Ids, cell_size=cell_size, device=device, gene_names=gene_names, peak_names=peak_names)

    np.save(output_file + "/pred.npy", result['pred_label'])
    np.save(output_file + "/cell_embedding.npy", result['cell_embedding'])
    np.save(output_file + "/gene_cell_embedding.npy", result['gene_cell_embedding'])
    np.save(output_file + "/peak_cell_embedding.npy", result['peak_cell_embedding'])

if __name__ == "__main__":
    main()