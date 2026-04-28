import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import *
from .tools import *

def z_score_normalize(x):
    mean = x._values().mean()
    std = x._values().std()
    return (x - mean) / (std + 1e-8)

def log_normalize(sparse_tensor):
    return torch.sparse_coo_tensor(
        sparse_tensor.indices(),
        torch.log1p(sparse_tensor.values()),
        sparse_tensor.size()
    )

class HGNN(nn.Module):
    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers, dropout=0.2, conv_name='hgt',
                 prev_norm=True, last_norm=True):
        super(HGNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.adapt_ws = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        self.embedding1 = nn.ModuleList()

        # Initialize MLP weight matrices
        for ti in range(num_types):
            self.embedding1.append(nn.Linear(in_dim[ti], 256))

        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(256, n_hid))

        # Initialize graph convolution layers
        for l in range(n_layers - 1):
            self.gcs.append(
                GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm=prev_norm))
        self.gcs.append(
            GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm=last_norm))

    def encode(self, x, t_id):
        h1 = F.relu(self.embedding1[t_id](x))
        return h1

    def forward(self, node_feature, node_type, edge_index, edge_type):
        node_embedding = []
        for t_id in range(self.num_types):
            node_embedding += list(self.encode(node_feature[t_id], t_id))

        node_embedding = torch.stack(node_embedding) 

        res = torch.zeros(node_embedding.size(0), self.n_hid).to(node_feature[0].device)
        # Process each node type
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            # Update result matrix
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_embedding[idx]))

        # Apply dropout to the result matrix
        meta_xs = self.drop(res) 
        del res

        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type)
        return meta_xs 


class Net(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x

class MultimodalFeatureEncoder(nn.Module):
    def __init__(self, rna_mat, atac_mat, batch_indices, init_labels, 
                 hidden_dim, num_heads, num_layers, label_smooth_rate, 
                 lr_rate, weight_decay, device, 
                 num_types=3, num_relations=2, epochs=1):
        super(MultimodalFeatureEncoder, self).__init__()
        
        self.rna_mat = rna_mat
        self.atac_mat = atac_mat
        self.batch_indices = batch_indices
        self.init_labels = np.array(init_labels)
        self.device = device
        self.total_epochs = epochs
        
        input_dims = [rna_mat.shape[0], rna_mat.shape[1], atac_mat.shape[1]]
        
        self.label_smoothing_loss = LabelSmoothing(label_smooth_rate)
        self.gnn_encoder = HGNN(
            in_dim=input_dims,
            n_hid=hidden_dim,
            num_types=num_types,
            num_relations=num_relations,
            n_heads=num_heads,
            n_layers=num_layers,
            dropout=0.3
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.gnn_encoder.parameters(), 
            lr=lr_rate, 
            weight_decay=weight_decay
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

    def _build_subgraph_tensors(self, idx_c, idx_g, idx_p):
        n_c, n_g, n_p = len(idx_c), len(idx_g), len(idx_p)
        
        feat_g = torch.tensor(self.rna_mat[idx_g, :].todense(), dtype=torch.float32)
        feat_c = torch.tensor(self.rna_mat[:, idx_c].T.todense(), dtype=torch.float32)
        feat_p = torch.tensor(self.atac_mat[idx_p, :].todense(), dtype=torch.float32)
        node_features = [feat_c.to(self.device), feat_g.to(self.device), feat_p.to(self.device)]

        sub_rna = self.rna_mat[idx_g, :][:, idx_c]
        sub_atac = self.atac_mat[idx_p, :][:, idx_c]

        # Edge Index
        # 0~n_c-1 Cell, n_c~n_c+n_g-1 Gene, n_c+n_g~... Peak
        rna_row, rna_col = sub_rna.nonzero()
        atac_row, atac_col = sub_atac.nonzero()
        
        # Gene-Cell edges
        g_nodes = torch.tensor(rna_row + n_c, dtype=torch.long)
        c_nodes_rna = torch.tensor(rna_col, dtype=torch.long)
        # Peak-Cell edges
        p_nodes = torch.tensor(atac_row + n_c + n_g, dtype=torch.long)
        c_nodes_atac = torch.tensor(atac_col, dtype=torch.long)

        edge_index = torch.cat([
            torch.stack([g_nodes, c_nodes_rna]),         # Gene -> Cell
            torch.stack([c_nodes_rna, g_nodes]),         # Cell -> Gene
            torch.stack([p_nodes, c_nodes_atac]),        # Peak -> Cell
            torch.stack([c_nodes_atac, p_nodes])         # Cell -> Peak
        ], dim=1).to(self.device)

        # Edge Type
        edge_type = torch.cat([
            torch.zeros(len(rna_row)),                   # 0: Gene -> Cell
            torch.ones(len(rna_row)),                    # 1: Cell -> Gene
            torch.full((len(atac_row),), 2.0),           # 2: Peak -> Cell
            torch.full((len(atac_row),), 3.0)            # 3: Cell -> Peak
        ]).long().to(self.device)

        # Node Type
        node_type = torch.cat([
            torch.zeros(n_c), 
            torch.ones(n_g), 
            torch.full((n_p,), 2.0)
        ]).long().to(self.device)

        return node_features, edge_index, edge_type, node_type, sub_rna, sub_atac

    def _compute_cosine_penalty(self, embeddings, labels):
        total_sim = 0.0
        unique_classes = torch.unique(labels)
        
        for cls in unique_classes:
            cluster_embs = embeddings[labels == cls]
            if cluster_embs.shape[0] > 0:
                sim_matrix = F.cosine_similarity(
                    cluster_embs.unsqueeze(1), 
                    cluster_embs.unsqueeze(0), 
                    dim=2
                )
                total_sim += sim_matrix.mean()
                
        return total_sim

    def train_model(self, num_batches):
        print('Multimodal Feature Encoder training initiated. Please wait...')
        
        emb_c, emb_g, emb_p, h_vars = None, None, None, None
        
        for epoch in tqdm(range(self.total_epochs), desc="Epochs"):
            for b_id in range(num_batches):
                batch_info = self.batch_indices[b_id]
                idx_g, idx_c, idx_p = batch_info['gene_index'], batch_info['cell_index'], batch_info['peak_index']
                
                node_feats, edge_idx, edge_typ, node_typ, sub_rna, sub_atac = self._build_subgraph_tensors(idx_c, idx_g, idx_p)
                batch_labels = torch.tensor(self.init_labels[idx_c], dtype=torch.long).to(self.device)

                reps = self.gnn_encoder(node_feats, node_typ, edge_idx, edge_typ)
                emb_c = reps[node_typ == 0]
                emb_g = reps[node_typ == 1]
                emb_p = reps[node_typ == 2]

                pred_rna = F.log_softmax(torch.mm(emb_g, emb_c.t()), dim=-1)
                pred_atac = F.log_softmax(torch.mm(emb_p, emb_c.t()), dim=-1)

                true_rna = F.softmax(torch.tensor(sub_rna.todense(), dtype=torch.float32).to(self.device), dim=-1)
                true_atac = F.softmax(torch.tensor(sub_atac.todense(), dtype=torch.float32).to(self.device), dim=-1)

                kl_loss_rna = F.kl_div(pred_rna, true_rna, reduction='mean')
                kl_loss_atac = F.kl_div(pred_atac, true_atac, reduction='mean')
                total_kl_loss = kl_loss_rna + kl_loss_atac

                cluster_loss = self.label_smoothing_loss(emb_c, batch_labels)
                cos_penalty = self._compute_cosine_penalty(emb_c, batch_labels)

                final_loss = cluster_loss - cos_penalty
                
                self.optimizer.zero_grad()
                final_loss.backward()
                self.optimizer.step()
                
                h_vars = emb_c

        print('Multimodal Feature Encoder training completed successfully.')
        return self.gnn_encoder, emb_c, emb_g, emb_p, h_vars
    

class IntegratedOmicTrainer(nn.Module):
    def __init__(self, gnn_backbone, h, label_smooth_rate, hidden_dim, 
                 num_batches, device, lr_rate, weight_decay, 
                 num_epochs, save_path):
        super(IntegratedOmicTrainer, self).__init__()
        
        self.device = device
        self.max_epochs = num_epochs
        self.n_batches = num_batches
        self.hid_dim = hidden_dim
        self.checkpoint_dir = save_path
        
        self.gnn_extractor = gnn_backbone
        self.prior_h = h
        
        self.interaction_decoder = nn.Sequential(
            nn.Linear(2 * self.hid_dim, self.hid_dim),
            nn.ReLU()
        ).to(self.device)
        
        self.cluster_smoother = LabelSmoothing(label_smooth_rate)  
        self.opt_gnn = torch.optim.AdamW(self.gnn_extractor.parameters(), lr=lr_rate, weight_decay=weight_decay)
        self.opt_decoder = torch.optim.AdamW(self.interaction_decoder.parameters(), lr=1e-2)

    @staticmethod
    def _build_expanded_gene_sparse(peak_mat, gene_mat, device):
        g_feat = torch.log1p(gene_mat)
        p_feat = torch.log1p(peak_mat)
        
        n_g, n_c = g_feat.shape
        n_p = p_feat.shape[0]
        
        g_row, g_col = torch.nonzero(g_feat, as_tuple=True)
        g_val = g_feat[g_row, g_col]
        nnz = g_row.shape[0]
        
        if nnz == 0:
            return torch.sparse_coo_tensor(torch.empty((2,0)), torch.empty(0), (n_g * n_p, n_c)).to(device)

        peak_offsets = (torch.arange(n_p, device=device) * n_g).unsqueeze(1) # [n_p, 1]
        base_g_rows = g_row.unsqueeze(0)                                     # [1, nnz]
        
        expanded_rows = (peak_offsets + base_g_rows).flatten()               # [n_p * nnz]
        expanded_cols = g_col.unsqueeze(0).expand(n_p, -1).flatten()         # [n_p * nnz]
        expanded_vals = g_val.unsqueeze(0).expand(n_p, -1).flatten()         # [n_p * nnz]
        
        return torch.sparse_coo_tensor(
            torch.stack([expanded_rows, expanded_cols]), 
            expanded_vals, 
            size=(n_g * n_p, n_c)
        ).to(device)

    @staticmethod
    def _build_expanded_peak_sparse(peak_mat, gene_mat, device):
        g_feat = torch.log1p(gene_mat)
        p_feat = torch.log1p(peak_mat)
        
        n_g = g_feat.shape[0]
        n_p, n_c = p_feat.shape
        
        p_row, p_col = torch.nonzero(p_feat, as_tuple=True)
        p_val = p_feat[p_row, p_col]
        nnz = p_row.shape[0]
        
        if nnz == 0:
            return torch.sparse_coo_tensor(torch.empty((2,0)), torch.empty(0), (n_g * n_p, n_c)).to(device)

        base_offsets = (p_row * n_g).unsqueeze(1)                   # [nnz, 1]
        gene_offsets = torch.arange(n_g, device=device).unsqueeze(0) # [1, n_g]
        
        expanded_rows = (base_offsets + gene_offsets).flatten()      # [nnz * n_g]
        expanded_cols = p_col.unsqueeze(1).expand(-1, n_g).flatten() # [nnz * n_g]
        expanded_vals = p_val.unsqueeze(1).expand(-1, n_g).flatten() # [nnz * n_g]
        
        return torch.sparse_coo_tensor(
            torch.stack([expanded_rows, expanded_cols]), 
            expanded_vals, 
            size=(n_g * n_p, n_c)
        ).to(device)
    
    def _get_structural_prior(self, gp_matrix, g_idx, p_idx, n_c):
        n_g, n_p = len(g_idx), len(p_idx)
        local_gp = gp_matrix[g_idx, :][:, p_idx].reshape(n_g * n_p, 1).tocoo()
        
        if local_gp.nnz == 0:
            return torch.sparse_coo_tensor(torch.empty((2,0)), torch.empty(0), (n_g * n_p, n_c)).to(self.device)

        base_rows = torch.tensor(local_gp.row, dtype=torch.long, device=self.device)
        base_data = torch.tensor(local_gp.data, dtype=torch.float32, device=self.device)
        
        rep_rows = base_rows.unsqueeze(1).expand(-1, n_c).flatten()
        rep_cols = torch.arange(n_c, device=self.device).unsqueeze(0).expand(local_gp.nnz, -1).flatten()
        rep_data = base_data.unsqueeze(1).expand(-1, n_c).flatten()
        
        return torch.sparse_coo_tensor(
            torch.stack([rep_rows, rep_cols]), 
            rep_data, 
            size=(n_g * n_p, n_c)
        ).to(self.device)

    def forward(self, batch_indices, rna_full, atac_full, gp_full, prior_labels):
        labels_arr = np.array(prior_labels)

        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            pbar = tqdm(range(self.n_batches), desc=f"Epoch {epoch+1}/{self.max_epochs}")
            
            for b_id in pbar:
                meta = batch_indices[b_id]
                idx_g, idx_c, idx_p = meta['gene_index'], meta['cell_index'], meta['peak_index']
                
                sub_rna = rna_full[idx_g, :][:, idx_c]
                sub_atac = atac_full[idx_p, :][:, idx_c]
                n_g, n_c, n_p = len(idx_g), len(idx_c), len(idx_p)
                
                feat_g = torch.tensor(rna_full[idx_g, :].todense(), dtype=torch.float32).to(self.device)
                feat_c = torch.tensor(rna_full[:, idx_c].T.todense(), dtype=torch.float32).to(self.device)
                feat_p = torch.tensor(atac_full[idx_p, :].todense(), dtype=torch.float32).to(self.device)
                
                target_rna = torch.tensor(sub_rna.todense(), dtype=torch.float32).to(self.device)
                target_atac = torch.tensor(sub_atac.todense(), dtype=torch.float32).to(self.device)
                
                r_row, r_col = sub_rna.nonzero()
                a_row, a_col = sub_atac.nonzero()
                
                edges = torch.cat([
                    torch.stack([torch.tensor(r_row + n_c), torch.tensor(r_col)]),
                    torch.stack([torch.tensor(r_col), torch.tensor(r_row + n_c)]),
                    torch.stack([torch.tensor(a_row + n_c + n_g), torch.tensor(a_col)]),
                    torch.stack([torch.tensor(a_col), torch.tensor(a_row + n_c + n_g)])
                ], dim=1).long().to(self.device)
                
                n_types = torch.cat([torch.zeros(n_c), torch.ones(n_g), torch.full((n_p,), 2.0)]).long().to(self.device)
                e_types = torch.cat([torch.zeros(len(r_row)), torch.ones(len(r_row)), 
                                     torch.full((len(a_row),), 2.0), torch.full((len(a_row),), 3.0)]).long().to(self.device)
                
                reps = self.gnn_extractor([feat_c, feat_g, feat_p], n_types, edges, e_types)
                emb_c, emb_g, emb_p = reps[n_types == 0], reps[n_types == 1], reps[n_types == 2]
                
                pred_rna = F.log_softmax(torch.mm(emb_g, emb_c.t()), dim=-1)
                pred_atac = F.log_softmax(torch.mm(emb_p, emb_c.t()), dim=-1)
                
                loss_rna = F.kl_div(pred_rna, F.softmax(target_rna, dim=-1), reduction='mean')
                loss_atac = F.kl_div(pred_atac, F.softmax(target_atac, dim=-1), reduction='mean')
                
                target_labels = torch.tensor(labels_arr[idx_c], dtype=torch.long).to(self.device)
                loss_clust = self.cluster_smoother(emb_c, target_labels)
                
                sparse_g = self._build_expanded_gene_sparse(target_atac, target_rna, self.device)
                sparse_p = self._build_expanded_peak_sparse(target_atac, target_rna, self.device)
                
                obs_inter = sparse_g.mul(sparse_p) 
                
                weight_a = self._get_structural_prior(gp_full, idx_g, idx_p, n_c)
                weighted_obs = obs_inter.mul(weight_a)
                
                clustered_obs = torch.sparse.mm(weighted_obs, emb_c) / 10000.0
                
                g_rep_expanded = emb_g.repeat_interleave(n_p, dim=0)
                p_rep_expanded = emb_p.repeat(n_g, 1)
                pred_egrn = self.interaction_decoder(torch.cat((g_rep_expanded, p_rep_expanded), dim=1))
                
                loss_egrn = F.kl_div(F.log_softmax(pred_egrn, dim=-1), 
                                     F.softmax(clustered_obs, dim=-1), reduction='mean')
                
                total_loss = loss_rna + loss_atac + loss_clust + loss_egrn
                
                self.opt_gnn.zero_grad()
                self.opt_decoder.zero_grad()
                total_loss.backward()
                self.opt_gnn.step()
                self.opt_decoder.step()
                
                epoch_loss += total_loss.item()
                pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})
                
            print(f"Epoch {epoch+1} Avg Loss: {epoch_loss / self.n_batches:.4f}")
            # if (epoch + 1) % 10 == 0:
            #     torch.save(self.gnn_extractor.state_dict(), f"{self.checkpoint_dir}/gnn_model_ep{epoch}.pth")
                
        return self.gnn_extractor

    def train_model(self, batch_meta, rna, atac, gp, labels):
        self.train()
        print("Starting Omics Integration Network Training...")
        return self.forward(batch_meta, rna, atac, gp, labels)
    

def OmicTrainerPred(RNA_matrix, ATAC_matrix, RP_matrix, MarsGT_gnn, indices, nodes_id, cell_size, device, gene_names,
                peak_names):
    n_batch = math.ceil(nodes_id.shape[0] / cell_size)
    embedding = []
    gene_embedding = []
    peak_embedding = []
    l_pre = []
    MarsGT_result = {}
    with torch.no_grad():
        for batch_id in tqdm(range(n_batch)):
            gene_index = indices[batch_id]['gene_index']
            cell_index = indices[batch_id]['cell_index']
            peak_index = indices[batch_id]['peak_index']

            gene_feature = RNA_matrix[list(gene_index),]
            cell_feature = RNA_matrix[:, list(cell_index)].T
            peak_feature = ATAC_matrix[list(peak_index),]
            gene_feature = torch.tensor(np.array(gene_feature.todense()), dtype=torch.float32).to(device)
            cell_feature = torch.tensor(np.array(cell_feature.todense()), dtype=torch.float32).to(device)
            peak_feature = torch.tensor(np.array(peak_feature.todense()), dtype=torch.float32).to(device)
            node_feature = [cell_feature, gene_feature, peak_feature]
            gene_cell_sub = RNA_matrix[list(gene_index),][:, list(cell_index)]
            peak_cell_sub = ATAC_matrix[list(peak_index),][:, list(cell_index)]
            gene_cell_edge_index = torch.LongTensor(
                [np.nonzero(gene_cell_sub)[0] + gene_cell_sub.shape[1], np.nonzero(gene_cell_sub)[1]]).to(device)
            peak_cell_edge_index = torch.LongTensor(
                [np.nonzero(peak_cell_sub)[0] + gene_cell_sub.shape[0] + gene_cell_sub.shape[1],
                 np.nonzero(peak_cell_sub)[1]]).to(device)
            edge_index = torch.cat((gene_cell_edge_index, peak_cell_edge_index), dim=1)
            node_type = torch.LongTensor(np.array(
                list(np.zeros(len(cell_index))) + list(np.ones(len(gene_index))) + list(
                    np.ones(len(peak_index)) * 2))).to(device)
            edge_type = torch.LongTensor(np.array(
                list(np.zeros(gene_cell_edge_index.shape[1])) + list(np.ones(peak_cell_edge_index.shape[1])))).to(
                device)
            node_rep = MarsGT_gnn.forward(node_feature, node_type,
                                          edge_index,
                                          edge_type).to(device)
            cell_emb = node_rep[node_type == 0]
            gene_emb = node_rep[node_type == 1]
            peak_emb = node_rep[node_type == 2]

            if device.type == "cuda":
                cell_emb = cell_emb.cpu()
                gene_emb = gene_emb.cpu()
                peak_emb = peak_emb.cpu()
            embedding.append(cell_emb.detach().numpy())
            gene_embedding.append(gene_emb.detach().numpy())
            peak_embedding.append(peak_emb.detach().numpy())

            cell_pre = list(cell_emb.argmax(dim=1).detach().numpy())
            l_pre.extend(cell_pre)

    cell_embedding = np.vstack(embedding)
    cell_clu = np.array(l_pre)
    #print(cell_embedding.shape, cell_clu.shape)

    gene_embedding = np.vstack(gene_embedding)
    peak_embedding = np.vstack(peak_embedding)
    gene_cell_embed = np.matmul(gene_embedding, cell_embedding.T)
    peak_cell_embed = np.matmul(peak_embedding, cell_embedding.T)
    #print(gene_cell_embed.shape, peak_cell_embed.shape)
    
    MarsGT_result = {'pred_label': cell_clu, 'cell_embedding': cell_embedding, 'gene_cell_embedding': gene_cell_embed, 'peak_cell_embedding': peak_cell_embed}
    return MarsGT_result