import math
import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg
import matplotlib.pyplot as plt

from common.abstract_recommender import GeneralRecommender


class NCL(nn.Module):
    def __init__(self, svc_neg_num=32, svc_temperature=0.1):
        super().__init__()
        self.svc_neg_num = svc_neg_num
        self.svc_temperature = svc_temperature

    def loss(self, str_embedding, vis_embedding):
        str_embedding = F.normalize(str_embedding, p=2, dim=-1, eps=1e-5)
        vis_embedding = F.normalize(vis_embedding, p=2, dim=-1, eps=1e-5)

        bs, _ = str_embedding.size()
        neg_sample_id = torch.randint(0, bs, [bs, self.svc_neg_num])
        neg_str_feat = str_embedding[neg_sample_id]
        neg_vis_feat = vis_embedding[neg_sample_id]
        str_samples = torch.cat([str_embedding.unsqueeze(1), neg_str_feat], 1)
        vis_samples = torch.cat([vis_embedding.unsqueeze(1), neg_vis_feat], 1)
        s2v_score = torch.matmul(vis_samples, str_embedding.unsqueeze(2)).squeeze(2) / self.svc_temperature
        v2s_score = torch.matmul(str_samples, vis_embedding.unsqueeze(2)).squeeze(2) / self.svc_temperature
        label = torch.zeros([bs, ], dtype=torch.long).to(str_embedding.device)
        s2v_loss = F.cross_entropy(s2v_score, label)
        v2s_loss = F.cross_entropy(v2s_score, label)
        svc_loss = s2v_loss + v2s_loss
        return svc_loss

    def forward(self, str_embedding, vis_embedding):
        svc_loss = self.loss(str_embedding, vis_embedding)
        total_loss = svc_loss
        return total_loss


class CoPARec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(CoPARec, self).__init__(config, dataset)
        self.current_epoch = None
        self.embedding_dim = config['embedding_size']
        self.gcn_layers = config['gcn_layers']
        self.knn_k = config['knn_k']
        self.dropout = config['dropout']
        self.tau = config['tau']
        self.m_alpha = config['m_alpha']
        self.lamda = config['lamda']
        self.save_intermediate = config['save_intermediate']
        self.save_path = config['save_path']
        self.NCL = NCL()

        self.vis_save_dir = os.path.join(self.save_path, "visualizations")
        os.makedirs(self.vis_save_dir, exist_ok=True)

        self.attention = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.intermediate_path = os.path.abspath(os.path.join(self.save_path, config['model'], config['dataset']))
        if not os.path.exists(self.intermediate_path):
            os.makedirs(self.intermediate_path)

        self.n_nodes = self.n_users + self.n_items
        interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.interaction_matrix = interaction_matrix
        self.a_similarity = self.precompute_a_similarity()
        self.ui_indices = torch.LongTensor(np.vstack((interaction_matrix.row, interaction_matrix.col))).to(self.device)
        items_in_interactions = self.ui_indices[1]
        item_degrees = torch.bincount(items_in_interactions, minlength=self.n_items).float()
        item_degrees[item_degrees == 0] = 0.00001
        self.item_degrees = item_degrees
        self.base_adj = self.get_base_adj(self.ui_indices.clone())
        self.v_feat_i = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        self.v_feat_u = self.cal_user_embedding_mean(self.v_feat)
        i_v_adj = self.get_knn_adj(self.v_feat_i.weight)
        u_v_adj = self.get_knn_adj(self.v_feat_u)
        self.t_feat_i = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        self.t_feat_u = self.cal_user_embedding_mean(self.t_feat)
        i_t_adj = self.get_knn_adj(self.t_feat_i.weight)
        u_t_adj = self.get_knn_adj(self.t_feat_u)
        self.base_ii = i_v_adj + i_t_adj
        self.base_uu = u_v_adj + u_t_adj

        self.i_pe = PE(self.embedding_dim, self.n_items, self.device)
        self.u_pe = PE(self.embedding_dim, self.n_users, self.device)
        self.i_edge_predictor = MLP_view(self.base_ii, self.embedding_dim, self.device)
        self.u_edge_predictor = MLP_view(self.base_uu, self.embedding_dim, self.device)
        self.v_mu = MLP(self.v_feat.size(-1), self.embedding_dim)
        self.t_mu = MLP(self.t_feat.size(-1), self.embedding_dim)
    def precompute_a_similarity(self):
        cache_file = os.path.join(self.intermediate_path, "a_similarity.pt")
        if os.path.exists(cache_file):
            return torch.load(cache_file)
        return a_similarity

    def cal_user_embedding_mean(self, embeddings):
        rows = self.ui_indices[0]
        cols = self.ui_indices[1]
        item_embeddings = embeddings[cols]
        weights = 1.0 + torch.log(0.01 / self.item_degrees[cols])  # (num_interactions,)
        weighted_item_embeddings = item_embeddings * weights.unsqueeze(1)
        user_embedding_sum = torch.zeros((self.n_users, embeddings.size(-1)), device=self.device)
        user_interaction_count = torch.zeros(self.n_users, device=self.device)
        user_embedding_sum.index_add_(0, rows, weighted_item_embeddings)
        user_interaction_count.index_add_(0, rows, weights)
        user_embedding_mean = user_embedding_sum / user_interaction_count.unsqueeze(1)
        user_embedding_mean = torch.nan_to_num(user_embedding_mean, nan=0.0, posinf=0.0, neginf=0.0)

        del user_embedding_sum, user_interaction_count
        torch.cuda.empty_cache()
        return user_embedding_mean

    def get_base_adj(self, ui_indices):
        adj_size = torch.Size((self.n_nodes, self.n_nodes))
        ui_indices[1] += self.n_users
        ui_graph = torch.sparse_coo_tensor(ui_indices, torch.ones_like(self.ui_indices[0], dtype=torch.float32),
                                           adj_size, device=self.device)
        iu_graph = ui_graph.T
        base_adj = ui_graph + iu_graph
        return base_adj

    def get_aug_adj_mat(self, base_adj, uu_graph, ii_graph):
        if uu_graph is None and ii_graph is None:
            return base_adj
        adj_size = torch.Size((self.n_nodes, self.n_nodes))

        uu_graph = torch.sparse_coo_tensor(uu_graph._indices(),
                                           uu_graph._values(), adj_size, device=self.device)
        ii_graph = torch.sparse_coo_tensor(ii_graph._indices() + self.n_users,
                                           ii_graph._values(), adj_size, device=self.device)
        aug_adj = uu_graph + base_adj + ii_graph
        return aug_adj

    def get_knn_adj(self, embeddings):
        context_norm = embeddings / torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        if sim.size() == self.a_similarity.size():
            combined_sim = (sim + self.a_similarity) / 2
        else:
            combined_sim = sim
        knn_val, knn_ind = torch.topk(combined_sim, self.knn_k, dim=-1)
        indices0 = torch.arange(knn_ind.size(0)).unsqueeze(1).expand(-1, self.knn_k).to(self.device)
        indices = torch.stack([indices0.flatten(), knn_ind.flatten()], dim=0)
        adj = torch.sparse_coo_tensor(indices, knn_val.flatten().squeeze(), combined_sim.size())

        return adj

    def cal_norm_laplacian(self, adj):
        indices = adj._indices()
        values = adj._values()

        row = indices[0]
        col = indices[1]
        rowsum = torch.sparse.sum(adj, dim=-1).to_dense()
        d_inv_sqrt = torch.pow(rowsum, -0.5)  # D^(-1/2)
        d_inv_sqrt = torch.clamp(d_inv_sqrt, 0.0, 10.0)

        row_inv_sqrt = d_inv_sqrt[row]
        col_inv_sqrt = d_inv_sqrt[col]
        values = values * row_inv_sqrt * col_inv_sqrt

        return torch.sparse_coo_tensor(indices, values, adj.shape)

    def sample_adj(self, adj, dropout):
        edge_value = adj._values()
        edge_value[torch.isnan(edge_value)] = 0.
        degree_len = int(edge_value.size(0) * (1. - dropout))
        degree_idx = torch.multinomial(edge_value, degree_len)
        keep_indices = adj._indices()[:, degree_idx]
        new_adj = torch.sparse_coo_tensor(keep_indices, edge_value[degree_idx], adj.shape)
        return self.cal_norm_laplacian(new_adj)

    def simple_gcn(self, ego_embeddings, norm_adj):
        all_embeddings = [ego_embeddings]
        for i in range(self.gcn_layers):
            side_embeddings = torch.sparse.mm(norm_adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        return all_embeddings

    def forward(self, training):
        i_v_mu = self.i_pe(self.v_mu(self.v_feat_i.weight))
        u_v_mu = self.u_pe(self.v_mu(self.v_feat_u))
        i_t_mu = self.i_pe(self.t_mu(self.t_feat_i.weight))
        u_t_mu = self.u_pe(self.t_mu(self.t_feat_u))
        u_embedding = self.m_alpha * u_t_mu + (1 - self.m_alpha) * u_v_mu
        i_embedding = self.m_alpha * i_t_mu + (1 - self.m_alpha) * i_v_mu
        similarity_matrix = torch.matmul(u_embedding, i_embedding.T)
        u_norm = torch.norm(u_embedding, p=2, dim=1, keepdim=True)
        i_norm = torch.norm(i_embedding, p=2, dim=1, keepdim=True)
        similarity_matrix = similarity_matrix / (u_norm * i_norm.T)
        top_k = 5
        similarity_scores = similarity_matrix
        topk_similarities, topk_indices = torch.topk(similarity_scores, k=top_k, largest=False)
        topk_item_embeddings = i_embedding[topk_indices]
        mean_topk_item_embeddings = topk_item_embeddings.mean(dim=1)
        mean_topk_item_embeddings = self.u_pe(mean_topk_item_embeddings)

        u_embedding = torch.cat([u_embedding, mean_topk_item_embeddings], dim=1)
        i_embedding_zero = torch.zeros_like(i_embedding)
        i_embedding = torch.cat([i_embedding, i_embedding_zero], dim=1)
        if self.training:
            ii_adj = self.i_edge_predictor(i_t_mu, i_v_mu)
            uu_adj = self.u_edge_predictor(u_t_mu, u_v_mu)
        else:
            ii_adj = self.get_knn_adj(i_embedding)
            uu_adj = self.get_knn_adj(u_embedding)
        adj = self.get_aug_adj_mat(self.base_adj, uu_adj, ii_adj).coalesce()

        if training and self.dropout > 0.:
            norm_adj = self.sample_adj(adj, self.dropout)
        else:
            norm_adj = self.cal_norm_laplacian(adj)

        all_g_embeddings = torch.cat([u_embedding, i_embedding], dim=0)
        all_g_embeddings = self.simple_gcn(all_g_embeddings, norm_adj)
        u_g_embedding, i_g_embedding = torch.split(all_g_embeddings, [self.n_users, self.n_items], dim=0)

        if training:
            return u_g_embedding, i_g_embedding, u_t_mu, i_t_mu, u_v_mu, i_v_mu
        else:
            return u_g_embedding, i_g_embedding, i_t_mu, i_v_mu

    def sl_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        bpr_loss = -torch.mean(maxi)

        return bpr_loss

    def infonce_loss(self, emb1, emb2):
        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)
        scores = torch.exp(torch.matmul(emb1, emb2.T) / self.tau)
        pos_sim = scores.diag()
        loss = -torch.log(pos_sim / torch.sum(scores, dim=1)).mean()
        return loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_g_embeddings, i_g_embeddings, u_t_mu, i_t_mu, u_v_mu, i_v_mu = self.forward(True)
        loss = self.NCL(u_t_mu,u_v_mu)
        loss_rec = self.sl_loss(u_g_embeddings[users], i_g_embeddings[pos_items], i_g_embeddings[neg_items])
        return loss_rec + self.lamda * loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        u_g_embeddings, i_g_embeddings, i_t_mu, i_v_mu = self.forward(False)
        if self.save_intermediate:
            torch.save(u_g_embeddings, os.path.join(self.intermediate_path, f'u_embeddings_{self.current_epoch}.pt'))
            torch.save(i_g_embeddings, os.path.join(self.intermediate_path, f'i_embeddings_{self.current_epoch}.pt'))
        u_embeddings = u_g_embeddings[user]
        scores = torch.matmul(u_embeddings, i_g_embeddings.transpose(0, 1))
        return scores

    def pre_epoch_processing(self, epoch_idx):
        self.current_epoch = epoch_idx


class MLP_view(nn.Module):
    def __init__(self, adj, in_dim, device):
        super(MLP_view, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU())
        self.edge_index = torch.clone(adj._indices())
        self.edge_val = torch.clone(adj._values())
        self.adj_size = adj.size()
        self.device = device

    def forward(self, Eu, Ev):
        transformed_u = self.fc1(Eu)
        transformed_v = self.fc2(Ev)

        src, dst = self.edge_index[0], self.edge_index[1]
        u_features = transformed_u[src]
        v_features = transformed_v[dst]
        distances = torch.norm(u_features - v_features, p=2, dim=1)

        similarities = torch.exp(distances)

        Ag = self.edge_val * torch.sigmoid(similarities)

        return torch.sparse_coo_tensor(self.edge_index, Ag, self.adj_size, device=self.device)


class PE(nn.Module):
    def __init__(self, pos_dim, max_len, device):
        super(PE, self).__init__()
        self.pos_dim = pos_dim
        self.max_len = max_len
        self.device = device
        self.pe = torch.zeros(max_len, pos_dim, device=self.device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, pos_dim, 2).float() * (-math.log(10000.0) / pos_dim))

        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        x = x + self.pe
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=1, activation="tanh", layer_norm=True):
        super(MLP, self).__init__()
        self.fcs = nn.ModuleList()
        activation_layer = nn.Tanh()

        if input_dim <= output_dim:
            for i in range(num_layers):
                self.fcs.append(nn.Linear(input_dim, output_dim))
                if activation_layer is not None:
                    self.fcs.append(activation_layer)
                if layer_norm:
                    self.fcs.append(nn.LayerNorm(output_dim))

        else:
            step_ratio = (output_dim / input_dim) ** (1 / num_layers)
            dims = [input_dim]
            for i in range(1, num_layers):
                current_dim = input_dim * (step_ratio ** i)
                hidden_dim = self.next_power_of_two(current_dim)
                self.fcs.append(nn.Linear(dims[-1], hidden_dim))
                if activation_layer is not None:
                    self.fcs.append(activation_layer)
                if layer_norm:
                    self.fcs.append(nn.LayerNorm(hidden_dim))

                dims.append(hidden_dim)

            self.fcs.append(nn.Linear(dims[-1], output_dim))
            if activation_layer is not None:
                self.fcs.append(activation_layer)
            if layer_norm:
                self.fcs.append(nn.LayerNorm(output_dim))

    def forward(self, x):
        for layer in self.fcs:
            x = layer(x)
        return x

    def next_power_of_two(self, n):
        return 2 ** math.ceil(math.log2(n))
