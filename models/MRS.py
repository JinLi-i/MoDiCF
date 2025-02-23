import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F

class MRS(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, head_num,
                 isSparse, layers, model_cat_rate, id_cat_rate, modalities, input_size):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_ui_layers = len(weight_size)
        self.weight_size = [embedding_dim] + weight_size
        self.n_modalities = len(modalities)
        self.modalities = modalities
        self.head_num = head_num
        self.isSparse = isSparse
        self.model_cat_rate = model_cat_rate
        self.id_cat_rate = id_cat_rate
        self.layers = layers
        self.input_size = input_size

        self.encoder = nn.ModuleDict()

        for i in range(self.n_modalities):
            branch_name = modalities[i] + '_enc'
            mid = int(self.input_size[i]/2)
            if self.embedding_dim < mid:
                mid = self.embedding_dim
            encoder = nn.Sequential(
                nn.Linear(self.input_size[i], mid),
                nn.LeakyReLU(),
                nn.Linear(mid, self.embedding_dim),
                nn.LeakyReLU(),
            )
            self.encoder[branch_name] = encoder

            for module in encoder:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)

        common_trans = nn.Linear(self.embedding_dim, self.embedding_dim)
        nn.init.xavier_uniform_(common_trans.weight)
        self.align = nn.ModuleDict()
        self.align['common_trans'] = common_trans

        self.user_id_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.softmax = nn.Softmax(dim=-1)
        self.act = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm1d(self.embedding_dim)
        self.tau = 0.5

        initializer = nn.init.xavier_uniform_

        self.weight_dict = nn.ParameterDict({
            'w_q': nn.Parameter(initializer(torch.empty([self.embedding_dim, self.embedding_dim]))),
            'w_k': nn.Parameter(initializer(torch.empty([self.embedding_dim, self.embedding_dim]))),
            'w_v': nn.Parameter(initializer(torch.empty([self.embedding_dim, self.embedding_dim]))),
            'w_self_attention_item': nn.Parameter(initializer(torch.empty([self.embedding_dim, self.embedding_dim]))),
            'w_self_attention_user': nn.Parameter(initializer(torch.empty([self.embedding_dim, self.embedding_dim]))),
            'w_self_attention_cat': nn.Parameter(
                initializer(torch.empty([self.head_num * self.embedding_dim, self.embedding_dim]))),
        })

        self.embedding_dict = {'user': {}, 'item': {}}

    def mm(self, x, y):
        if self.isSparse:
            res = torch.sparse.mm(x, y)
            return res
        else:
            return torch.mm(x, y)

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def batched_contrastive_loss(self, z1, z2, batch_size=4096):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))
            between_sim = f(self.sim(z1[mask], z2))

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        loss_vec = torch.cat(losses)
        return loss_vec.mean()

    def csr_norm(self, csr_mat, mean_flag=False):
        rowsum = np.array(csr_mat.sum(1))
        rowsum = np.power(rowsum + 1e-8, -0.5).flatten()
        rowsum[np.isinf(rowsum)] = 0.
        rowsum_diag = sp.diags(rowsum)

        colsum = np.array(csr_mat.sum(0))
        colsum = np.power(colsum + 1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.
        colsum_diag = sp.diags(colsum)

        if mean_flag == False:
            return rowsum_diag * csr_mat * colsum_diag
        else:
            return rowsum_diag * csr_mat

    def matrix_to_tensor(self, cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))
        values = torch.from_numpy(cur_matrix.data)
        shape = torch.Size(cur_matrix.shape)

        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()

    def para_dict_to_tenser(self, para_dict):
        tensors = []

        for beh in para_dict.keys():
            tensors.append(para_dict[beh])
        tensors = torch.stack(tensors, dim=0)

        return tensors

    def multi_head_self_attention(self, trans_w, embedding_t_1, embedding_t):

        q = self.para_dict_to_tenser(embedding_t)
        v = k = self.para_dict_to_tenser(embedding_t_1)
        beh, N, d_h = q.shape[0], q.shape[1], self.embedding_dim / self.head_num

        Q = torch.matmul(q, trans_w['w_q'])
        K = torch.matmul(k, trans_w['w_k'])
        V = v

        Q = Q.reshape(beh, N, self.head_num, int(d_h)).permute(2, 0, 1, 3)
        K = Q.reshape(beh, N, self.head_num, int(d_h)).permute(2, 0, 1, 3)

        Q = torch.unsqueeze(Q, 2)
        K = torch.unsqueeze(K, 1)
        V = torch.unsqueeze(V, 1)

        att = torch.mul(Q, K) / torch.sqrt(torch.tensor(d_h))
        att = torch.sum(att, dim=-1)
        att = torch.unsqueeze(att, dim=-1)
        att = F.softmax(att, dim=2)

        Z = torch.mul(att, V)
        Z = torch.sum(Z, dim=2)

        Z_list = [value for value in Z]
        Z = torch.cat(Z_list, -1)
        Z = torch.matmul(Z, self.weight_dict['w_self_attention_cat'])

        self.model_cat_rate * F.normalize(Z, p=2, dim=2)
        return Z, att.detach()

    def forward(self, ui_graph, iu_graph, mm_ui_graph, mm_iu_graph, mm_feats):
        mm_enc_feats_list = []
        mm_user_feats_list = []
        mm_item_feats_list = []
        mm_user_id_list = []
        mm_item_id_list = []
        for m in range(self.n_modalities):
            feats = mm_feats[m]
            mm_item_feats = self.encoder[self.modalities[m] + '_enc'](feats)
            mm_enc_feats_list.append(mm_item_feats)

            # TODO: code problem in MMSSL
            for i in range(self.layers):
                mm_user_feats = self.mm(ui_graph, mm_item_feats)
                mm_item_feats = self.mm(iu_graph, mm_user_feats)
                mm_user_id = self.mm(mm_ui_graph[m], self.item_id_embedding.weight)
                mm_item_id = self.mm(mm_iu_graph[m], self.user_id_embedding.weight)
            mm_user_feats_list.append(mm_user_feats)
            mm_item_feats_list.append(mm_item_feats)

            self.embedding_dict['user'][self.modalities[m]] = mm_user_id
            self.embedding_dict['item'][self.modalities[m]] = mm_item_id
            mm_user_id_list.append(mm_user_id)
            mm_item_id_list.append(mm_item_id)

        user_z, _ = self.multi_head_self_attention(self.weight_dict, self.embedding_dict['user'],
                                                   self.embedding_dict['user'])
        item_z, _ = self.multi_head_self_attention(self.weight_dict, self.embedding_dict['item'],
                                                   self.embedding_dict['item'])
        user_emb = user_z.mean(0)
        item_emb = item_z.mean(0)
        u_g_embeddings = self.user_id_embedding.weight + self.id_cat_rate * F.normalize(user_emb, p=2, dim=1)
        i_g_embeddings = self.item_id_embedding.weight + self.id_cat_rate * F.normalize(item_emb, p=2, dim=1)

        user_emb_list = [u_g_embeddings]
        item_emb_list = [i_g_embeddings]
        for i in range(self.n_ui_layers):
            if i == (self.n_ui_layers - 1):
                u_g_embeddings = self.softmax(torch.mm(ui_graph, i_g_embeddings))
                i_g_embeddings = self.softmax(torch.mm(iu_graph, u_g_embeddings))

            else:
                u_g_embeddings = torch.mm(ui_graph, i_g_embeddings)
                i_g_embeddings = torch.mm(iu_graph, u_g_embeddings)

            user_emb_list.append(u_g_embeddings)
            item_emb_list.append(i_g_embeddings)

        u_g_embeddings = torch.mean(torch.stack(user_emb_list), dim=0)
        i_g_embeddings = torch.mean(torch.stack(item_emb_list), dim=0)
        for m in range(self.n_modalities):
            u_g_embeddings = u_g_embeddings + self.model_cat_rate*F.normalize(mm_user_feats_list[m], p=2, dim=1)
            i_g_embeddings = i_g_embeddings + self.model_cat_rate*F.normalize(mm_item_feats_list[m], p=2, dim=1)

        return (u_g_embeddings, i_g_embeddings, mm_item_feats_list, mm_user_feats_list,
                u_g_embeddings, i_g_embeddings, mm_user_id_list, mm_item_id_list)
