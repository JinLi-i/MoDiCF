import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import random as rd

import torch
import numpy as np
import pickle

from datetime import datetime
from models import *
from parser import parse_args

from utils import *

from data import *
from scipy.sparse import csr_matrix
from torch import autograd

isLoad = True
args = parse_args()

def seed_everything(seed=0):
    rd.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Trainer:
    def __init__(self, ori_mm_data, incomplete_mm_data, missing_items, indicator, input_size, modalities,
                 MRS_data, lr, lr_decay, batch_size, epoch, diff_model, MR, complete,
                 lambda_rec, device=torch.device("cpu"), min_data=None, max_data=None):
        self.ori_mm_data = ori_mm_data
        self.incomplete_mm_data = incomplete_mm_data
        self.missing_items = missing_items
        self.indicator = indicator
        self.n_modalities = len(ori_mm_data)
        self.n_sample = ori_mm_data[0].shape[0]
        self.lr = lr
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.epoch = epoch
        self.device = device
        self.diff_model = diff_model.to(device)
        self.lambda_rec = lambda_rec
        if type(min_data) is int:
            self.min_data = [min_data] * self.n_modalities
            self.max_data = [max_data] * self.n_modalities
        else:
            self.min_data = min_data
            self.max_data = max_data
        self.modalities = modalities
        self.input_size = input_size

        self.MRS_data = MRS_data
        self.dataset = args.dataset
        if isLoad:
            completed_data_file = args.load_dir + 'completed_data.pkl'
            if os.path.exists(completed_data_file):
                with open(args.load_dir + 'completed_data.pkl', 'rb') as f:
                    complete_mm_data = pickle.load(f)
                print("Completed data loaded from: " + args.load_dir + 'completed_data.pkl')
            else:
                complete_mm_data, eval_loss = self.evaluate(next_data=incomplete_mm_data)
                print("Complete data generated. Eval Loss: {:.4f}".format(eval_loss))
            self.complete_mm_data = [torch.tensor(complete_mm_data[i]).to(device) for i in range(self.n_modalities)]
        else:
            self.complete_mm_data = incomplete_mm_data

        self.Ks = eval(args.Ks)
        self.test_flag = args.test_flag
        self.tester = MRS_test(data=self.MRS_data, batch_size=batch_size, test_flag=self.test_flag,
                               Ks=self.Ks, incomplete_item_num=len(missing_items))
        self.verbose = args.verbose
        self.anchor_rate = args.anchor_rate
        self.sample_num_ii = args.sample_num_ii

        # MRS parameters
        self.weight_size = eval(args.weight_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.emb_dim = args.gnn_embed_size
        self.isSparse = args.isSparse
        self.model_cat_rate = args.delta
        self.head_num = args.H
        self.id_cat_rate = args.eta
        self.layers = args.L
        self.T = args.T
        self.m_topk_rate = args.m_topk_rate
        self.cl_rate = args.lambda_1
        self.feat_reg_decay = args.lambda_2
        self.tau = args.tau
        self.embed_size = args.embed_size

        self.ui_graph = self.ui_graph_raw = pickle.load(open('./data/' + self.dataset + '/train_mat','rb'))
        self.mm_ui_index = [{'x': [], 'y': []} for _ in range(self.n_modalities)]
        self.n_users = self.ui_graph.shape[0]
        self.n_items = self.ui_graph.shape[1]
        self.iu_graph = self.ui_graph.T
        self.ui_graph = self.matrix_to_tensor(self.csr_norm(self.ui_graph, mean_flag=True))
        self.iu_graph = self.matrix_to_tensor(self.csr_norm(self.iu_graph, mean_flag=True))
        self.mm_ui_graph = [self.ui_graph for _ in range(self.n_modalities)]
        self.mm_iu_graph = [self.iu_graph for _ in range(self.n_modalities)]
        self.MRS_model = MRS(n_users=self.n_users, n_items=self.n_items, embedding_dim=self.emb_dim,
                         weight_size=self.weight_size, head_num=self.head_num, isSparse=self.isSparse,
                         layers=self.layers, model_cat_rate=self.model_cat_rate,
                         id_cat_rate=self.id_cat_rate, modalities=self.modalities,
                         input_size=self.input_size).to(device)

        self.IRS_model = ItemRS(input_size=self.input_size, modalities=self.modalities,
                                embedding_dim=self.emb_dim).to(device)
        self.counterfactual_coeff = args.gamma
        self.c_list = None
        self.alpha = args.alpha_2

        self.optimizer = torch.optim.Adam(list(self.MRS_model.parameters()) +
                                          list(self.IRS_model.parameters()),
                                          lr=self.lr)
        self.diff_optimizer = torch.optim.Adam(self.diff_model.parameters(), lr=self.lr)
        if self.lr_decay > 0:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epoch, eta_min=self.lr*self.lr_decay)
            self.diff_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.diff_optimizer, T_max=self.epoch, eta_min=self.lr*self.lr_decay)
        else:
            self.scheduler = None
            self.diff_scheduler = None

        print("Missing rate = {:2f}, complete strategy: {}".format(MR, complete))

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
            cur_matrix = cur_matrix.tocoo()  #
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  #
        values = torch.from_numpy(cur_matrix.data)  #
        shape = torch.Size(cur_matrix.shape)

        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()  #

    def innerProduct(self, u_pos, i_pos, u_neg, j_neg):
        pred_i = torch.sum(torch.mul(u_pos, i_pos), dim=-1)
        pred_j = torch.sum(torch.mul(u_neg, j_neg), dim=-1)
        return pred_i, pred_j

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)

    def gradient_penalty(self, D, xr, xf):

        LAMBDA = 0.3

        xf = xf.detach()
        xr = xr.detach()

        alpha = torch.rand(self.batch_size * 2, 1).cuda()
        alpha = alpha.expand_as(xr)

        interpolates = alpha * xr + ((1 - alpha) * xf)
        interpolates.requires_grad_()

        disc_interpolates = D(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones_like(disc_interpolates),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

        return gp

    def weighted_sum(self, anchor, nei, co):

        ac = torch.multiply(anchor, co).sum(-1).sum(-1)
        nc = torch.multiply(nei, co).sum(-1).sum(-1)

        an = (anchor.permute(1, 0, 2)[0])
        ne = (nei.permute(1, 0, 2)[0])

        an_w = an * (ac.unsqueeze(-1).repeat(1, self.embed_size))
        ne_w = ne * (nc.unsqueeze(-1).repeat(1, self.embed_size))

        res = (self.anchor_rate * an_w + (1 - self.anchor_rate) * ne_w).reshape(-1, self.sample_num_ii,
                                                                                self.embed_size).sum(1)

        return res

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def batched_contrastive_loss(self, z1, z2, batch_size=1024):

        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)  #

        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            tmp_i = indices[i * batch_size:(i + 1) * batch_size]

            tmp_refl_sim_list = []
            tmp_between_sim_list = []
            for j in range(num_batches):
                tmp_j = indices[j * batch_size:(j + 1) * batch_size]
                tmp_refl_sim = f(self.sim(z1[tmp_i], z1[tmp_j]))
                tmp_between_sim = f(self.sim(z1[tmp_i], z2[tmp_j]))

                tmp_refl_sim_list.append(tmp_refl_sim)
                tmp_between_sim_list.append(tmp_between_sim)

            refl_sim = torch.cat(tmp_refl_sim_list, dim=-1)
            between_sim = torch.cat(tmp_between_sim_list, dim=-1)

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag() / (
                        refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:,
                                                               i * batch_size:(i + 1) * batch_size].diag()) + 1e-8))

            del refl_sim, between_sim, tmp_refl_sim_list, tmp_between_sim_list

        loss_vec = torch.cat(losses)
        return loss_vec.mean()

    def feat_reg_loss_calculation(self, g_mm_item, g_mm_user):
        feat_reg = 0
        for i in range(self.n_modalities):
            feat_reg += 1. / 2 * (g_mm_item[i] ** 2).sum() + 1. / 2 * (g_mm_user[i] ** 2).sum()
        feat_reg = feat_reg / self.n_items
        feat_emb_loss = self.feat_reg_decay * feat_reg
        return feat_emb_loss

    def u_sim_calculation(self, users, user_final, item_final):
        topk_u = user_final[users]
        u_ui = torch.tensor(self.ui_graph_raw[users].todense()).cuda()

        num_batches = (self.n_items - 1) // self.batch_size + 1
        indices = torch.arange(0, self.n_items).cuda()
        u_sim_list = []

        for i_b in range(num_batches):
            index = indices[i_b * self.batch_size:(i_b + 1) * self.batch_size]
            sim = torch.mm(topk_u, item_final[index].T)
            sim_gt = torch.multiply(sim, (1 - u_ui[:, index]))
            u_sim_list.append(sim_gt)

        u_sim = F.normalize(torch.cat(u_sim_list, dim=-1), p=2, dim=1)
        return u_sim

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1./2*(users**2).sum() + 1./2*(pos_items**2).sum() + 1./2*(neg_items**2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def bpr_loss_counterfactual(self, users, pos_items, neg_items, pos_item_data, neg_item_data):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        pos_item_scores, pos_item_scores_sigmoid = self.IRS_model(pos_item_data)
        neg_item_scores, neg_item_scores_sigmoid = self.IRS_model(neg_item_data)

        pos_scores = pos_scores * pos_item_scores_sigmoid.squeeze()
        neg_scores = neg_scores * neg_item_scores_sigmoid.squeeze()

        regularizer = 1./2*(users**2).sum() + 1./2*(pos_items**2).sum() + 1./2*(neg_items**2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss_ori = -torch.mean(maxi)

        maxii = F.logsigmoid(pos_item_scores - neg_item_scores)
        mf_loss_item = -torch.mean(maxii)

        mf_loss = mf_loss_ori + self.alpha * mf_loss_item

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def train(self):
        (line_diff_loss, line_train_rec_loss, line_eval_rec_loss, line_var_loss,
         line_g_loss, line_cl_loss, line_var_recall, line_var_precision, line_var_ndcg) = (
            [], [], [], [], [], [], [], [], [])
        n_batch = self.MRS_data.n_train // self.batch_size + 1
        self.best_recall20 = 0

        for epoch in range(self.epoch):
            loss, diff_loss, rec_loss, mf_loss, emb_loss, reg_loss, feat_loss = 0., 0., 0., 0., 0., 0., 0.
            contrastive_loss = 0.
            self.topk_p_dict, self.topk_id_dict = {}, {}

            for idx in range(n_batch):
                self.diff_model.train()
                self.MRS_model.train()
                self.IRS_model.train()
                users, pos_items, neg_items = self.MRS_data.sample()

                b_diff_loss, b_rec_loss = (self.diff_model(data=self.complete_mm_data, indicator=self.indicator,
                                    print_progress=False, min_data=self.min_data, max_data=self.max_data))
                line_diff_loss.append(b_diff_loss.data)
                line_train_rec_loss.append(b_rec_loss.data)

                G_ua_embeddings, G_ia_embeddings, G_mm_item_embeds, G_mm_user_embeds, \
                    G_user_emb, _, G_mm_user_id, G_mm_item_id, \
                    = self.MRS_model(self.ui_graph, self.iu_graph, self.mm_ui_graph, self.mm_iu_graph, self.complete_mm_data)

                G_u_g_embeddings = G_ua_embeddings[users]
                G_pos_i_g_embeddings = G_ia_embeddings[pos_items]
                G_neg_i_g_embeddings = G_ia_embeddings[neg_items]
                mm_pos_data = [self.complete_mm_data[i][pos_items] for i in range(self.n_modalities)]
                mm_neg_data = [self.complete_mm_data[i][neg_items] for i in range(self.n_modalities)]
                G_batch_mf_loss, G_batch_emb_loss, G_batch_reg_loss = self.bpr_loss_counterfactual(
                    G_u_g_embeddings, G_pos_i_g_embeddings, G_neg_i_g_embeddings, mm_pos_data, mm_neg_data
                )

                G_mm_u_sim_detach = []
                for i in range(self.n_modalities):
                    G_mm_u_sim = self.u_sim_calculation(users, G_mm_user_embeds[i], G_mm_item_embeds[i])
                    G_mm_u_sim_detach.append(G_mm_u_sim.detach())

                if idx%self.T==0 and idx!=0:
                    self.mm_ui_graph_tmp = []
                    self.mm_iu_graph_tmp = []
                    for i in range(self.n_modalities):
                        self.mm_ui_graph_tmp.append(
                            csr_matrix((torch.ones(len(self.mm_ui_index[i]['x'])),(self.mm_ui_index[i]['x'], self.mm_ui_index[i]['y'])), shape=(self.n_users, self.n_items))
                        )
                        self.mm_iu_graph_tmp.append(
                            self.mm_ui_graph_tmp[i].T
                        )
                        self.mm_ui_graph[i] = self.sparse_mx_to_torch_sparse_tensor(
                            self.csr_norm(self.mm_ui_graph_tmp[i], mean_flag=True)
                        ).cuda()
                        self.mm_iu_graph[i] = self.sparse_mx_to_torch_sparse_tensor(
                            self.csr_norm(self.mm_iu_graph_tmp[i], mean_flag=True)
                        ).cuda()

                        self.mm_ui_index[i] = {'x':[], 'y':[]}

                else:
                    for i in range(self.n_modalities):
                        _, mm_ui_id = torch.topk(G_mm_u_sim_detach[i],
                                                 int(self.n_items*self.m_topk_rate), dim=-1)
                        self.mm_ui_index[i]['x'] += np.array(torch.tensor(users)
                                                             .repeat(1, int(self.n_items*self.m_topk_rate)).view(-1)).tolist()
                        self.mm_ui_index[i]['y'] += np.array(mm_ui_id.cpu().view(-1)).tolist()

                feat_emb_loss = self.feat_reg_loss_calculation(G_mm_item_embeds, G_mm_user_embeds)

                batch_contrastive_loss = 0
                for i in range(self.n_modalities):
                    batch_contrastive_l = self.batched_contrastive_loss(G_mm_user_id[i], G_user_emb)
                    batch_contrastive_loss += batch_contrastive_l

                batch_loss = (b_diff_loss + self.lambda_rec * b_rec_loss + G_batch_mf_loss + G_batch_emb_loss +
                              G_batch_reg_loss + feat_emb_loss + self.cl_rate * batch_contrastive_loss)  # feat_emb_loss

                line_var_loss.append(batch_loss.detach().data)
                line_cl_loss.append(batch_contrastive_loss.detach().data)

                self.optimizer.zero_grad()
                self.diff_optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                self.diff_optimizer.step()

                loss += float(batch_loss)
                diff_loss += float(b_diff_loss)
                rec_loss += float(b_rec_loss)
                mf_loss += float(G_batch_mf_loss)
                emb_loss += float(G_batch_emb_loss)
                reg_loss += float(G_batch_reg_loss)
                feat_loss += float(feat_emb_loss)
                contrastive_loss += float(batch_contrastive_loss)

            del G_ua_embeddings, G_ia_embeddings, G_u_g_embeddings, G_neg_i_g_embeddings, G_pos_i_g_embeddings

            if math.isnan(loss) == True:
                print('ERROR: loss is nan.')
                return

            perf_str = 'Epoch %d: train==[%.5f = %.5f + %.5f + %.5f + %.5f + %.5f + %.5f + %.5f]' % (
                epoch+1, loss, diff_loss, rec_loss, mf_loss, emb_loss, reg_loss, feat_loss, contrastive_loss)
            print(perf_str)
            self.complete_mm_data, rec_train_loss, rec_eval_loss = self.gen_train(next_data=self.complete_mm_data)
            line_eval_rec_loss.append(rec_eval_loss)

            if (epoch + 1) % self.verbose == 0:
                users_to_test = list(self.MRS_data.test_set.keys())

                test_ret = self.test(users_to_test, is_val=False)

                print("Test_@%s: Recall=[%.5f, %.5f, %.5f, %.5f],  precision=[%.5f, %.5f, %.5f, %.5f], hit=[%.5f, %.5f, %.5f, %.5f], "
                      "ndcg=[%.5f, %.5f, %.5f, %.5f], Train Rec Loss=[%.5f], Eval Rec Loss=[%.5f], c=[%.5f], "
                      "F=[%.5f, %.5f, %.5f, %.5f], F_fuse=[%.5f, %.5f, %.5f, %.5f]" % \
                            (str(self.Ks), test_ret['recall'][0], test_ret['recall'][1], test_ret['recall'][2], test_ret['recall'][-1],
                    test_ret['precision'][0], test_ret['precision'][1], test_ret['precision'][2], test_ret['precision'][-1], test_ret['hit_ratio'][0],
                            test_ret['hit_ratio'][1], test_ret['hit_ratio'][2], test_ret['hit_ratio'][-1], test_ret['ndcg'][0], test_ret['ndcg'][1],
                            test_ret['ndcg'][2], test_ret['ndcg'][-1], rec_train_loss, rec_eval_loss, test_ret['c'], test_ret['fair_p'][0],
                             test_ret['fair_p'][1], test_ret['fair_p'][2], test_ret['fair_p'][-1], test_ret['f_fair'][0],
                             test_ret['f_fair'][1], test_ret['f_fair'][2], test_ret['f_fair'][-1]))

                if test_ret['recall'][2] > self.best_recall20:
                    self.best_recall20 = test_ret['recall'][2]
            else:
                print("Train Rec Loss=[%.5f], Eval Rec Loss=[%.5f]" % (rec_train_loss, rec_eval_loss))


    def test(self, users_to_test, is_val):
        self.MRS_model.eval()
        self.IRS_model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings, *rest = self.MRS_model(self.ui_graph, self.iu_graph, self.mm_ui_graph,
                                                             self.mm_iu_graph, self.complete_mm_data)
            item_scores, item_scores_sigmoid = self.IRS_model(self.complete_mm_data)
        self.incomplete_items_counts = {item: 0 for item in self.missing_items}
        if self.c_list is None:
            result_eval = self.tester.test_torch_counterfactual(ua_embeddings, ia_embeddings, item_scores_sigmoid.squeeze(),
                                                           users_to_test, is_val, c=self.counterfactual_coeff,
                                                           incomplete_items=self.incomplete_items_counts, export=False)
        else:
            result_eval = self.tester.test_torch_counterfactual(ua_embeddings, ia_embeddings, item_scores_sigmoid.squeeze(),
                                                           users_to_test, is_val, c_list=self.c_list,
                                                           incomplete_items=self.incomplete_items_counts, export=False)
        return result_eval

    def gen_train(self, next_data):
        self.diff_model.train()
        self.diff_optimizer.zero_grad()
        next_data, rec_train_loss, rec_eval_loss = self.diff_model.complete_train(data=next_data,
                                                                             indicator=self.indicator,
                                                                             target_data=self.ori_mm_data,
                                                                             min_data=self.min_data,
                                                                             max_data=self.max_data)
        rec_train_loss.backward()
        self.diff_optimizer.step()

        return next_data, rec_train_loss.item(), rec_eval_loss.item()

    def evaluate(self, next_data):
        self.diff_model.eval()
        with torch.no_grad():
            complete_data, eval_loss = self.diff_model.complete(data=next_data,
                                                       indicator=self.indicator,
                                                       target_data=self.ori_mm_data,
                                                       min_data=self.min_data,
                                                       max_data=self.max_data)
        print("Dataset completed. Eval Loss: {:.4f}".format(eval_loss))
        save_file = args.load_dir + "completed_data.pkl"
        with open(save_file, 'wb') as f:
            pickle.dump(complete_data, f)
        return complete_data, eval_loss

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = args.dataset
    MR = args.MR
    complete = args.complete
    seed = args.seed
    isNormalize = args.normalize
    isReduceDim = args.reduce_dim
    dim = args.dim
    ori_mm_data, incomplete_mm_data, modalities, n_sample, input_size, missing_items, indicator, svd, min_data, max_data = (
        load_multimodal_dataset(dataset=dataset, MR=MR, complete=complete, seed=seed, device=device,
                                normalize=isNormalize, reduced=isReduceDim, dim=dim))

    seed_everything(seed)
    embed_channel = args.embed_channel
    embed_size = args.embed_size
    conv1d_kernel_size = args.conv1d_kernel_size
    num_sample_steps = args.num_sample_steps
    lambda_rec = args.alpha_1
    sampling_steps = args.T_s
    batch_size = args.batch_size
    Unet_channels = eval(args.unet_channels)

    diff_model = MSDiffusion(modalities=modalities, input_channel=1, input_size=input_size, embed_channel=embed_channel,
                        embed_size=embed_size, conv1d_kernel_size=conv1d_kernel_size, num_sample_steps=num_sample_steps,
                        sampling_steps=sampling_steps, Unet_channels=Unet_channels)

    if isLoad:
        checkpoint_dir = args.load_dir
        checkpoint_model = args.load_model
        checkpoint = torch.load(checkpoint_dir + checkpoint_model, map_location=device)
        diff_model.load_state_dict(checkpoint)
        print("Pretrained diffusion model is loaded from: " + checkpoint_dir + checkpoint_model)

    MRS_data = MRS_Data(path='./data/' + args.dataset, batch_size=batch_size)

    trainer = Trainer(ori_mm_data=ori_mm_data, incomplete_mm_data=incomplete_mm_data, missing_items=missing_items,
                      indicator=indicator, input_size=input_size, modalities=modalities,
                      MRS_data=MRS_data, lr=args.lr, lr_decay=args.lr_decay,
                      batch_size=batch_size, epoch=args.epoch, diff_model=diff_model,
                      MR=args.MR, complete=args.complete, lambda_rec=lambda_rec,
                      device=device, min_data=0, max_data=1)

    trainer.train()

if __name__ == '__main__':
    main(args)