import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .diffusion import *

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

class MSDiffusion(nn.Module):
    def __init__(self, modalities, input_channel, input_size, embed_channel, embed_size, conv1d_kernel_size,
                 num_sample_steps=1000, sampling_steps=50, Unet_channels=[32, 64, 128, 256]):
        super(MSDiffusion, self).__init__()
        self.modalities = modalities
        self.n_modalities = len(modalities)
        self.embed_channel = embed_channel
        self.embed_size = embed_size
        self.conv1d_kernel_size = conv1d_kernel_size
        self.input_channel = input_channel
        self.input_size = input_size
        self.num_sample_steps = num_sample_steps
        self.sampling_steps = sampling_steps

        self.models = nn.ModuleDict()

        for i in range(self.n_modalities):
            branch_name = modalities[i] + '_'
            mid = int(self.input_size[i]/2)
            if self.embed_size < mid:
                mid = self.embed_size
            self.models[branch_name + 'enc'] = nn.Sequential(
                nn.Linear(self.input_size[i], mid),
                nn.Linear(mid, self.embed_size),
                nn.ReLU(),
                Reshape((-1, 1, self.embed_size)),
                nn.Conv1d(1, self.embed_channel, kernel_size=1, padding=0),
            )
            self.models[branch_name + 'dec'] = nn.Sequential(
                nn.Conv1d(self.embed_channel, 1, kernel_size=1, padding=0),
                nn.ReLU(),
                Reshape((-1, self.embed_size)),
                nn.Linear(self.embed_size, mid),
                nn.Linear(mid, self.input_size[i])
            )

            self.models[branch_name + 'diffusion'] = DiffusionNet(input_size=embed_size, input_channel=embed_channel,
                                                                  timesteps=num_sample_steps,
                                                                sampling_timesteps=sampling_steps, objective='pred_noise',
                                                                beta_schedule='sigmoid', ddim_sampling_eta=0,
                                                                embed_dim=self.embed_channel, channels=Unet_channels)

        for i in range(self.n_modalities):
            self.models['condition_' + str(i)] = nn.Conv1d(self.embed_channel * (self.n_modalities-1),
                                                           self.embed_channel, 1, padding=0)


    def forward(self, data, indicator, print_progress=False, min_data=None, max_data=None):
        dtype = data[0].dtype
        device = data[0].device

        embeds = []
        for m in range(self.n_modalities):
            x = data[m]
            embed = self.models[self.modalities[m] + '_enc'](x)
            embeds.append(embed)

        diff_loss = torch.tensor(0.0, dtype=dtype, device=device)
        rec_loss = torch.tensor(0.0, dtype=dtype, device=device)
        rec_diff = [None for _ in range(self.n_modalities)]
        rec_dec = [None for _ in range(self.n_modalities)]
        for m in range(self.n_modalities):
            m_complete_id = np.where(indicator[:, m] == 1)[0]
            if len(m_complete_id) == 0:
                continue

            other_modalities = [embeds[j][m_complete_id, :, :] for j in range(self.n_modalities) if j != m]
            other_modalities = torch.cat(other_modalities, dim=1)
            m_condition = self.models['condition_' + str(m)](other_modalities)

            m_embeds = embeds[m][m_complete_id, :, :]
            m_diff_loss = self.models[self.modalities[m] + '_diffusion'](m_embeds, condition=m_condition)
            diff_loss += m_diff_loss

            rec_dec[m] = self.models[self.modalities[m] + '_dec'](m_embeds)
            m_rec_loss = F.mse_loss(rec_dec[m], data[m][m_complete_id, :])
            rec_loss += m_rec_loss

        return diff_loss, rec_loss

    def complete(self, data, indicator, target_data, min_data=None, max_data=None):
        dtype = data[0].dtype
        device = data[0].device
        batch_ind_sum = np.sum(indicator, axis=1)
        incomplete_id = np.where(batch_ind_sum < self.n_modalities)[0]

        if len(incomplete_id) == 0:
            return data, torch.tensor(0., dtype=dtype, device=device)

        embeds = []
        for m in range(self.n_modalities):
            x = data[m]
            embed = self.models[self.modalities[m] + '_enc'](x)
            embeds.append(embed)

        rec_diff = [None for _ in range(self.n_modalities)]
        rec_dec = [None for _ in range(self.n_modalities)]
        rec_data = [data[m].detach().cpu().numpy() for m in range(self.n_modalities)]
        rec_loss = torch.tensor(0.0, dtype=dtype, device=device)
        for m in range(self.n_modalities):
            m_incomplete_id = np.where(indicator[:, m] == 0)[0]
            if len(m_incomplete_id) == 0:
                continue
            other_modalities = [embeds[j][m_incomplete_id, :, :] for j in range(self.n_modalities) if j != m]
            other_modalities = torch.cat(other_modalities, dim=1)
            m_condition = self.models['condition_' + str(m)](other_modalities)
            m_embeds = embeds[m][m_incomplete_id, :, :]

            rec_diff[m] = self.models[self.modalities[m] + '_diffusion'].sample(shape=m_embeds.shape,
                                                                               condition=m_condition,
                                                                               print_progress=False,
                                                                               min_data=min_data[m],
                                                                               max_data=max_data[m])
            rec_dec[m] = self.models[self.modalities[m] + '_dec'](rec_diff[m])
            rec_data[m][m_incomplete_id, :] = rec_dec[m].detach().cpu().numpy()

            loss = F.mse_loss(rec_dec[m], target_data[m][m_incomplete_id, :])
            rec_loss += loss

        return rec_data, rec_loss.item()


    def complete_train(self, data, indicator, target_data, min_data=None, max_data=None):
        dtype = data[0].dtype
        device = data[0].device
        batch_ind_sum = np.sum(indicator, axis=1)

        embeds = []
        for m in range(self.n_modalities):
            x = data[m]
            embed = self.models[self.modalities[m] + '_enc'](x)
            embeds.append(embed)

        # complete the whole dataset
        rec_diff = [None for _ in range(self.n_modalities)]
        rec_dec = [None for _ in range(self.n_modalities)]
        rec_data = [None for _ in range(self.n_modalities)]
        out_data = [data[m].clone().detach() for m in range(self.n_modalities)]
        rec_train_loss = torch.tensor(0.0, dtype=dtype, device=device)
        rec_eval_loss = torch.tensor(0.0, dtype=dtype, device=device)
        for m in range(self.n_modalities):
            other_modalities = [embeds[j] for j in range(self.n_modalities) if j != m]
            other_modalities = torch.cat(other_modalities, dim=1)
            m_condition = self.models['condition_' + str(m)](other_modalities)
            m_embeds = embeds[m]

            rec_diff[m] = self.models[self.modalities[m] + '_diffusion'].sample(shape=m_embeds.shape,
                                                                                condition=m_condition,
                                                                                print_progress=False,
                                                                                min_data=min_data[m],
                                                                                max_data=max_data[m])
            rec_dec[m] = self.models[self.modalities[m] + '_dec'](rec_diff[m])
            rec_data[m] = rec_dec[m]

            m_complete_id = np.where(indicator[:, m] == 1)[0]
            if len(m_complete_id) != 0:
                m_rec_train_loss = F.mse_loss(rec_dec[m][m_complete_id, :], target_data[m][m_complete_id, :])
                rec_train_loss += m_rec_train_loss

            with torch.no_grad():
                m_incomplete_id = np.where(indicator[:, m] == 0)[0]
                if len(m_incomplete_id) != 0:
                    m_rec_eval_loss = F.mse_loss(rec_dec[m][m_incomplete_id, :], target_data[m][m_incomplete_id, :])
                    rec_eval_loss += m_rec_eval_loss
                    out_data[m][m_incomplete_id, :] = rec_dec[m][m_incomplete_id, :].clone().detach()

        return out_data, rec_train_loss, rec_eval_loss