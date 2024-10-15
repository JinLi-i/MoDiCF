import torch.nn as nn
import torch
import numpy as np
from torch import sqrt
from torch import nn, einsum
from .transformers_encoder.transformer import TransformerEncoder
from torch.special import expm1
from torch.cuda.amp import autocast
from functools import partial, wraps
import math
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

def exists(val):
    return val is not None

def identity(t):
    return t

def is_lambda(f):
    return callable(f) and f.__name__ == "<lambda>"

def default(val, d):
    if exists(val):
        return val
    return d() if is_lambda(d) else d

def cast_tuple(t, l = 1):
    return ((t,) * l) if not isinstance(t, tuple) else t

def append_dims(t, dims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * dims))

def l2norm(t):
    return F.normalize(t, dim = -1)

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# diffusion helpers
def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None]

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class UNet(nn.Module):
    """U-Net architecture."""

    def __init__(self, input_channel=32, channels=[32, 64, 128, 256], embed_dim=256):
        """Initialize a time-dependent score-based network.

        Args:
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))
        # Encoding layers where the temporal resolution decreases
        self.conv1 = nn.Conv1d(input_channel, channels[0], 3, stride=1, padding=1, bias=False)
        self.attention_1 = TransformerEncoder(embed_dim=channels[0],
                                              num_heads=8,
                                              layers=2,
                                              attn_dropout=0.0,
                                              relu_dropout=0.0,
                                              res_dropout=0.0,
                                              embed_dropout=0.0,
                                              attn_mask=True)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv1d(channels[0], channels[1], 3, stride=2, padding=1, bias=False)
        self.conv2_cond = nn.Conv1d(channels[0], channels[1], 3, stride=2, padding=1, bias=False)
        self.attention_2 = TransformerEncoder(embed_dim=channels[1],
                                              num_heads=8,
                                              layers=2,
                                              attn_dropout=0.0,
                                              relu_dropout=0.0,
                                              res_dropout=0.0,
                                              embed_dropout=0.0,
                                              attn_mask=True)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv1d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.conv3_cond = nn.Conv1d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.attention_3 = TransformerEncoder(embed_dim=channels[2],
                                              num_heads=8,
                                              layers=2,
                                              attn_dropout=0.0,
                                              relu_dropout=0.0,
                                              res_dropout=0.0,
                                              embed_dropout=0.0,
                                              attn_mask=True)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv1d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        self.conv4_cond = nn.Conv1d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        self.attention_4 = TransformerEncoder(embed_dim=channels[3],
                                              num_heads=8,
                                              layers=2,
                                              attn_dropout=0.0,
                                              relu_dropout=0.0,
                                              res_dropout=0.0,
                                              embed_dropout=0.0,
                                              attn_mask=True)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the temporal resolution increases
        self.tconv4 = nn.ConvTranspose1d(channels[3], channels[2], 3, stride=2, padding=1, bias=False, output_padding=1)
        self.tconv4_cond = nn.ConvTranspose1d(channels[3], channels[2], 3, stride=2, padding=1, bias=False,
                                              output_padding=1)
        self.attention_t4 = TransformerEncoder(embed_dim=channels[2],
                                               num_heads=8,
                                               layers=2,
                                               attn_dropout=0.0,
                                               relu_dropout=0.0,
                                               res_dropout=0.0,
                                               embed_dropout=0.0,
                                               attn_mask=True)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose1d(channels[2] + channels[2], channels[1], 3, stride=2, padding=1, bias=False,
                                         output_padding=1)
        self.tconv3_cond = nn.ConvTranspose1d(channels[2], channels[1], 3, stride=2, padding=1, bias=False,
                                              output_padding=1)
        self.attention_t3 = TransformerEncoder(embed_dim=channels[1],
                                               num_heads=8,
                                               layers=2,
                                               attn_dropout=0.0,
                                               relu_dropout=0.0,
                                               res_dropout=0.0,
                                               embed_dropout=0.0,
                                               attn_mask=True)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose1d(channels[1] + channels[1], channels[0], 3, stride=2, padding=1, bias=False,
                                         output_padding=1)
        self.tconv2_cond = nn.ConvTranspose1d(channels[1], channels[0], 3, stride=2, padding=1, bias=False,
                                              output_padding=1)
        self.attention_t2 = TransformerEncoder(embed_dim=channels[0],
                                               num_heads=8,
                                               layers=2,
                                               attn_dropout=0.0,
                                               relu_dropout=0.0,
                                               res_dropout=0.0,
                                               embed_dropout=0.0,
                                               attn_mask=True)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(4, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose1d(channels[0] + channels[0], input_channel, 3, stride=1, padding=1)
        self.tconv1_cond = nn.ConvTranspose1d(channels[0], input_channel, 3, stride=1, padding=1)
        self.attention_t1 = TransformerEncoder(embed_dim=embed_dim,
                                               num_heads=8,
                                               layers=2,
                                               attn_dropout=0.0,
                                               relu_dropout=0.0,
                                               res_dropout=0.0,
                                               embed_dropout=0.0,
                                               attn_mask=True)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, t, condition=None):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t))
        # Encoding path
        h1 = self.conv1(x)
        if condition is not None:
            h1_with_cond = self.attention_1(h1.permute(2, 0, 1), condition.permute(2, 0, 1), condition.permute(2, 0, 1))
            h1 += h1_with_cond.permute(1, 2, 0)
        ## Incorporate information from t
        h1 += self.dense1(embed)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        if condition is not None:
            condition = self.conv2_cond(condition)  # align condition with h2
            h2_with_cond = self.attention_2(h2.permute(2, 0, 1), condition.permute(2, 0, 1), condition.permute(2, 0, 1))
            h2 += h2_with_cond.permute(1, 2, 0)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        if condition is not None:
            condition = self.conv3_cond(condition)  # align condition with h3
            h3_with_cond = self.attention_3(h3.permute(2, 0, 1), condition.permute(2, 0, 1), condition.permute(2, 0, 1))
            h3 += h3_with_cond.permute(1, 2, 0)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        if condition is not None:
            condition = self.conv4_cond(condition)  # align condition with h4
            h4_with_cond = self.attention_4(h4.permute(2, 0, 1), condition.permute(2, 0, 1), condition.permute(2, 0, 1))
            h4 += h4_with_cond.permute(1, 2, 0)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        if condition is not None:
            condition = self.tconv4_cond(condition)
            h_with_cond = self.attention_t4(h.permute(2, 0, 1), condition.permute(2, 0, 1), condition.permute(2, 0, 1))
            h += h_with_cond.permute(1, 2, 0)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        if condition is not None:
            condition = self.tconv3_cond(condition)
            h_with_cond = self.attention_t3(h.permute(2, 0, 1), condition.permute(2, 0, 1), condition.permute(2, 0, 1))
            h += h_with_cond.permute(1, 2, 0)
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        if condition is not None:
            condition = self.tconv2_cond(condition)
            h_with_cond = self.attention_t2(h.permute(2, 0, 1), condition.permute(2, 0, 1), condition.permute(2, 0, 1))
            h += h_with_cond.permute(1, 2, 0)
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))
        if condition is not None:
            condition = self.tconv1_cond(condition)
            h_with_cond = self.attention_t1(h.permute(2, 0, 1), condition.permute(2, 0, 1), condition.permute(2, 0, 1))
            h += h_with_cond.permute(1, 2, 0)

        return h


class DiffusionNet(nn.Module):
    def __init__(self, input_size, input_channel=16, timesteps=1000, sampling_timesteps=None, objective='pred_noise',
                 beta_schedule='sigmoid', schedule_fn_kwargs = dict(), ddim_sampling_eta = 0.,
                 offset_noise_strength = 0., min_snr_loss_weight=False,  min_snr_gamma=5,
                 channels=[32, 64, 128, 256], embed_dim=256):
        super().__init__()
        self.input_size = input_size

        self.model = UNet(input_channel=input_channel, channels=channels, embed_dim=embed_dim)

        self.objective = 'pred_noise'
        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps,
                                          timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))


    @property
    def device(self):
        return next(self.model.parameters()).device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x, t, condition=None, clip_x_start=False, rederive_pred_noise=False,
                          min_data=None, max_data=None):
        if condition is not None:
            noise = torch.randn_like(condition)
            perturbed_condition = (
                    extract(self.sqrt_alphas_cumprod, t, condition.shape) * condition +
                    extract(self.sqrt_one_minus_alphas_cumprod, t, condition.shape) * noise
            )
        else:
            perturbed_condition = None
        model_output = self.model(x, t, perturbed_condition)

        if clip_x_start and min_data is not None and max_data is not None:
            maybe_clip = partial(torch.clamp, min = min_data, max = max_data)
        else:
            maybe_clip = identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        else:
            assert "Unknown objective"

        return pred_noise, x_start

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, condition=None, clip_denoised=False, min_data=None, max_data=None):
        pred_noise, pred_x_start = self.model_predictions(x, t, condition=condition, clip_x_start=clip_denoised,
                                                          min_data=min_data, max_data=max_data)
        x_start = pred_x_start

        if clip_denoised and min_data is not None and max_data is not None:
            x_start.clamp_(min_data, max_data)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    # sampling related functions

    @torch.no_grad()
    def p_sample(self, x, t: int, condition=None, clip_denoised=False, min_data=None, max_data=None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times,
                                                                          condition=condition,
                                                                          clip_denoised=clip_denoised,
                                                                          min_data=min_data, max_data=max_data)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, condition=None, print_progress=False,
                      clip_denoised=False, min_data=None, max_data=None):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device=device)

        x_start = None

        if print_progress:
            for_iter = tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps)
        else:
            for_iter = range(self.num_timesteps)

        for t in for_iter:
            img, x_start = self.p_sample(img, t, condition=condition, clip_denoised=clip_denoised, min_data=min_data, max_data=max_data)

        ret = img

        # ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def ddim_sample(self, shape, condition=None, clip_denoised=False, print_progress=False,
                    min_data=None, max_data=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = (
            shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective)

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device=device)

        x_start = None

        if print_progress:
            iter = tqdm(time_pairs, desc='sampling loop time step')
        else:
            iter = time_pairs

        if min_data is not None and max_data is not None:
            clip_x_start = True
        else:
            clip_x_start = False

        for time, time_next in iter:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(x, time_cond, condition, clip_x_start=clip_x_start,
                                                         rederive_pred_noise=True, min_data=min_data, max_data=max_data)

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        ret = x

        # ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, shape, condition=None, clip_denoised=False, print_progress=False, min_data=None, max_data=None):
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(shape, condition=condition, clip_denoised=clip_denoised,
                         print_progress=print_progress, min_data=min_data, max_data=max_data)
        # return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size))

    # training related functions - noise prediction

    @autocast(enabled=False)
    def q_sample(self, x_start, t, noise=None, condition=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noised = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        if condition is not None:
            perturbed_condition = (
                    extract(self.sqrt_alphas_cumprod, t, condition.shape) * condition +
                    extract(self.sqrt_one_minus_alphas_cumprod, t, condition.shape) * noise
            )
        else:
            perturbed_condition = None

        return x_noised, perturbed_condition

    def p_losses(self, x_start, t, noise=None, condition=None):
        # reverse process
        # torch.manual_seed(0)
        noise = default(noise, lambda: torch.randn_like(x_start))
        # noise = torch.randn_like(x_start)
        x, condition_noised = self.q_sample(x_start=x_start, t=t, noise=noise, condition=condition)
        # model_out = self.model(x, log_snr)
        model_out = self.model(x, t, condition=condition_noised)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction='none')

        loss = reduce(loss, 'b ... -> b', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)

        return loss.mean()

    def forward(self, x, condition=None):
        b, c, d, device, input_size, = *x.shape, x.device, self.input_size
        assert d == input_size, f'height and width of image must be {input_size}'

        # x = normalize_to_neg_one_to_one(x)
        # times = torch.zeros((x.shape[0],), device=self.device).float().uniform_(0, 1)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(x, t, condition=condition)