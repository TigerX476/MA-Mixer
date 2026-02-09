import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer, LSTMLayer
from layers.Embed import DataEmbedding_inverted, DataEmbedding_inverted_non_linear, DataEmbedding_inverted_Dyt
from layers.Fourier_layer import SpectralConv1d, MLLABlock, MTDecoderLayer, Mlp_conv, TemporalCasualLayer, Fourier_layer, EncoderLayer_Smamba, Encoder_Smamba
import numpy as np
from einops import rearrange
from mamba_ssm import Mamba
import random

fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
torch.cuda.manual_seed_all(fix_seed)
np.random.seed(fix_seed)

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.down_sampling_window = configs.down_sampling_window
        self.down_sampling_method = configs.down_sampling_method
        self.down_sampling_layers = configs.e_layers - 1
        self.enc_in = configs.enc_in
        self.layer = configs.e_layers

        # # core S-mamba
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        self.encoder = nn.ModuleList(
            [
                EncoderLayer_Smamba(
                    Mamba(
                        d_model=configs.d_model,  # Model dimension d_model
                        d_state=configs.d_state,  # SSM state expansion factor
                        d_conv=2,  # Local convolution width
                        expand=1,  # Block expansion factor)
                    ),
                    Mamba(
                        d_model=configs.d_model,  # Model dimension d_model
                        d_state=configs.d_state,  # SSM state expansion factor
                        d_conv=2,  # Local convolution width
                        expand=1,  # Block expansion factor)
                    ),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ]
        )

        # self.encoder = nn.ModuleList(
        #     [
        #         EncoderLayer_Smamba(
        #             LSTMLayer(
        #                 input_dim=configs.d_model,  # Model dimension d_model
        #                 hidden_dim=configs.d_ff  # SSM state expansion factor
        #             ),
        #             LSTMLayer(
        #                 input_dim=configs.d_model,  # Model dimension d_model
        #                 hidden_dim=configs.d_ff  # SSM state expansion factor
        #             ),
        #             AttentionLayer(
        #                 FullAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                               output_attention=configs.output_attention), configs.d_model,
        #                 configs.n_heads
        #             ),
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation
        #         ) for l in range(configs.e_layers)
        #     ]
        # )

        self.router = nn.ModuleList([StateTransitionConsistencyRouter(configs.d_model) for l in range(configs.e_layers)])

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        self.norm = nn.LayerNorm(configs.d_model)

        self.projector_at = nn.Sequential(nn.Linear(configs.d_model, configs.d_model * 4), nn.GELU(),nn.Linear(configs.d_model * 4, configs.d_model))
        self.projector_ma = nn.Sequential(nn.Linear(configs.d_model, configs.d_model * 4), nn.GELU(),nn.Linear(configs.d_model * 4, configs.d_model))

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        ####S-mamba
        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g timestamp) can be also embedded as tokens
        _, M, _ = enc_out.shape
        shortcut = enc_out
        at_input = enc_out
        ma_input = enc_out
        router_base = enc_out

        for i in range(self.layer):
            r = self.router[i](router_base)
            ma_input, at_input, router_base = self.encoder[i](ma_input, at_input, r)

        ma_enc_out = ma_input
        at_enc_out = at_input

        at_input = self.projector_at(at_input)
        ma_input = self.projector_ma(ma_input)

        lambda1 = (at_input - shortcut) ** 2
        lambda2 = (ma_input - shortcut) ** 2
        enc_out = self.norm(lambda1 * ma_enc_out + lambda2 * at_enc_out)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, lambda1, lambda2

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, lambda1, lambda2 = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :], lambda1, lambda2  # [B, L, D]

class StateDynamicsRouter(nn.Module):
    """
    Router based on temporal state dynamics:
    ||x_t - x_{t-1}||
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        """
        x: (B, T, D)
        return: r (B, T, 1)
        """
        # temporal difference
        delta = torch.zeros_like(x)
        delta[:, 1:] = x[:, 1:] - x[:, :-1]

        # L2 norm over feature dimension
        delta_norm = torch.norm(delta, dim=-1, keepdim=True)  # (B,T,1)

        r = torch.sigmoid(self.scale * delta_norm + self.bias)
        return r


class StateTransitionConsistencyRouter(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        # lightweight state predictor
        self.state_pred = nn.Linear(hidden_dim, hidden_dim)

        # routing scale
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        """
        x: (B, T, D)
        return: r (B, T, 1)
        """
        B, T, D = x.shape

        # predict next-state from previous state
        x_pred = torch.zeros_like(x)
        x_pred[:, 1:] = self.state_pred(x[:, :-1])

        # state transition error
        error = torch.norm(x - x_pred, dim=-1, keepdim=True)  # (B,T,1)

        # routing weight
        r = torch.sigmoid(self.scale * error + self.bias)
        return r
