import torch.nn as nn
import torch
from layers.Transformer_EncDec import Encoder, EncoderLayer
from torch.nn.utils import weight_norm
import math
import random
import torch.nn.functional as F
import numpy as np

fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
torch.cuda.manual_seed_all(fix_seed)
np.random.seed(fix_seed)

class SpectralConv1d(nn.Module):
	def __init__(self, in_channels, out_channels, modes1):
		super(SpectralConv1d, self).__init__()

		"""
		1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
		"""

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

		self.scale = (1 / (in_channels * out_channels))
		self.weights1 = nn.Parameter(
			self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, dtype=torch.cfloat))

	# Complex multiplication
	def compl_mul1d(self, input, weights):
		# (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
		return torch.einsum("bix,iox->box", input, weights)

	def forward(self, x):
		batchsize = x.shape[0]
		# Compute Fourier coeffcients up to factor of e^(- something constant)
		x_ft = torch.fft.rfft(x)

		# Multiply relevant Fourier modes
		out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
		out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

		# Return to physical space
		x = torch.fft.irfft(out_ft, n=x.size(-1))
		return x

class RoPE(nn.Module):
	def __init__(self, shape, base=10000):
		super(RoPE, self).__init__()

		channel_dims, feature_dim = shape[:-1], shape[-1]
		k_max = feature_dim // (2 * len(channel_dims))

		assert feature_dim % k_max == 0

		theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
		angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in
							torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

		rotations_re = torch.cos(angles).unsqueeze(dim=-1)
		rotations_im = torch.sin(angles).unsqueeze(dim=-1)
		rotations = torch.cat([rotations_re, rotations_im], dim=-1)
		self.register_buffer('rotations', rotations)

	def forward(self, x):
		if x.dtype != torch.float32:
			x = x.to(torch.float32)
		x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
		pe_x = torch.view_as_complex(self.rotations) * x
		return torch.view_as_real(pe_x).flatten(-2)

class LinearAttention(nn.Module):
	def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):
		super().__init__()
		self.dim = dim
		self.input_resolution = input_resolution
		self.num_heads = num_heads
		self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
		self.elu = nn.ELU()
		self.lepe = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
		self.rope = RoPE(shape=(input_resolution, dim))

	def forward(self, x):
		b, n, c = x.shape
		num_heads = self.num_heads
		head_dim = c // num_heads

		qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
		q, k = qk[0], qk[1]

		q = self.elu(q) + 1.0
		k = self.elu(k) + 1.0
		q_rope = self.rope(q.reshape(b, n, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
		k_rope = self.rope(k.reshape(b, n, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
		q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
		k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
		v = x.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

		z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
		kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
		x = q_rope @ kv * z

		x = x.transpose(1, 2).reshape(b, n, c)
		v = v.transpose(1, 2).reshape(b, n, c).permute(0, 2, 1)
		x = x + self.lepe(v).permute(0, 2, 1).reshape(b, n, c)

		return x

class Mlp(nn.Module):
	def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features
		self.fc1 = nn.Linear(in_features, hidden_features)
		self.act = act_layer()
		self.fc2 = nn.Linear(hidden_features, out_features)
		self.drop = nn.Dropout(drop)

	def forward(self, x):
		x = self.fc1(x)
		x = self.act(x)
		x = self.drop(x)
		x = self.fc2(x)
		x = self.drop(x)
		return x

class Mlp_conv(nn.Module):
	def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features
		self.fc1 = nn.Conv1d(in_features, hidden_features, kernel_size=1)
		self.act = act_layer()
		self.fc2 = nn.Conv1d(hidden_features, out_features, kernel_size=1)


	def forward(self, x):
		x = self.fc1(x)
		x = self.act(x)
		x = self.fc2(x)
		return x


class MLLABlock(nn.Module):
	def __init__(self, dim, input_resolution, num_heads=1, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
				 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
		super().__init__()
		self.dim = dim
		self.input_resolution = input_resolution
		self.num_heads = num_heads
		self.mlp_ratio = mlp_ratio

		self.cpe1 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
		self.norm1 = norm_layer(dim)
		self.in_proj = nn.Linear(dim, dim)
		self.act_proj = nn.Linear(dim, dim)
		self.dwc = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
		self.act = nn.SiLU()
		self.attn = LinearAttention(dim=dim, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=qkv_bias)
		self.out_proj = nn.Linear(dim, dim)
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

		self.cpe2 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
		self.norm2 = norm_layer(dim)
		self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

	def forward(self, x):
		B, L, C = x.shape
		H = self.input_resolution
		assert L == H, "input feature has wrong size"

		x = x + self.cpe1(x.permute(0, 2, 1)).permute(0, 2, 1)
		shortcut = x

		x = self.norm1(x)
		act_res = self.act(self.act_proj(x))
		x = self.in_proj(x).permute(0, 2, 1)
		x = self.act(self.dwc(x)).permute(0, 2, 1)

		x = self.attn(x)

		x = self.out_proj(x * act_res)
		x = shortcut + self.drop_path(x)
		x = x + self.cpe2(x.permute(0, 2, 1)).permute(0, 2, 1)

		x = x + self.drop_path(self.mlp(self.norm2(x)))
		return x

class LinearAttention_Dec(nn.Module):
	def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):
		super().__init__()
		self.dim = dim
		self.input_resolution = input_resolution
		self.num_heads = num_heads
		# self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
		self.q = nn.Linear(dim, dim, bias=qkv_bias)
		self.k = nn.Linear(dim, dim, bias=qkv_bias)
		self.elu = nn.ELU()
		self.lepe = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
		self.rope = RoPE(shape=(input_resolution, dim))

	def forward(self, x,y):
		b, n, c = x.shape
		num_heads = self.num_heads
		head_dim = c // num_heads

		# qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
		q = self.q(x)
		k = self.k(y)

		q = self.elu(q) + 1.0
		k = self.elu(k) + 1.0
		q_rope = self.rope(q.reshape(b, n, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
		k_rope = self.rope(k.reshape(b, n, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
		q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
		k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
		v = y.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

		z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
		kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
		x = q_rope @ kv * z

		x = x.transpose(1, 2).reshape(b, n, c)
		v = v.transpose(1, 2).reshape(b, n, c).permute(0, 2, 1)
		x = x + self.lepe(v).permute(0, 2, 1).reshape(b, n, c)

		return x

class MTDecoderLayer(nn.Module):
	def __init__(self, d_model,input_resolution, nhead, dim_feedforward, dropout):
		super(MTDecoderLayer, self).__init__()
		self.self_attn = LinearAttention(d_model, input_resolution=input_resolution, num_heads=nhead)
		self.multihead_attn = LinearAttention_Dec(dim=d_model,input_resolution=input_resolution, num_heads=nhead)

		self.linear1 = nn.Linear(d_model, dim_feedforward)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(dim_feedforward, d_model)
		self.dropout1 = nn.Dropout(dropout)

		self.in_proj = nn.Linear(d_model, d_model)
		self.dwc = nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model)
		self.act = nn.SiLU()
		self.act_proj = nn.Linear(d_model, d_model)
		self.act_proj2 = nn.Linear(d_model, d_model)

		self.activation = nn.GELU()


	def forward(self, tgt, hiddden_states, memory=None, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
		m = hiddden_states
		shortcut = tgt
		act_res = self.act(self.act_proj(shortcut))
		act_res2 = self.act(self.act_proj2(shortcut))

		tgt = self.in_proj(tgt).permute(0, 2, 1)
		tgt = self.act(self.dwc(tgt)).permute(0, 2, 1)

		tmp2 = self.self_attn(tgt)
		tgt = tmp2 * act_res + tgt

		tmp4 = self.multihead_attn(tgt, m)
		tgt = tmp4 * act_res2 + tgt

		tmp6 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
		tgt = tgt + self.dropout1(tmp6)

		return tgt


class Crop(nn.Module):

	def __init__(self, crop_size):
		super(Crop, self).__init__()
		self.crop_size = crop_size

	def forward(self, x):
		_, _, N = x.shape
		# 裁剪张量以去除额外的填充
		return x[:, :, :N-self.crop_size].contiguous()


# 实现了一个膨胀卷积层，由两个膨胀卷积块组成。每个膨胀卷积块包含一个带有权重归一化的卷积层、裁剪模块、ReLU激活函数和 Dropout 正则化。此外，还包括了一个用于快捷连接的卷积层
class TemporalCasualLayer(nn.Module):

	def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.1):
		super(TemporalCasualLayer, self).__init__()
		padding = (kernel_size - 1) * dilation
		conv_params = {
			'kernel_size': kernel_size,
			'stride': stride,
			'padding': padding,
			'dilation': dilation
		}

		self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, **conv_params))
		self.crop1 = Crop(padding)

		self.net = nn.Sequential(self.conv1, self.crop1)

	def forward(self, x):
		# 应用因果卷积和快捷连接
		y = self.net(x)
		return y

# class Mamba(nn.Module):
# 	def __init__(
# 		self,
# 		d_model,
# 		d_state=16,
# 		d_conv=4,
# 		expand=2,
# 		dt_rank="auto",
# 		dt_min=0.001,
# 		dt_max=0.1,
# 		dt_init="random",
# 		dt_scale=1.0,
# 		dt_init_floor=1e-4,
# 		conv_bias=True,
# 		bias=False,
# 		use_fast_path=True,  # Fused kernel options
# 		layer_idx=None,
# 		device=None,
# 		dtype=None,
# 	):
# 		factory_kwargs = {"device": device, "dtype": dtype}
# 		super().__init__()
# 		self.d_model = d_model
# 		self.d_state = d_state
# 		self.d_conv = d_conv
# 		self.expand = expand
# 		self.d_inner = int(self.expand * self.d_model)
# 		self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
# 		self.use_fast_path = use_fast_path
# 		self.layer_idx = layer_idx
#
# 		self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
#
# 		self.conv1d = nn.Conv1d(
# 			in_channels=self.d_inner,
# 			out_channels=self.d_inner,
# 			bias=conv_bias,
# 			kernel_size=d_conv,
# 			groups=self.d_inner,
# 			padding=d_conv - 1,
# 			**factory_kwargs,
# 		)
#
# 		self.activation = "silu"
# 		self.act = nn.SiLU()
#
# 		self.x_proj = nn.Linear(
# 			self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
# 		)
# 		self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
#
# 		# Initialize special dt projection to preserve variance at initialization
# 		dt_init_std = self.dt_rank**-0.5 * dt_scale
# 		if dt_init == "constant":
# 			nn.init.constant_(self.dt_proj.weight, dt_init_std)
# 		elif dt_init == "random":
# 			nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
# 		else:
# 			raise NotImplementedError
#
# 		# Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
# 		dt = torch.exp(
# 			torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
# 			+ math.log(dt_min)
# 		).clamp(min=dt_init_floor)
# 		# Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
# 		inv_dt = dt + torch.log(-torch.expm1(-dt))
# 		with torch.no_grad():
# 			self.dt_proj.bias.copy_(inv_dt)
# 		# Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
# 		self.dt_proj.bias._no_reinit = True
#
# 		# S4D real initialization
# 		A = repeat(
# 			torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
# 			"n -> d n",
# 			d=self.d_inner,
# 		).contiguous()
# 		A_log = torch.log(A)  # Keep A_log in fp32
# 		self.A_log = nn.Parameter(A_log)
# 		self.A_log._no_weight_decay = True
#
# 		# D "skip" parameter
# 		self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
# 		self.D._no_weight_decay = True
#
# 		self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
#
# 	def forward(self, hidden_states, inference_params=None):
# 		"""
# 		hidden_states: (B, L, D)
# 		Returns: same shape as hidden_states
# 		"""
# 		batch, seqlen, dim = hidden_states.shape
#
# 		conv_state, ssm_state = None, None
# 		if inference_params is not None:
# 			conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
# 			if inference_params.seqlen_offset > 0:
# 				# The states are updated inplace
# 				out, _, _ = self.step(hidden_states, conv_state, ssm_state)
# 				return out
#
# 		# We do matmul and transpose BLH -> HBL at the same time
# 		xz = rearrange(
# 			self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
# 			"d (b l) -> b d l",
# 			l=seqlen,
# 		)
# 		if self.in_proj.bias is not None:
# 			xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
#
# 		A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
# 		# In the backward pass we write dx and dz next to each other to avoid torch.cat
# 		if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
# 			out = mamba_inner_fn(
# 				xz,
# 				self.conv1d.weight,
# 				self.conv1d.bias,
# 				self.x_proj.weight,
# 				self.dt_proj.weight,
# 				self.out_proj.weight,
# 				self.out_proj.bias,
# 				A,
# 				None,  # input-dependent B
# 				None,  # input-dependent C
# 				self.D.float(),
# 				delta_bias=self.dt_proj.bias.float(),
# 				delta_softplus=True,
# 			)
# 		else:
# 			x, z = xz.chunk(2, dim=1)
# 			# Compute short convolution
# 			if conv_state is not None:
# 				# If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
# 				# Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
# 				conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
# 			if causal_conv1d_fn is None:
# 				x = self.act(self.conv1d(x)[..., :seqlen])
# 			else:
# 				assert self.activation in ["silu", "swish"]
# 				x = causal_conv1d_fn(
# 					x=x,
# 					weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
# 					bias=self.conv1d.bias,
# 					activation=self.activation,
# 				)
#
# 			# We're careful here about the layout, to avoid extra transposes.
# 			# We want dt to have d as the slowest moving dimension
# 			# and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
# 			x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
# 			dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
# 			dt = self.dt_proj.weight @ dt.t()
# 			dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
# 			B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
# 			C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
# 			assert self.activation in ["silu", "swish"]
# 			y = selective_scan_fn(
# 				x,
# 				dt,
# 				A,
# 				B,
# 				C,
# 				self.D.float(),
# 				z=z,
# 				delta_bias=self.dt_proj.bias.float(),
# 				delta_softplus=True,
# 				return_last_state=ssm_state is not None,
# 			)
# 			if ssm_state is not None:
# 				y, last_state = y
# 				ssm_state.copy_(last_state)
# 			y = rearrange(y, "b d l -> b l d")
# 			out = self.out_proj(y)
# 		return out
#
# 	def step(self, hidden_states, conv_state, ssm_state):
# 		dtype = hidden_states.dtype
# 		assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
# 		xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
# 		x, z = xz.chunk(2, dim=-1)  # (B D)
#
# 		# Conv step
# 		if causal_conv1d_update is None:
# 			conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
# 			conv_state[:, :, -1] = x
# 			x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
# 			if self.conv1d.bias is not None:
# 				x = x + self.conv1d.bias
# 			x = self.act(x).to(dtype=dtype)
# 		else:
# 			x = causal_conv1d_update(
# 				x,
# 				conv_state,
# 				rearrange(self.conv1d.weight, "d 1 w -> d w"),
# 				self.conv1d.bias,
# 				self.activation,
# 			)
#
# 		x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
# 		dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
# 		# Don't add dt_bias here
# 		dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
# 		A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
#
# 		# SSM step
# 		if selective_state_update is None:
# 			# Discretize A and B
# 			dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
# 			dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
# 			dB = torch.einsum("bd,bn->bdn", dt, B)
# 			ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
# 			y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
# 			y = y + self.D.to(dtype) * x
# 			y = y * self.act(z)  # (B D)
# 		else:
# 			y = selective_state_update(
# 				ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
# 			)
#
# 		out = self.out_proj(y)
# 		return out.unsqueeze(1), conv_state, ssm_state
#
# 	def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
# 		device = self.out_proj.weight.device
# 		conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
# 		conv_state = torch.zeros(
# 			batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
# 		)
# 		ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
# 		# ssm_dtype = torch.float32
# 		ssm_state = torch.zeros(
# 			batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
# 		)
# 		return conv_state, ssm_state
#
# 	def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
# 		assert self.layer_idx is not None
# 		if self.layer_idx not in inference_params.key_value_memory_dict:
# 			batch_shape = (batch_size,)
# 			conv_state = torch.zeros(
# 				batch_size,
# 				self.d_model * self.expand,
# 				self.d_conv,
# 				device=self.conv1d.weight.device,
# 				dtype=self.conv1d.weight.dtype,
# 			)
# 			ssm_state = torch.zeros(
# 				batch_size,
# 				self.d_model * self.expand,
# 				self.d_state,
# 				device=self.dt_proj.weight.device,
# 				dtype=self.dt_proj.weight.dtype,
# 				# dtype=torch.float32,
# 			)
# 			inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
# 		else:
# 			conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
# 			# TODO: What if batch size changes between generation, and we reuse the same states?
# 			if initialize_states:
# 				conv_state.zero_()
# 				ssm_state.zero_()
# 		return conv_state, ssm_state


class Add_Norm(nn.Module):
	def __init__(self, d_model, dropout, residual, drop_flag=1):
		super(Add_Norm, self).__init__()
		self.dropout = nn.Dropout(dropout)
		self.norm = nn.LayerNorm(d_model)
		self.residual = residual
		self.drop_flag = drop_flag

	def forward(self, new, old):
		new = self.dropout(new) if self.drop_flag else new
		return self.norm(old + new) if self.residual else self.norm(new)


class EncoderLayer_mamba(nn.Module):
	def __init__(self, mamba_forward, mamba_backward, d_model=128, d_ff=256, dropout=0.2,
				 activation="relu", bi_dir=0, residual=1):
		super(EncoderLayer_mamba, self).__init__()
		self.bi_dir = bi_dir
		self.mamba_forward = mamba_forward
		self.residual = residual
		self.addnorm_for = Add_Norm(d_model, dropout, residual, drop_flag=0)

		if self.bi_dir:
			self.mamba_backward = mamba_backward
			self.addnorm_back = Add_Norm(d_model, dropout, residual, drop_flag=0)

		self.ffn = nn.Sequential(
			nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1),
			nn.ReLU() if activation == "relu" else nn.GELU(),
			nn.Dropout(dropout),
			nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
		)
		self.addnorm_ffn = Add_Norm(d_model, dropout, residual, drop_flag=1)

	def forward(self, x):
		# [B, S, D]
		output_forward = self.mamba_forward(x)
		output_forward = self.addnorm_for(output_forward, x)

		if self.bi_dir:
			output_backward = self.mamba_backward(x.flip(dims=[1])).flip(dims=[1])
			output_backward = self.addnorm_back(output_backward, x)
			output = output_forward + output_backward
		else:
			output = self.addnorm_for(output_forward, x)
		temp = output
		output = self.ffn(output.transpose(-1, 1)).transpose(-1, 1)
		output = self.addnorm_ffn(output, temp)
		return output


class Encoder_mamba(nn.Module):
	def __init__(self, mamba_layers, norm_layer=None):
		super(Encoder_mamba, self).__init__()
		self.mamba_layers = nn.ModuleList(mamba_layers)
		self.norm = norm_layer

	def forward(self, x):
		# [B, S, D]
		for mamba_block in self.mamba_layers:
			x = mamba_block(x)

		if self.norm is not None:
			x = self.norm(x)

		return x


class PatchEmbedding(nn.Module):
	def __init__(self, seq_len, d_model, patch_len, stride, dropout,
				 process_layer=None,
				 pos_embed_type=None, learnable=False, r_layers=1,
				 ch_ind=0):
		super(PatchEmbedding, self).__init__()
		# Patching
		self.patch_len = patch_len
		self.stride = stride
		self.ch_ind = ch_ind
		self.process_layer = process_layer
		self.pos_embed_type = pos_embed_type

		# Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
		self.value_embedding = nn.Linear(patch_len, d_model)

		# Positional embedding
		if self.pos_embed_type is not None:
			self.position_embedding = PositionalEmbedding(seq_len, d_model, pos_embed_type, learnable, r_layers,
														  patch_len)

		# Residual dropout
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, x_mark=None):
		# x: [B, M, L], x_mark: [B, 4, L]
		n_vars = x.shape[1]
		x = self.process_layer(x)
		x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

		if x_mark is not None and not self.ch_ind:
			x_mark = x_mark.unfold(dimension=-1, size=self.patch_len, step=self.stride)
			x = torch.cat([x, x_mark], dim=1)  # [B, (M+4), N, P]

		# Input value embedding
		x = self.value_embedding(x)  # [B, M, N, D]

		x = torch.reshape(x, (-1, x.shape[2], x.shape[3]))  # [B*M, N, P]

		if self.pos_embed_type is not None:
			x = x + self.position_embedding(x)

		return self.dropout(x), n_vars

class TruncateModule(nn.Module):
	def __init__(self, target_length):
		super(TruncateModule, self).__init__()
		self.target_length = target_length

	def forward(self, x, truncate_length):
		return x[: ,: ,:truncate_length]

def PositionalEncoding(q_len, d_model, normalize=False):
	pe = torch.zeros(q_len, d_model)
	position = torch.arange(0, q_len).unsqueeze(1)
	div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
	pe[:, 0::2] = torch.sin(position * div_term)
	pe[:, 1::2] = torch.cos(position * div_term)
	if normalize:
		pe = pe - pe.mean()
		pe = pe / (pe.std() * 10)
	return pe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
	cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
	if normalize:
		cpe = cpe - cpe.mean()
		cpe = cpe / (cpe.std() * 10)
	return cpe


def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
	x = .5 if exponential else 1
	i = 0
	for i in range(100):
		cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
		print(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
		if abs(cpe.mean()) <= eps: break
		elif cpe.mean() > eps: x += .001
		else: x -= .001
		i += 1
	if normalize:
		cpe = cpe - cpe.mean()
		cpe = cpe / (cpe.std() * 10)
	return cpe

class LocalRNN(nn.Module):
	def __init__(self, input_dim, output_dim, rnn_type='GRU', ksize=3):
		super(LocalRNN, self).__init__()
		"""
		LocalRNN structure
		"""
		self.ksize = ksize
		if rnn_type == 'GRU':
			self.rnn = nn.GRU(output_dim, output_dim, batch_first=True)
		elif rnn_type == 'LSTM':
			self.rnn = nn.LSTM(output_dim, output_dim, batch_first=True)
		else:
			self.rnn = nn.RNN(output_dim, output_dim, batch_first=True)

		# To speed up
		idx = [i for j in range(self.ksize-1,10000,1) for i in range(j-(self.ksize-1),j+1,1)]
		self.select_index = torch.LongTensor(idx).cuda()
		self.zeros = torch.zeros((self.ksize-1, input_dim)).cuda()

	def forward(self, x):
		nbatches, l, input_dim = x.shape
		# x: [bs x patch_num x d_model] → [b x seq_len x ksize x d_model]
		x = self.get_K(x)
		batch, l, ksize, d_model = x.shape
		h = self.rnn(x.view(-1, self.ksize, d_model))[0][:,-1,:]
		return h.view(batch, l, d_model)

	def get_K(self, x):
		batch_size, l, d_model = x.shape
		zeros = self.zeros.unsqueeze(0).repeat(batch_size, 1, 1).to(x.device)
		x = torch.cat((zeros, x), dim=1)
		key = torch.index_select(x, 1, self.select_index[:self.ksize*l].to(x.device))
		key = key.reshape(batch_size, l, self.ksize, -1)
		return key

class PositionalEmbedding(nn.Module):
	def __init__(self, q_len=5000, d_model=128, pos_embed_type='sincos', learnable=False, r_layers=1, c_in=21, scale=1):
		super(PositionalEmbedding, self).__init__()
		self.pos_embed_type = pos_embed_type
		self.learnable = learnable
		self.scale = scale
		if pos_embed_type == None:
			W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
			nn.init.uniform_(W_pos, -0.02, 0.02)
		elif pos_embed_type == 'zero':
			W_pos = torch.empty((q_len, 1))
			nn.init.uniform_(W_pos, -0.02, 0.02)
		elif pos_embed_type == 'zeros':
			W_pos = torch.empty((q_len, d_model))
			nn.init.uniform_(W_pos, -0.02, 0.02)
		elif pos_embed_type == 'normal' or pos_embed_type == 'gauss':
			W_pos = torch.zeros((q_len, 1))
			torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
		elif pos_embed_type == 'uniform':
			W_pos = torch.zeros((q_len, 1))
			nn.init.uniform_(W_pos, a=0.0, b=0.1)
		elif pos_embed_type == 'random': W_pos = torch.rand(c_in, q_len, d_model)
		elif pos_embed_type == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
		elif pos_embed_type == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
		elif pos_embed_type == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
		elif pos_embed_type == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
		elif pos_embed_type == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
		elif pos_embed_type == 'localrnn': W_pos = nn.Sequential(*[LocalRNN(d_model, d_model) for _ in r_layers])
		elif pos_embed_type == 'rnn': W_pos = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=r_layers)
		else: raise ValueError(f"{pos_embed_type} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
			'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
		if 'rnn' in pos_embed_type:
			self.pos = W_pos
		else:
			W_pos = W_pos.unsqueeze(0)  # [1, L, D] or [1, C, L, D]
			if learnable:
				self.pos = nn.Parameter(W_pos, requires_grad=learnable)
			else:
				self.register_buffer('pos', W_pos)
				self.pos = W_pos

	def forward(self, x):
		if 'rnn' in self.pos_embed_type:
			output, _ = self.pos(x)
			return output
		# pos generated for individual variable
		if self.pos.dim()>3:
			batch_size = x.size(0) // self.pos.size(1)
			self.pos = self.pos.repeat(batch_size, 1, 1, 1)
			self.pos = torch.reshape(self.pos, (-1, self.pos.shape[2], self.pos.shape[3]))
			return self.pos
		else:
			return self.pos[:, self.scale-1:x.size(1)*self.scale:self.scale]

class TransformerEncoderLayer(nn.Module):
	def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
		super(TransformerEncoderLayer, self).__init__()
		self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)
		self.linear1 = nn.Linear(d_model, dim_feedforward)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(dim_feedforward, d_model)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)

		self.activation = nn.LeakyReLU(True)

	def forward(self, src, src_mask=None, src_key_padding_mask=None,is_causal=None,tgt_is_causal=None,memory_is_causal=None):
		src2 = self.self_attn(src, src, src)[0]
		src = src + self.dropout1(src2)
		src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
		src = src + self.dropout2(src2)
		return src

class EncoderLayer_Smamba(nn.Module):
	def __init__(self, attention, attention_r, global_att, d_model, d_ff=None, dropout=0.1, activation="relu"):
		super(EncoderLayer_Smamba, self).__init__()
		d_ff = d_ff or 4 * d_model
		self.attention = attention
		self.attention_r = attention_r
		self.global_att = global_att
		self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
		self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
		self.dropout = nn.Dropout(dropout)
		self.activation = F.relu if activation == "relu" else F.gelu
		self.fourier_proj_a = nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 1))

		self.norm_x = nn.LayerNorm(d_model)
		self.norm_x0 = nn.LayerNorm(d_model)
		self.norm_z0 = nn.LayerNorm(d_model)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)

		self.scale_net = nn.Linear(d_model, d_model)
		self.bias_net = nn.Linear(d_model, d_model)

	def FFw(self, x, fourier_proj,norm):
		dt_fft = torch.fft.fft(x.permute(0, 2, 1), dim=-2)
		dt = torch.cat([dt_fft.real.unsqueeze(-1), dt_fft.imag.unsqueeze(-1)], dim=-1)
		dt = fourier_proj(dt).squeeze(-1)
		x = torch.fft.ifft(dt, dim=-2).real
		return norm(x.permute(0, 2, 1))

	def forward(self, x, z, r, attn_mask=None, tau=None, delta=None):
		x = self.FFw(x, self.fourier_proj_a,self.norm_x)
		global_z = self.global_att(
			z, z, z,
			attn_mask=attn_mask,
			tau=tau, delta=delta
		)[0]
		x = self.attention(x) + self.attention_r(x.flip(dims=[1])).flip(dims=[1]) + x

		x0 = self.norm_x0(x)
		z0 = self.norm_z0(global_z)

		y = torch.cat([x0, z0], dim=-2)
		_, v, _ = y.shape
		y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
		y = self.dropout(self.conv2(y).transpose(-1, 1))
		v = v // 2

		x0 = y[:, :v, :] + x0
		z0 = y[:, v:, :] + z0

		out = (1 - r) * x0 + r * z0

		return self.norm1(x0 + out),self.norm2(z0 + out), out


class Encoder_Smamba(nn.Module):
	def __init__(self, attn_layers, conv_layers=None):
		super(Encoder_Smamba, self).__init__()
		self.attn_layers = nn.ModuleList(attn_layers)
		self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None

	def forward(self, x,y, attn_mask=None, tau=None, delta=None):
		# x [B, L, D]
		for attn_layer in self.attn_layers:
			x,y = attn_layer(x,y, attn_mask=attn_mask, tau=tau, delta=delta)

		return x,y

from math import sqrt
import numpy as np
class FullAttention(nn.Module):
	def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
		super(FullAttention, self).__init__()
		self.scale = scale
		self.mask_flag = mask_flag
		self.output_attention = output_attention
		self.dropout = nn.Dropout(attention_dropout)

	def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
		B, L, H, E = queries.shape
		_, S, _, D = values.shape
		scale = self.scale or 1. / sqrt(E)

		scores = torch.einsum("blhe,bshe->bhls", queries, keys)

		if self.mask_flag:
			if attn_mask is None:
				attn_mask = TriangularCausalMask(B, L, device=queries.device)

			scores.masked_fill_(attn_mask.mask, -np.inf)

		A = self.dropout(torch.softmax(scale * scores, dim=-1))
		V = torch.einsum("bhls,bshd->blhd", A, values)

		if self.output_attention:
			return (V.contiguous(), A)
		else:
			return (V.contiguous(), None)

class TriangularCausalMask():
	def __init__(self, B, L, device="cpu"):
		mask_shape = [B, 1, L, L]
		with torch.no_grad():
			self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

	@property
	def mask(self):
		return self._mask

class AttentionLayer(nn.Module):
	def __init__(self, attention, d_model, n_heads, d_keys=None,
				 d_values=None):
		super(AttentionLayer, self).__init__()

		d_keys = d_keys or (d_model // n_heads)
		d_values = d_values or (d_model // n_heads)

		self.inner_attention = attention
		self.query_projection = nn.Linear(d_model, d_keys * n_heads)
		self.key_projection = nn.Linear(d_model, d_keys * n_heads)
		self.value_projection = nn.Linear(d_model, d_values * n_heads)
		self.out_projection = nn.Linear(d_values * n_heads, d_model)
		self.n_heads = n_heads

	def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
		B, L, _ = queries.shape
		_, S, _ = keys.shape
		H = self.n_heads

		queries = self.query_projection(queries).view(B, L, H, -1)
		keys = self.key_projection(keys).view(B, S, H, -1)
		values = self.value_projection(values).view(B, S, H, -1)

		out, attn = self.inner_attention(
			queries,
			keys,
			values,
			attn_mask,
			tau=tau,
			delta=delta
		)
		out = out.view(B, L, -1)

		return self.out_projection(out), attn

class DataEmbedding_inverted(nn.Module):
	def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
		super(DataEmbedding_inverted, self).__init__()
		self.value_embedding = nn.Linear(c_in, d_model)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x, x_mark):
		x = x.permute(0, 2, 1)

		# x: [Batch Variate Time]
		if x_mark is None:

			x = self.value_embedding(x)
		else:
			x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
		return self.dropout(x)

class EmbLayer(nn.Module):

	def __init__(self, patch_len, patch_step, seq_len, d_model):
		super().__init__()
		self.patch_len = patch_len
		self.patch_step = patch_step

		patch_num = int((seq_len - patch_len) / patch_step + 1)
		self.d_model = d_model // patch_num
		self.ff = nn.Sequential(
			nn.Linear(patch_len, self.d_model),
		)
		self.flatten = nn.Flatten(start_dim=-2)

		self.ff_1 = nn.Sequential(
			nn.Linear(self.d_model * patch_num, d_model),
		)

	def forward(self, x):
		B, V, L = x.shape
		x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_step)
		x = self.ff(x)
		x = self.flatten(x)

		x = self.ff_1(x)
		return x


class Emb(nn.Module):

	def __init__(self, seq_len, d_model, patch_len=[48, 24, 12, 6]):
		super().__init__()
		patch_step = patch_len
		d_model = d_model//4
		self.EmbLayer_1 = EmbLayer(patch_len[0], patch_step[0] // 2, seq_len, d_model)
		self.EmbLayer_2 = EmbLayer(patch_len[1], patch_step[1] // 2, seq_len, d_model)
		self.EmbLayer_3 = EmbLayer(patch_len[2], patch_step[2] // 2, seq_len, d_model)
		self.EmbLayer_4 = EmbLayer(patch_len[3], patch_step[3] // 2, seq_len, d_model)

	def forward(self, x):
		s_x1 = self.EmbLayer_1(x)
		s_x2 = self.EmbLayer_2(x)
		s_x3 = self.EmbLayer_3(x)
		s_x4 = self.EmbLayer_4(x)
		s_out = torch.cat([s_x1, s_x2, s_x3, s_x4], -1)
		return s_out

class Fourier_layer(nn.Module):
	def __init__(self, SpectralConv, conv1, conv2, norm=None):
		super(Fourier_layer, self).__init__()

		self.spec = SpectralConv
		self.conv1 = conv1
		self.conv2 = conv2
		if norm:
			self.norm = norm

	def forward(self, x):
		x = self.spec(x)
		x1 = self.conv1(x)
		x2 = self.conv2(x)
		x = x1 * x2 + x
		x = F.gelu(x)
		return self.norm(x)
