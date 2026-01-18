"""
The Borzoi model architecture and its required classes.
"""

from enformer_pytorch.modeling_enformer import exponential_linspace_int, relative_shift, GELU, get_positional_features_gamma, get_positional_features_exponential, get_positional_features_central_mask
from torch import Tensor, nn, einsum
from scipy.sparse.linalg import svds
import torch
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Union
from einops import rearrange, reduce
from einops.layers.torch import Rearrange


import torch
from torch import Tensor, nn
from scipy.sparse.linalg import svds
import torch
import numpy as np
import torch.nn.functional as F



import copy



def convert_to_lora_proper(model, linear_rank=16):
    new_model = copy.deepcopy(model)  # clone entire model

    def replace_layers(module):
        for name, child in module.named_children():
            if name == "mha":
                continue  
            replace_layers(child)
            if isinstance(child, nn.Linear):
                max_k = min(child.in_features, child.out_features)
                if linear_rank <= max_k:
                    setattr(module, name, LoraLinear(child, k=linear_rank))

    replace_layers(new_model)
    return new_model


def get_positional_embed(seq_len, feature_size, device, use_tf_gamma, dtype = torch.float):
    distances = torch.arange(-seq_len + 1, seq_len, device = device)

    assert not use_tf_gamma or seq_len == 1536, 'if using tf gamma, only sequence length of 1536 allowed for now'

    feature_functions = [
        get_positional_features_exponential,
        get_positional_features_central_mask,
        get_positional_features_gamma if not use_tf_gamma else None
    ]

    num_components = len(feature_functions) * 2

    if (feature_size % num_components) != 0:
        raise ValueError(f'feature size is not divisible by number of components ({num_components})')

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len, dtype = dtype))

    embeddings = torch.cat(embeddings, dim = -1)
    embeddings = torch.cat((embeddings, torch.sign(distances)[..., None] * embeddings), dim = -1)
    return embeddings.to(dtype)



class LoraLinear(nn.Module):
    def __init__(self, k, in_features, out_features, bias = True, device=None, dtype= None):
        super().__init__()
        if k == "full":
            self.k = "full"
        elif in_features <= k or out_features <= k:
            self.k = "full"
        else:
            self.k = k

        if self.k =="full":
            self.layer = nn.Linear(in_features,out_features, bias=bias)
        else:
        # LoRA layers
            self.loraw11 = nn.Linear(in_features, self.k, bias=False)
            self.loraw12 = nn.Linear(self.k, out_features, bias = bias)


    def make_layer(self):
        U, S, Vt = svds(self.w, k=self.k)


        w11 = np.diag(np.sqrt(S)) @ Vt  # [k, in_dim]
        w12 = U @ np.diag(np.sqrt(S))    # [out_dim, k]

        w11pt = torch.from_numpy(w11).float()
        w12pt = torch.from_numpy(w12).float()
        bpt = None if self.b is None else torch.from_numpy(self.b).float()
        return w11pt, w12pt, bpt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.k =="full":
            x = self.layer(x)
        else:
            x = self.loraw11(x)
            x = self.loraw12(x)
        return x
    def init_weights(self, w11: torch.Tensor | None = None,
                     w12: torch.Tensor | None = None,
                     b1: torch.Tensor | None = None):
        if self.k =="full":
            return
        if w11 is not None:
            assert w11.shape == (self.k, self.in_c, self.ksize)
            self.loraw11.weight.data.copy_(w11)
        if w12 is not None:
            assert w12.shape == (self.out_c, self.k, 1)
            self.loraw12.weight.data.copy_(w12)
        if b1 is not None:
            assert b1.shape == (self.out_c,)
            self.loraw12.bias.data.copy_(b1)


class WrappedConv1d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, device = None, dtype=None):
        super().__init__()

        self.layer = nn.Conv1d(in_c, out_c, kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                groups=groups,
                                bias=bias,
                                device = device,
                                dtype = dtype)




    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        return x





class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size

        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias = False)

        nn.init.dirac_(self.to_attn_logits.weight)

        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)

    def _pool_fn(self, x):
        # Use tensor ops only. This will be traced into FX as operators.
        p = self.pool_size
        # -1 in reshape is fine and traceable; torch will emit a reshape node.
        return x.reshape(x.shape[0], x.shape[1], -1, p)


    def forward(self, x, *, n=None, b=None):
        if n is None:
            b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, n), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)

        x = self._pool_fn(x)
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self._pool_fn(mask), mask_value)

        attn = logits.softmax(dim = -1)

        return (x * attn).sum(dim = -1)

def get_central_mask(x: Tensor, length, out_channels: int, device = "cpu", dtype = torch.float32) -> Tensor:
    """
    Create a positional embedding based on a central mask.

    Args:
        x : Input tensor of shape (N, L, C)
        out_channels: Number of channels in the output

    Returns:
        Positional embedding tensor of shape (L, channels)
    """
    seq_len = length
    features = out_channels // 2

    pow_rate = torch.exp(
        torch.log(torch.tensor([seq_len], device=device, dtype=dtype) + 1)
        / features
    )

    # Get the distance of each position from the center
    positions = torch.arange(-seq_len + 1, seq_len, device=device, dtype=dtype)

    # Create center widths
    center_widths = (
        pow_rate ** torch.arange(1, features + 1, device=device, dtype=dtype) - 1
    )

    # Create embeddings
    embeddings = center_widths[None, ...] > positions.abs()[..., None]

    # Create signed embeddings
    signed = torch.sign(positions)[..., None] * embeddings

    # Concatenate signed and unsigned embeddings
    embeddings = torch.cat((embeddings, signed), dim=-1)

    return embeddings


class BaseModel(nn.Module):
    """
    Base model class
    """

    def __init__(self, embedding: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.embedding = embedding
        self.head = head
        assert hasattr(self.head, "n_tasks"), "head does not have an attribute n_tasks."

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.embedding(x)
        x = self.head(x)
        return x



class Attention(nn.Module):
    def __init__(
        self,
        k: int, 
        in_len: int,
        key_len: int,
        value_len: int,
        n_heads: int,
        n_pos_features: int,
        pos_dropout: float = 0,
        attn_dropout: float = 0,
        device=None,
        dtype=None,
    ):
        """
        Multi-head Attention (MHA) layer. Modified from
        https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/modeling_enformer.py

        Args:
            in_len: Length of the input
            key_len: Length of the key vectors
            value_len: Length of the value vectors.
            n_heads: Number of attention heads
            n_pos_features: Number of positional embedding features
            pos_dropout: Dropout probability in the positional embeddings
            attn_dropout: Dropout probability in the output layer
            device: Device for the layers.
            dtype: Data type for the layers.
        """
        super().__init__()

        # Save params
        self.in_len = in_len
        self.key_len = key_len
        self.value_len = value_len
        self.n_heads = n_heads
        self.n_pos_features = n_pos_features
        self.device = device
        # print(in_len)
        # print(key_len)
        # print(value_len)
        # print(n_heads)
        # print(n_pos_features)




        # Create linear layers
        self.to_q = LoraLinear(k, 
            self.in_len,
            self.key_len * self.n_heads,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.to_k = LoraLinear(k, 
            self.in_len,
            self.key_len * self.n_heads,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.to_v = LoraLinear(k, 
            self.in_len,
            self.value_len * self.n_heads,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.to_out = LoraLinear(k,
            self.value_len * self.n_heads, self.in_len, device=device, dtype=dtype
        )

        # relative positional encoding
        self.positional_embed = get_central_mask
        self.to_pos_k = LoraLinear(k,
            self.n_pos_features,
            self.key_len * self.n_heads,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.rel_content_bias = nn.Parameter(
            torch.randn(1, self.n_heads, 1, self.key_len, device=device, dtype=dtype)
        )
        self.rel_pos_bias = nn.Parameter(
            torch.randn(1, self.n_heads, 1, self.key_len, device=device, dtype=dtype)
        )

        # dropouts
        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def _get_pos_k(self, x, length):
        positions = self.positional_embed(x, length, out_channels=self.n_pos_features, device = self.device)
        positions = self.pos_dropout(positions)
        pos_k = self.to_pos_k(positions)
        # pos_k = rearrange(pos_k, "n (h d) -> h n d", h=self.n_heads)
        n, _ = pos_k.shape  # shape: (n, h*d)
        pos_k = pos_k.view(n, self.n_heads, -1)  # shape: (n, h, d)
        pos_k = pos_k.permute(1, 0, 2) 
        return pos_k

    def get_attn_scores(self, x, return_v=False):
        # Q, K, V
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        # Get content embeddings: b n (h*d) -> b h n d
        b, n, _ = q.shape
        q = q.view(b, n, self.n_heads, -1).permute(0, 2, 1, 3)
        k = k.view(b, n, self.n_heads, -1).permute(0, 2, 1, 3)
        v = v.view(b, n, self.n_heads, -1).permute(0, 2, 1, 3)

        q = q / (self.key_len ** 0.5)

        # Content logits
        content_logits = torch.einsum(
            "b h i d, b h j d -> b h i j", q + self.rel_content_bias, k
        )

        # Positional embeddings
        # print(x.shape)
        pos_k = self._get_pos_k(x, 4096)
        # print(pos_k.shape)

        # Positional logits
        pos_logits = torch.einsum("b h i d, h j d -> b h i j", q + self.rel_pos_bias, pos_k)
        # print(pos_logits.shape)
        pos_logits = relative_shift(pos_logits)
        # print(pos_logits.shape)
        # print(content_logits.shape)

        # Add content and positional embeddings
        logits = content_logits + pos_logits

        # Softmax
        attn = logits.softmax(dim=-1)

        if return_v:
            return self.attn_dropout(attn), v
        else:
            return self.attn_dropout(attn)


    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        # Get attention scores
        attn, v = self.get_attn_scores(x, return_v=True)

        # Output
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        b, h, n, d = out.shape
        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, h * d)
        
        return self.to_out(out)


class FlashAttention(nn.Module):
    def __init__(
        self, embed_dim: int, n_heads: int, dropout_p=0.0, device=None, dtype=None
    ):
        """
        Flash Attention layer with RoPE for positional encoding.

        Args:
            embed_dim: Number of channels
            n_heads: Number of attention heads
            dropout_p: Dropout probability for attention
            device: Device for the layers.
            dtype: Data type for the layers.
        """

        super().__init__()

        try:
            from flash_attn import flash_attn_qkvpacked_func
            from flash_attn.layers.rotary import RotaryEmbedding
        except ImportError:
            raise ImportError(
                "gReLU needs to be installed with flash-attn to use Flash Attention. \
                    Please see README for instructions."
            )

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.dropout_p = dropout_p

        # Create linear layers
        self.qkv = nn.Linear(
            self.embed_dim, self.embed_dim * 3, bias=False, device=device, dtype=dtype
        )
        self.out = nn.Linear(self.embed_dim, self.embed_dim, device=device, dtype=dtype)

        # positional encoding
        self.rotary_embed = RotaryEmbedding(self.head_dim, device=device)

        # no parameters, just an operation
        self.flash_attn_qkvpacked_func = flash_attn_qkvpacked_func

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Output tensor
        """
        # Replace qkv rearrange
        qkv = self.qkv(x)  # shape: (b, l, 3 * nheads * headdim)
        b, l, _ = qkv.shape
        qkv = qkv.view(b, l, 3, self.n_heads, self.head_dim)  # (b, l, qkv, nheads, headdim)

        qkv = self.rotary_embed(qkv)

        # Replace output rearrange
        out = self.flash_attn_qkvpacked_func(qkv, self.dropout_p, window_size=(-1, -1))
        # out shape: (b, l, nheads, headdim)
        out = out.contiguous().view(b, l, self.n_heads * self.head_dim)  # (b, l, nheads*headdim)

        return self.out(out)

class EnformerAttention(nn.Module):
    def __init__(
        self,
        k_l, 
        dim,
        *,
        num_rel_pos_features,
        heads = 8,
        dim_key = 64,
        dim_value = 64,
        dropout = 0.,
        pos_dropout = 0.,
        use_tf_gamma = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.heads = heads
        self.device = device
        self.dtype = dtype
        self.n = dim_value * heads

        self.to_q = LoraLinear(k_l, dim, dim_key * heads, bias = False)
        self.to_k = LoraLinear(k_l, dim, dim_key * heads, bias = False)
        self.to_v = LoraLinear(k_l, dim, dim_value * heads, bias = False)

        self.to_out = LoraLinear(k_l, dim_value * heads, dim)
        # nn.init.zeros_(self.to_out.weight)
        # nn.init.zeros_(self.to_out.bias)

        # relative positional encoding

        self.num_rel_pos_features = num_rel_pos_features

        self.to_rel_k = LoraLinear(k_l, num_rel_pos_features, dim_key * heads, bias = False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropouts

        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # whether to use tf gamma

        self.use_tf_gamma = use_tf_gamma

    def forward(self, x):
        h = self.heads
        n = self.n
        

        # Project q, k, v
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Reshape and permute safely using inferred dimensions
        b = x.shape[0]
        d_q = q.shape[-1] // h
        d_k = k.shape[-1] // h
        d_v = v.shape[-1] // h

        q = q.view(b, n, h, d_q).transpose(1, 2)  # [b, h, n, d_q]
        k = k.view(b, n, h, d_k).transpose(1, 2)  # [b, h, n, d_k]
        v = v.view(b, n, h, d_v).transpose(1, 2)  # [b, h, n, d_v]

        q = q * self.scale

        # Content attention
        content_logits = torch.matmul(q + self.rel_content_bias, k.transpose(-2, -1))  # [b, h, n, n]

        # Relative positional attention
        positions = get_positional_embed(n, self.num_rel_pos_features, self.device,
                                        use_tf_gamma=self.use_tf_gamma,
                                        dtype=self.dtype)
        positions = self.pos_dropout(positions)
        rel_k = self.to_rel_k(positions)  # [n, h*d]
        
        # Safe reshape for relative keys
        n_rel, dh_rel = rel_k.shape
        # print("Should be 3071: ", n_rel)
        # print("Should be 512: ", dh_rel)
        d_rel = dh_rel // h
        # print("Should be 8: ", h)
        # print("Should be 64: ", d_rel)
        rel_k = rel_k.view(n_rel, h, d_rel).transpose(0, 1)  # [h, n_rel, d_rel]

        # Relative logits using matmul
        rel_logits = torch.matmul(q + self.rel_pos_bias, rel_k.transpose(1, 2))  # [b, h, n, n]
        rel_logits = relative_shift(rel_logits)

        logits = content_logits + rel_logits
        attn = torch.softmax(logits, dim=-1)
        attn = self.attn_dropout(attn)

        # Output
        out = torch.matmul(attn, v)  # [b, h, n, d_v]
        out = out.transpose(1, 2).contiguous().view(b, n, h * d_v)  # [b, n, h*d_v]

        return self.to_out(out)


class Norm(nn.Module):
    """
    A batch normalization or layer normalization layer.

    Args:
        func: Type of normalization function. Supported values are 'batch',
            'syncbatch', 'instance',  or 'layer'. If None, will return nn.Identity.
        in_dim: Number of features in the input tensor.
        **kwargs: Additional arguments to pass to the normalization function.
            Common arguments include:
            - eps: Small constant added to denominator for numerical stability.
                Defaults to 1e-5 for all normalization types unless overridden.
            - momentum: Value used for the running_mean and running_var computation.
                Defaults to 0.1 for batch and sync batch norm.
            - affine: If True, adds learnable affine parameters. Defaults to True.
            - track_running_stats: If True, tracks running mean and variance.
                Defaults to True for batch and sync batch norm.
    """

    def __init__(
        self, func: Optional[str] = None, in_dim: Optional[int] = None, **kwargs
    ) -> None:
        super().__init__()

        if func == "batch":
            if in_dim is None:
                raise ValueError("Number of input features must be provided.")
            self.layer = nn.BatchNorm1d(in_dim, **kwargs)

        elif func == "syncbatch":
            if in_dim is None:
                raise ValueError("Number of input features must be provided.")
            self.layer = nn.SyncBatchNorm(in_dim, **kwargs)

        elif func == "layer":
            if in_dim is None:
                raise ValueError("Number of input features must be provided.")
            self.layer = nn.LayerNorm(in_dim, **kwargs)

        elif func == "instance":
            if in_dim is None:
                raise ValueError("Number of input features must be provided.")
            # overwrite the defaults to make them consistant with batch norm
            kwargs = kwargs.copy()
            kwargs["affine"] = kwargs.get("affine", True)
            kwargs["track_running_stats"] = kwargs.get("track_running_stats", True)
            self.layer = nn.InstanceNorm1d(in_dim, **kwargs)

        elif func is None:
            self.layer = nn.Identity()

        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        return self.layer(x)



class Crop(nn.Module):
    """
    Optional cropping layer.

    Args:
        crop_len: Number of positions to crop at each end of the input.
        receptive_field: Receptive field of the model to calculate crop_len.
            Only needed if crop_len is None.
    """

    def __init__(
        self, crop_len: int = 0, receptive_field: Optional[int] = None
    ) -> None:
        super().__init__()
        if crop_len == 0:
            self.layer = nn.Identity()
        else:
            if crop_len == "auto":
                assert (
                    receptive_field is not None
                ), "Receptive field must be provided for autocropping"
                # crop_len = int(np.ceil(receptive_field / 2))
                crop_len = int(receptive_field // 2)
            self.layer = nn.ConstantPad1d(-crop_len, 0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        return self.layer(x)


class Activation(nn.Module):
    """
    A nonlinear activation layer.

    Args:
        func: The type of activation function. Supported values are:
            - 'relu': Standard ReLU activation
            - 'elu': Exponential Linear Unit
            - 'softplus': Softplus activation
            - 'gelu': Standard GELU activation using PyTorch's default approximation
            - 'gelu_borzoi': GELU activation using tanh approximation (different from PyTorch's default)
            - 'gelu_enformer': Custom GELU implementation from Enformer
            - 'exp': Exponential activation
            - None: Returns identity function (no activation)

    Raises:
        NotImplementedError: If 'func' is not a supported activation function.
    """

    def __init__(self, func: str) -> None:
        super().__init__()

        if func == "relu":
            self.layer = nn.ReLU()
        elif func == "elu":
            self.layer = nn.ELU()
        elif func == "gelu":
            self.layer = nn.GELU()
        elif func == "gelu_borzoi":
            self.layer = nn.GELU(approximate = 'tanh')
        elif func == "gelu_enformer":
            self.layer = GELU()
        elif func == "softplus":
            self.layer = nn.Softplus()
        elif func == "exp":
            self.layer = torch.exp
        elif func is None:
            self.layer = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        return self.layer(x)

class LinearBlock(nn.Module):
    """
    Linear layer followed by optional normalization,
    activation and dropout.

    Args:
        in_len: Length of input
        out_len: Length of output
        act_func: Name of activation function
        dropout: Dropout probability
        norm: If True, apply layer normalization
        norm_kwargs: Optional dictionary of keyword arguments to pass to the normalization layer
        bias: If True, include bias term.
        dtype: Data type of the weights
        device: Device on which to store the weights
    """

    def __init__(
        self,
        k:int, 
        in_len: int,
        out_len: int,
        act_func: str = "relu",
        dropout: float = 0.0,
        norm: bool = False,
        norm_kwargs: Optional[dict] = None,
        bias: bool = True,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()

        self.norm = Norm(
            func="layer" if norm else None, in_dim=in_len, **(norm_kwargs or dict()), dtype=dtype, device=device
        )
        self.linear = LoraLinear(k, in_len, out_len, bias=bias, dtype=dtype, device=device)
        self.dropout = Dropout(dropout)
        self.act = Activation(act_func)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.norm(x)
        x = self.linear(x)
        x = self.dropout(x)
        x = self.act(x)
        return x


class FeedForwardBlock(nn.Module):
    """
    2-layer feed-forward network. Can be used to follow layers such as GRU and attention.

    Args:
        in_len: Length of the input tensor
        dropout: Dropout probability
        act_func: Name of the activation function
        norm_kwargs: Optional dictionary of keyword arguments to pass to the normalization layers
        **kwargs: Additional arguments to be passed to the linear layers
    """

    def __init__(
        self,
        k:int, 
        in_len: int,
        dropout: float = 0.0,
        act_func: str = "relu",
        norm_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dense1 = LinearBlock(
            k, 
            in_len,
            in_len * 2,
            norm=True,
            norm_kwargs=norm_kwargs,
            dropout=dropout,
            act_func=act_func,
            bias=True,
            **kwargs,
        )
        self.dense2 = LinearBlock(
            k,
            in_len * 2,
            in_len,
            norm=False,
            norm_kwargs=norm_kwargs,
            dropout=dropout,
            act_func=None,
            bias=True,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class TransformerBlock(nn.Module):
    """
    A block containing a multi-head attention layer followed by a feed-forward
    network and residual connections.

    Args:
        in_len: Length of the input
        n_heads: Number of attention heads
        attn_dropout: Dropout probability in the output layer
        ff_droppout: Dropout probability in the linear feed-forward layers
        flash_attn: If True, uses Flash Attention with Rotational Position Embeddings. key_len, value_len,
            pos_dropout and n_pos_features are ignored.
        n_pos_features: Number of positional embedding features
        key_len: Length of the key vectors
        value_len: Length of the value vectors.
        pos_dropout: Dropout probability in the positional embeddings
        norm_kwargs: Optional dictionary of keyword arguments to pass to the normalization layers
        dtype: Data type of the weights
        device: Device on which to store the weights
    """

    flash_attn_warn = False

    def __init__(
        self,
        k:int,
        in_len: int,
        n_heads: int,
        attn_dropout: float,
        ff_dropout: float,
        flash_attn: bool,
        n_pos_features: Optional[int] = None,
        key_len: Optional[int] = None,
        value_len: Optional[int] = None,
        pos_dropout: Optional[float] = None,
        norm_kwargs: Optional[dict] = None,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()
        self.norm = Norm("layer", in_len, **(norm_kwargs or dict()))

        if flash_attn:
            if (
                not (
                    n_pos_features is None
                    and key_len is None
                    and value_len is None
                    and pos_dropout is None
                )
                and not TransformerBlock.flash_attn_warn
            ):
                warnings.warn(
                    "WARNING: FlashAttention does not use pos_dropout, key_len, value_len, n_pos_features arguments. \
                        Ignore if you are loading a pre-trained model."
                )
                TransformerBlock.flash_attn_warn = True

            self.mha = FlashAttention(
                embed_dim=in_len,
                n_heads=n_heads,
                dropout_p=attn_dropout,
                dtype=dtype,
                device=device,
            )
        else:
            self.mha = Attention(k, 
                in_len=in_len,
                n_heads=n_heads,
                n_pos_features=n_pos_features,
                key_len=key_len,
                value_len=value_len,
                pos_dropout=pos_dropout,
                attn_dropout=attn_dropout,
                dtype=dtype,
                device=device,
            )
        self.dropout = Dropout(ff_dropout)
        self.ffn = FeedForwardBlock(
            k,
            in_len=in_len,
            dropout=ff_dropout,
            act_func="relu",
            norm_kwargs=norm_kwargs,
            dtype=dtype,
            device=device,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x_input = x
        x = self.norm(x)
        x = self.mha(x)
        x = self.dropout(x)
        x = torch.add(x_input, x)
        ffn_input = x
        x = self.ffn(x)
        x = torch.add(ffn_input, x)
        return x


class TransformerTower(nn.Module):
    """
    Multiple stacked transformer encoder layers.

    Args:
        in_channels: Number of channels in the input
        n_blocks: Number of stacked transformer blocks
        n_heads: Number of attention heads
        n_pos_features: Number of positional embedding features
        key_len: Length of the key vectors
        value_len: Length of the value vectors.
        pos_dropout: Dropout probability in the positional embeddings
        attn_dropout: Dropout probability in the attention layer
        ff_dropout: Dropout probability in the feed-forward layers
        norm_kwargs: Optional dictionary of keyword arguments to pass to the normalization layers
        flash_attn: If True, uses Flash Attention with Rotational Position Embeddings
        dtype: Data type of the weights
        device: Device on which to store the weights
    """

    def __init__(
        self,
        k:int, ##linear
        in_channels: int,
        n_blocks: int = 1,
        n_heads: int = 1,
        n_pos_features: int = 32,
        key_len: int = 64,
        value_len: int = 64,
        pos_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        norm_kwargs: Optional[dict] = None,
        flash_attn: bool = False,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    k,
                    in_len=in_channels,
                    n_heads=n_heads,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    flash_attn=flash_attn,
                    n_pos_features=n_pos_features,
                    key_len=key_len,
                    value_len=value_len,
                    pos_dropout=pos_dropout,
                    norm_kwargs=norm_kwargs,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = x.permute(0, 2, 1)  # b t l -> b l t
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 1)  # b l t -> b t l
        return x


class UnetBlock(nn.Module):
    """
    Upsampling U-net block

    Args:
        in_channels: Number of channels in the input
        y_in_channels: Number of channels in the higher-resolution representation.
        norm_type: Type of normalization to apply: 'batch', 'syncbatch', 'layer', 'instance' or None
        norm_kwargs: Optional dictionary of keyword arguments to pass to the normalization layers
        act_func: Name of the activation function. Defaults to 'gelu_borzoi' which uses
            tanh approximation (different from PyTorch's default GELU implementation).
        dtype: Data type of the weights
        device: Device on which to store the weights
    """

    def __init__(
        self,
      
        in_channels: int,
        y_in_channels: int,
        norm_type="batch",
        norm_kwargs: Optional[dict] = None,
        act_func="gelu_borzoi",
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()
        self.conv = ConvBlock( 
            in_channels,
            in_channels,
            1,
            norm=True,
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
            act_func=act_func,
            order="NACDR",
            dtype=dtype,
            device=device,
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.channel_transform = ChannelTransformBlock(
            y_in_channels,
            in_channels,
            norm=True,
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
            act_func=act_func,
            order="NACD",
            if_equal=True,
            dtype=dtype,
            device=device,
        )
        self.sconv = SeparableConv(in_channels, 3, dtype=dtype, device=device)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.conv(x)
        x = self.upsample(x)
        x = torch.add(x, self.channel_transform(y))
        x = self.sconv(x)
        return x



class SeparableConv(nn.Module):
    """
    Equivalent class to `tf.keras.layers.SeparableConv1D`

    Args:
        in_channels: Number of channels in the input
        kernel_size: Convolutional kernel width
        dtype: Data type of the weights
        device: Device on which to store the weights
    """

    def __init__(
        self, in_channels: int, kernel_size: int, dtype=None, device=None
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            padding=[(kernel_size - 1) // 2],
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.pointwise = WrappedConv1d(
            in_channels,
            in_channels,
            kernel_size=1,
            bias=True,
            dtype=dtype,
            device=device,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class UnetTower(nn.Module):
    """
    Upsampling U-net tower for the Borzoi model

    Args:
        in_channels: Number of channels in the input
        y_in_channels: Number of channels in the higher-resolution representations.
        n_blocks: Number of U-net blocks
        act_func: Name of the activation function. Defaults to 'gelu_borzoi' which uses
            tanh approximation (different from PyTorch's default GELU implementation).
        kwargs: Additional arguments to be passed to the U-net blocks
    """

    def __init__(
        self, in_channels: int, y_in_channels: List[int], n_blocks: int, act_func: str = "gelu_borzoi", **kwargs
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for y_c in y_in_channels:
            self.blocks.append(UnetBlock(in_channels, y_c, act_func=act_func, **kwargs))

    def forward(self, x: Tensor, ys: List[Tensor]) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)
            ys: Higher-resolution representations

        Returns:
            Output tensor
        """
        for b, y in zip(self.blocks, ys):
            x = b(x, y)
        return x



class ConvBlock(nn.Module):
    """
    Convolutional layer along with optional normalization,
    activation, dilation, dropout, residual connection, and pooling.
    The order of these operations can be specified, except
    for pooling, which always comes last.

    Args:
        in_channels: Number of channels in the input
        out_channels: Number of channels in the output
        kernel_size: Convolutional kernel width
        dilation: Dilation
        act_func: Name of the activation function
        pool_func: Name of the pooling function
        pool_size: Pooling width
        dropout: Dropout probability
        norm: If True, apply normalization layer
        norm_type: Type of normalization to apply: 'batch', 'syncbatch', 'layer', 'instance' or None
        norm_kwargs: Additional arguments to be passed to the normalization layer
        residual: If True, apply residual connection
        order: A string representing the order in which operations are
            to be performed on the input. For example, "CDNRA" means that the
            operations will be performed in the order: convolution, dropout,
            batch norm, residual addition, activation. Pooling is not included
            as it is always performed last.
        return_pre_pool: If this is True and pool_func is not None, the final
            output will be a tuple (output after pooling, output_before_pooling).
            This is useful if the output before pooling is required by a later
            layer.
        dtype: Data type of the weights
        device: Device on which to store the weights
        **kwargs: Additional arguments to be passed to nn.Conv1d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        act_func: str = "relu",
        pool_func: Optional[str] = None,
        pool_size: Optional[str] = None,
        dropout: float = 0.0,
        norm: bool = True,
        norm_type="batch",
        norm_kwargs: Optional[dict] = None,
        residual: bool = False,
        order: str = "CDNRA",
        bias: bool = True,
        return_pre_pool: bool = False,
        dtype=None,
        device=None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Check order
        assert sorted(order) == [
            "A",
            "C",
            "D",
            "N",
            "R",
        ], "The string supplied in order must contain one occurrence each of A, C, D, N and R."
        self.order = order

        # Create norm
        if norm:
            if self.order.index("N") > self.order.index("C"):
                self.norm = Norm(
                    norm_type,
                    in_dim=out_channels,
                    dtype=dtype,
                    device=device,
                    **(norm_kwargs or dict()),
                )
            else:
                self.norm = Norm(
                    norm_type,
                    in_dim=in_channels,
                    dtype=dtype,
                    device=device,
                    **(norm_kwargs or dict()),
                )
        else:
            self.norm = Norm(None)

        # Create other layers
        self.conv = WrappedConv1d( 
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=[(kernel_size - 1) // 2],
            dilation=dilation,
            dtype=dtype,
            device=device,
            **kwargs,
        )
        self.act = Activation(act_func)
        self.pool = Pool(func=pool_func, pool_size=pool_size, in_channels=out_channels)
        self.dropout = Dropout(dropout)
        self.residual = residual
        if self.residual:
            self.channel_transform = ChannelTransform(
                in_channels, out_channels, dtype=dtype, device=device
            )
        self.order = order
        assert (
            len(set(self.order).difference(set("CDNRA"))) == 0
        ), "The string supplied in order contains a non-recognized letter."
        self.return_pre_pool = return_pre_pool

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x : Input data.
        """
        if self.residual:
            x_input = self.channel_transform(x)

        # Intermediate layers
        for name in self.order:
            if name == "C":
                x = self.conv(x)
            elif name == "D":
                x = self.dropout(x)
            elif name == "N":
                x = self.norm(x)
            elif name == "R":
                if self.residual:
                    x = torch.add(x, x_input)
            elif name == "A":
                x = self.act(x)

        # Pool
        if self.return_pre_pool:
            return self.pool(x), x
        else:
            return self.pool(x)


class ChannelTransform(nn.Module):
    """
    A convolutional layer to transform the number of channels in the input.

    Args:
        in_channels: Number of channels in the input
        out_channels: Number of channels in the output
        if_equal: Whether to create layer if input and output channels are equal
        **kwargs: Additional arguments to pass to the convolutional layer.
    """

    def __init__(
        self, in_channels: int, out_channels: int = 1, if_equal: bool = False, **kwargs
    ) -> None:
        super().__init__()
        if (in_channels == out_channels) and (not if_equal):
            self.layer = nn.Identity()
        else:
            self.layer = WrappedConv1d( 
                in_channels, out_channels, kernel_size=1, padding=0  , **kwargs
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        return self.layer(x)


class Dropout(nn.Module):
    """
    Optional dropout layer

    Args:
        p: Dropout probability. If this is set to 0, will return nn.Identity.
    """

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.layer = nn.Dropout(p) if p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        return self.layer(x)



class AdaptivePool(nn.Module):
    """
    An Adaptive Pooling layer. This layer does not have a defined pooling width but
    instead pools together all the values in the last axis.

    Args:
        func: Type of pooling function. Supported values are 'avg' or 'max'. If None,
            will return nn.Identity.

    Raises:
        NotImplementedError: If 'func' is not a supported pooling function.
    """

    def __init__(self, func: Optional[str] = None) -> None:
        super().__init__()

        if func == "avg":
            self.layer = nn.AdaptiveAvgPool1d(1)
        elif func == "max":
            self.layer = nn.AdaptiveMaxPool1d(1)
        elif func is None:
            self.layer = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        return self.layer(x)

class Pool(nn.Module):
    """
    A pooling layer.

    Args:
        func: Type of pooling function. Supported values are 'avg', 'max',
            or 'attn'. If None, will return nn.Identity.
        pool_size: The number of positions to pool together
        in_channels: Number of channels in the input. Only needeed for attention pooling.
        **kwargs: Additional arguments to pass to the pooling function.

    Raises:
        NotImplementedError: If 'func' is not a supported pooling function.
    """

    def __init__(
        self,
        func: Optional[str],
        pool_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.func = func

        if func == "avg":
            self.layer = nn.AvgPool1d(kernel_size=pool_size, **kwargs)
        elif func == "max":
            self.layer = nn.MaxPool1d(kernel_size=pool_size, **kwargs)
        elif func == "attn":
            if in_channels is None:
                raise ValueError("The number of input channels must be provided.")
            self.layer = AttentionPool(dim=in_channels, pool_size=pool_size, **kwargs)
            self.n = in_channels
            self.b = 1
        elif func is None:
            self.layer = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        if self.func != "attn":
            return self.layer(x)
        else:
            return self.layer(x, n=self.n, b=self.b)


class Stem(nn.Module):
    """
    Convolutional layer followed by optional activation and pooling.
    Meant to take one-hot encoded DNA sequence as input

    Args:
        out_channels: Number of channels in the output
        kernel_size: Convolutional kernel width
        act_func: Name of the activation function
        pool_func: Name of the pooling function
        pool_size: Width of pooling layer
        dtype: Data type of the weights
        device: Device on which to store the weights
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        act_func: str = "relu",
        pool_func: Optional[str] = None,
        pool_size: Optional[str] = None,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()
        self.conv = WrappedConv1d(
            4,
            out_channels,
            kernel_size,
            stride=1,
            padding=[(kernel_size - 1) // 2],
            dilation=1,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self.act = Activation(act_func)
        self.pool = Pool(pool_func, pool_size=pool_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.conv(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class BorzoiConvTower(nn.Module):
    """
    Convolutional tower for the Borzoi model.

    Args:
        stem_channels: Number of channels in the first (stem) convolutional layer
        stem_kernel_size:  Width of the convolutional kernel in the first (stem) convolutional layer
        init_channels: Number of channels in the first convolutional block after the stem
        out_channels: Number of channels in the output
        kernel_size: Width of the convolutional kernel
        n_blocks: Number of convolutional/pooling blocks, including the stem
        norm_type: Type of normalization to apply: 'batch', 'syncbatch', 'layer', 'instance' or None
        norm_kwargs: Additional arguments to be passed to the normalization layer
        act_func: Name of the activation function. Defaults to 'gelu_borzoi' which uses
            tanh approximation (different from PyTorch's default GELU implementation).
        dtype: Data type for the layers.
        device: Device for the layers.
    """

    def __init__(
        self,
        stem_channels: int,
        stem_kernel_size: int,
        init_channels: int,
        out_channels: int,
        kernel_size: int,
        n_blocks: int,
        norm_type="batch",
        norm_kwargs=None,
        act_func="gelu_borzoi",
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()

        # Empty list
        self.blocks = nn.ModuleList()

        # Add stem
        self.blocks.append(
            Stem(
                out_channels=stem_channels,
                kernel_size=stem_kernel_size,
                act_func=None,
                pool_func="max",
                pool_size=2,
                dtype=dtype,
                device=device,
            )
        )

        # Get number of channels for the remaining n_blocks-1 blocks
        self.filters = [stem_channels] + exponential_linspace_int(
            init_channels, out_channels, (n_blocks - 1), 32
        )

        for i in range(1, n_blocks):
            self.blocks.append(
                ConvBlock(
                    in_channels=self.filters[i - 1],
                    out_channels=self.filters[i],
                    kernel_size=kernel_size,
                    norm=True,
                    norm_type=norm_type,
                    norm_kwargs=norm_kwargs,
                    act_func=act_func,
                    order="NACDR",
                    pool_func="max",
                    pool_size=2,
                    return_pre_pool=(i > (n_blocks - 3)),
                    dtype=dtype,
                    device=device,
                )
            )
        assert len(self.blocks) == n_blocks

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        for block in self.blocks[:-2]:
            x = block(x)
        x, y1 = self.blocks[-2](x)
        x, y0 = self.blocks[-1](x)
        return x, y0, y1

class ChannelTransformBlock(nn.Module):
    """
    Convolutional layer with kernel size=1 along with optional normalization, activation
    and dropout

    Args:
        in_channels: Number of channels in the input
        out_channels: Number of channels in the output
        act_func: Name of the activation function
        dropout: Dropout probability
        norm_type: Type of normalization to apply: 'batch', 'syncbatch', 'layer', 'instance' or None
        norm_kwargs: Optional dictionary of keyword arguments to pass to the normalization layers
        order: A string representing the order in which operations are
            to be performed on the input. For example, "CDNA" means that the
            operations will be performed in the order: convolution, dropout,
            batch norm, activation.
        if_equal: If True, create a layer even if the input and output channels are equal.
        device: Device on which to store the weights
        dtype: Data type of the weights
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool = False,
        act_func: str = "relu",
        dropout: float = 0.0,
        order: str = "CDNA",
        norm_type="batch",
        norm_kwargs: Optional[dict] = None,
        if_equal: bool = False,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()

        # Check order
        assert sorted(order) == [
            "A",
            "C",
            "D",
            "N",
        ], "The string supplied in order must contain one occurrence each of A, C, D and N."
        self.order = order

        # Create batch norm
        if norm:
            if self.order.index("N") > self.order.index("C"):
                self.norm = Norm(
                    norm_type,
                    in_dim=out_channels,
                    dtype=dtype,
                    device=device,
                    **(norm_kwargs or dict()),
                )
            else:
                self.norm = Norm(
                    "batch",
                    in_dim=in_channels,
                    dtype=dtype,
                    device=device,
                    **(norm_kwargs or dict()),
                )
        else:
            self.norm = Norm(None)

        # Create other layers
        self.conv = ChannelTransform(
            in_channels, out_channels, if_equal=if_equal, dtype=dtype, device=device
        )
        self.act = Activation(act_func)
        self.dropout = Dropout(dropout)
        self.order = order

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        for name in self.order:
            if name == "C":
                x = self.conv(x)
            elif name == "D":
                x = self.dropout(x)
            elif name == "N":
                x = self.norm(x)
            elif name == "A":
                x = self.act(x)
        return x


class ConvHead(nn.Module):
    """
    A 1x1 Conv layer that transforms the the number of channels in the input and then
    optionally pools along the length axis.

    Args:
        n_tasks: Number of tasks (output channels)
        in_channels: Number of channels in the input
        norm: If True, batch normalization will be included.
        act_func: Activation function for the convolutional layer
        pool_func: Pooling function.
        norm: If True, batch normalization will be included.
        norm_kwargs: Optional dictionary of keyword arguments to pass to the normalization layer
        dtype: Data type for the layers.
        device: Device for the layers.
    """

    def __init__(
        self,
        n_tasks: int,
        in_channels: int,
        act_func: Optional[str] = None,
        pool_func: Optional[str] = None,
        norm: bool = False,
        norm_kwargs: Optional[dict] = None,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()
        # Save all params
        self.n_tasks = n_tasks
        self.in_channels = in_channels
        self.act_func = act_func
        self.pool_func = pool_func
        self.norm = norm

        # Create layers
        self.channel_transform = ChannelTransformBlock( 
            self.in_channels,
            self.n_tasks,
            act_func=self.act_func,
            norm=self.norm,
            norm_kwargs=(norm_kwargs or dict()),
            dtype=dtype,
            device=device,
        )
        self.pool = AdaptivePool(self.pool_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input data.
        """
        x = self.channel_transform(x)
        x = self.pool(x)
        return x




class BorzoiTrunk(nn.Module):
    """
    Trunk consisting of conv, transformer and U-net layers for the Borzoi model.

    Args:
        stem_channels: Number of channels in the first (stem) convolutional layer
        stem_kernel_size:  Width of the convolutional kernel in the first (stem) convolutional layer
        init_channels: Number of channels in the first convolutional block after the stem
        n_conv: Number of convolutional/pooling blocks, including the stem
        kernel_size: Width of the convolutional kernel
        channels: Number of channels in the output
        n_transformers: Number of transformer blocks
        key_len: Length of the key
        value_len: Length of the value
        pos_dropout: Dropout rate for positional embeddings
        attn_dropout: Dropout rate for attention
        n_heads: Number of attention heads
        n_pos_features: Number of positional features
        crop_len: Length of the crop
        flash_attn: If True, uses Flash Attention with Rotational Position Embeddings. key_len, value_len,
            pos_dropout and n_pos_features are ignored.
        norm_type: Type of normalization to apply: 'batch', 'syncbatch', 'layer', 'instance' or None
        norm_kwargs: Additional arguments to be passed to the normalization layer
        act_func: Name of the activation function. Defaults to 'gelu_borzoi' which uses
            tanh approximation (different from PyTorch's default GELU implementation).
        dtype: Data type for the layers.
        device: Device for the layers.
    """

    def __init__(
        self,
        k_l:int, 
        # Stem
        stem_channels: int,
        stem_kernel_size: int,
        # Conv tower
        init_channels: int,
        n_conv: int,
        kernel_size: int,
        channels: int,
        # Transformer tower
        n_transformers: int,
        key_len: int,
        value_len: int,
        pos_dropout: float,
        attn_dropout: float,
        ff_dropout: float,
        n_heads: int,
        n_pos_features: int,
        # Crop
        crop_len: int,
        flash_attn: bool,
        norm_type="batch",
        norm_kwargs=None,
        act_func="gelu_borzoi",
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()

        self.conv_tower = BorzoiConvTower(
            stem_channels=stem_channels,
            stem_kernel_size=stem_kernel_size,
            init_channels=init_channels,
            out_channels=channels,
            kernel_size=kernel_size,
            n_blocks=n_conv,
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
            act_func=act_func,
            dtype=dtype,
            device=device,
        )
        self.transformer_tower = TransformerTower(k_l, 
            n_blocks=n_transformers,
            in_channels=channels,
            key_len=key_len,
            value_len=value_len,
            pos_dropout=pos_dropout,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            norm_kwargs=norm_kwargs,
            n_heads=n_heads,
            n_pos_features=n_pos_features,
            flash_attn=flash_attn,
            dtype=dtype,
            device=device,
        )
        self.unet_tower = UnetTower(
            n_blocks=2,
            in_channels=channels,
            y_in_channels=[channels, self.conv_tower.filters[-2]],
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
            act_func=act_func,
            dtype=dtype,
            device=device,
        )
        self.pointwise_conv = ConvBlock( 
            in_channels=channels,
            out_channels=round(channels * 1.25),
            kernel_size=1,
            act_func=act_func,
            dropout=0.1,
            norm=True,
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
            order="NACDR",
            device=device,
            dtype=dtype,
        )
        self.act = Activation(act_func)
        self.crop = Crop(crop_len=crop_len)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x, y0, y1 = self.conv_tower(x)
        x = self.transformer_tower(x)
        x = self.unet_tower(x, [y0, y1])
        x = self.pointwise_conv(x)
        x = self.act(x)
        x = self.crop(x)
        return x

class BorzoiModel(BaseModel):
    """
    Model consisting of Borzoi conv and transformer layers followed by U-net upsampling and optional pooling.

    Args:
        stem_channels: Number of channels in the first (stem) convolutional layer
        stem_kernel_size:  Width of the convolutional kernel in the first (stem) convolutional layer
        init_channels: Number of channels in the first convolutional block after the stem
        channels: Number of channels in the output of the convolutional tower
        kernel_size: Width of the convolutional kernel
        n_conv: Number of convolutional/pooling blocks
        n_transformers: Number of stacked transformer blocks
        n_pos_features: Number of features in the positional embeddings
        n_heads: Number of attention heads
        key_len: Length of the key vectors
        value_len: Length of the value vectors.
        pos_dropout: Dropout probability in the positional embeddings
        attn_dropout: Dropout probability in the attention layer
        crop_len: Number of positions to crop at either end of the output
        head_act_func: Name of the activation function to use in the final layer
        final_pool_func: Name of the pooling function to apply to the final output.
            If None, no pooling will be applied at the end.
        flash_attn: If True, uses Flash Attention with Rotational Position Embeddings. key_len, value_len,
            pos_dropout and n_pos_features are ignored.
        norm_kwargs: Optional dictionary of keyword arguments to pass to the normalization layers.
            Defaults to {"eps": 0.001}.
        act_func: Name of the activation function. Defaults to 'gelu_borzoi' which uses
            tanh approximation (different from PyTorch's default GELU implementation).
        dtype: Data type for the layers.
        device: Device for the layers.
    """

    def __init__(
        self,
        k_l:int,
        n_tasks: int,
        # Stem
        stem_channels: int = 512,
        stem_kernel_size: int = 15,
        # Conv tower
        init_channels: int = 608,
        channels: int = 1536,
        n_conv: int = 7,
        kernel_size: int = 5,
        # Transformer tower
        n_transformers: int = 8,
        key_len: int = 64,
        value_len: int = 192,
        pos_dropout: float = 0.01,
        attn_dropout: float = 0.05,
        ff_dropout: float = 0.2,
        norm_kwargs: Optional[dict] = None,
        n_heads: int = 8,
        n_pos_features: int = 32,
        # Head
        crop_len: int = 16,
        act_func: str = "gelu_borzoi",
        final_act_func: Optional[str] = None,
        final_pool_func: Optional[str] = "avg",
        flash_attn=False,
        dtype=None,
        device=None,
    ) -> None:
        norm_kwargs = norm_kwargs or {"eps": 0.001}
        super().__init__(
            embedding=BorzoiTrunk(k_l,
                stem_channels=stem_channels,
                stem_kernel_size=stem_kernel_size,
                init_channels=init_channels,
                channels=channels,
                n_conv=n_conv,
                kernel_size=kernel_size,
                n_transformers=n_transformers,
                key_len=key_len,
                value_len=value_len,
                pos_dropout=pos_dropout,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                norm_kwargs=norm_kwargs,
                n_heads=n_heads,
                n_pos_features=n_pos_features,
                crop_len=crop_len,
                flash_attn=flash_attn,
                act_func=act_func,
                dtype=dtype,
                device=device,
            ),
            head=ConvHead( 
                n_tasks,
                in_channels=round(channels * 1.25),
                norm=False,
                norm_kwargs=norm_kwargs,
                act_func=final_act_func,
                pool_func=final_pool_func,
                dtype=dtype,
                device=device,
            ),
        )
        self.device  = device


class BorzoiPretrainedModel(BaseModel):
    """
    Borzoi model with published weights (ported from Keras).

    Args:
        n_tasks: Number of tasks for the model to predict
        fold: Which fold of the model to load (default=0)
        n_transformers: Number of transformer blocks to use (default=8)
        crop_len: Number of positions to crop at either end of the output (default=0)
        act_func: Name of the activation function. Defaults to 'gelu_borzoi' which uses
            tanh approximation (different from PyTorch's default GELU implementation).
        norm_kwargs: Optional dictionary of keyword arguments to pass to the normalization layers.
            Defaults to {"eps": 0.001}.
        final_pool_func: Name of the pooling function to apply to the final output (default="avg")
        dtype: Data type for the layers
        device: Device for the layers
    """

    def __init__(
        self,
        k_l:int,
        n_tasks: int,
        # weights
        fold: int = 0,
        n_transformers: int = 8,
        # head
        crop_len=0,
        act_func="gelu_borzoi",
        norm_kwargs: Optional[dict] = None,
        final_pool_func="avg",
        dtype=None,
        device=None,
    ):
        norm_kwargs = norm_kwargs or {"eps": 0.001}
        model = BorzoiModel(
            k_l,
            crop_len=crop_len,
            n_tasks=7611,
            stem_channels=512,
            stem_kernel_size=15,
            init_channels=608,
            n_conv=7,
            kernel_size=5,
            n_transformers=8,
            key_len=64,
            value_len=192,
            pos_dropout=0.01,
            attn_dropout=0.05,
            ff_dropout=0.2,
            norm_kwargs=norm_kwargs,
            n_heads=8,
            n_pos_features=32,
            act_func=act_func,
            final_act_func=None,
            final_pool_func=None,
            dtype=dtype,
            device=device,
        )

        # Load state dict
        from grelu.resources import get_artifact

        art = get_artifact(
            f"human_state_dict_fold{fold}", project="borzoi", alias="latest"
        )
        with TemporaryDirectory() as d:
            art.download(d)
            state_dict = torch.load(Path(d) / f"fold{fold}.h5")

        model.load_state_dict(state_dict)

        # Fix depth
        model.embedding.transformer_tower.blocks = (
            model.embedding.transformer_tower.blocks[:n_transformers]
        )

        # Change head
        head = ConvHead( 
            n_tasks=n_tasks,
            in_channels=1920,
            pool_func=final_pool_func,
            dtype=dtype,
            device=device,
        )

        super().__init__(embedding=model.embedding, head=head)


class EnformerModel(BaseModel):
    """
    Enformer model architecture.

    Args:
        n_tasks: Number of tasks for the model to predict
        n_conv: Number of convolutional/pooling blocks
        channels: Number of output channels for the convolutional tower
        n_transformers: Number of stacked transformer blocks
        n_heads: Number of attention heads
        key_len: Length of the key vectors
        value_len: Length of the value vectors.
        pos_dropout: Dropout probability in the positional embeddings
        attn_dropout: Dropout probability in the output layer
        ff_droppout: Dropout probability in the linear feed-forward layers
        crop_len: Number of positions to crop at either end of the output
        final_act_func: Name of the activation function to use in the final layer
        final_pool_func: Name of the pooling function to apply to the final output.
            If None, no pooling will be applied at the end.
        dtype: Data type for the layers.
        device: Device for the layers.
    """

    def __init__(
        self,
        k_l,
        n_tasks: int,
        # Conv
        n_conv: int = 7,
        channels: int = 1536,
        # Transformer
        n_transformers: int = 11,
        n_heads: int = 8,
        key_len: int = 64,
        attn_dropout: float = 0.05,
        pos_dropout: float = 0.01,
        ff_dropout: float = 0.4,
        # Crop
        crop_len: int = 0,
        # Head
        final_act_func: Optional[str] = None,
        final_pool_func: Optional[str] = "avg",
        dtype=None,
        device=None,
    ) -> None:
        super().__init__(
            embedding=EnformerTrunk(
                k_l, 
                n_conv=n_conv,
                channels=channels,
                n_transformers=n_transformers,
                n_heads=n_heads,
                key_len=key_len,
                attn_dropout=attn_dropout,
                pos_dropout=pos_dropout,
                ff_dropout=ff_dropout,
                crop_len=crop_len,
                dtype=dtype,
                device=device,
            ),
            head=ConvHead( 
                n_tasks=n_tasks,
                in_channels=2 * channels,
                act_func=final_act_func,
                norm=False,
                pool_func=final_pool_func,
                dtype=dtype,
                device=device,
            ),
        )


class EnformerPretrainedModel(BaseModel):
    """
    Borzoi model with published weights (ported from Keras).
    """

    def __init__(
        self,
        k_l,
        n_tasks: int,
        n_transformers: int = 11,
        # head
        crop_len=0,
        final_pool_func="avg",
        dtype=None,
        device=None,
    ):
        model = EnformerModel(k_l,
            crop_len=crop_len,
            n_tasks=5313,
            channels=1536,
            n_transformers=11,
            n_heads=8,
            key_len=64,
            attn_dropout=0.05,
            pos_dropout=0.01,
            ff_dropout=0.4,
            final_act_func=None,
            final_pool_func=None,
            dtype=dtype,
            device=device,
        )

        # Load state dict
        from grelu.resources import get_artifact

        art = get_artifact("human_state_dict", project="enformer", alias="latest")
        with TemporaryDirectory() as d:
            art.download(d)
            state_dict = torch.load(Path(d) / "human.h5")

        model.load_state_dict(state_dict)

        # Fix depth
        model.embedding.transformer_tower.blocks = (
            model.embedding.transformer_tower.blocks[:n_transformers]
        )

        # Change head
        head = ConvHead( 
            n_tasks=n_tasks,
            in_channels=3072,
            pool_func=final_pool_func,
            dtype=dtype,
            device=device,
        )

        super().__init__(embedding=model.embedding, head=head)


class EnformerConvTower(nn.Module):
    """
    Args:
        n_blocks: Number of convolutional/pooling blocks including the stem.
        out_channels: Number of channels in the output
        dtype: Data type for the layers.
        device: Device for the layers.
    """

    def __init__(
        self,
        n_blocks: int,
        out_channels: int,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()
        half_dim = out_channels // 2

        # Empty list
        self.blocks = nn.ModuleList()

        # Add stem
        self.blocks.append(
            nn.Sequential(
                WrappedConv1d(4, half_dim, 15, padding=[(15 - 1) // 2], device=device, dtype=dtype),
                ConvBlock(
                    in_channels=half_dim,
                    out_channels=half_dim,
                    kernel_size=1,
                    act_func="gelu_enformer",
                    residual=True,
                    order="NACDR",
                    pool_func="attn",
                    pool_size=2,
                    dtype=dtype,
                    device=device,
                ),
            )
        )

        # List input and output channels for the remaining n_blocks - 1 blocks
        filters = [half_dim] + exponential_linspace_int(
            half_dim, out_channels, num=(n_blocks - 1), divisible_by=128
        )

        # Add the remaining n_blocks - 1 blocks
        for i in range(1, n_blocks):
            self.blocks.append(
                nn.Sequential(
                    ConvBlock(
                        in_channels=filters[i - 1],
                        out_channels=filters[i],
                        kernel_size=5,
                        act_func="gelu_enformer",
                        residual=False,
                        order="NACDR",
                        dtype=dtype,
                        device=device,
                    ),
                    ConvBlock(
                        in_channels=filters[i],
                        out_channels=filters[i],
                        kernel_size=1,
                        act_func="gelu_enformer",
                        residual=True,
                        order="NACDR",
                        pool_func="attn",
                        pool_size=2,
                        dtype=dtype,
                        device=device,
                    ),
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        for block in self.blocks:
            x = block(x)
        return x


class EnformerTransformerBlock(nn.Module):
    """
    Transformer tower for enformer model

    Args:
        in_len: Length of the input
        n_blocks: Number of stacked transformer blocks
        n_heads: Number of attention heads
        n_pos_features: Number of positional embedding features
        key_len: Length of the key vectors
        value_len: Length of the value vectors.
        pos_dropout: Dropout probability in the positional embeddings
        attn_dropout: Dropout probability in the output layer
        ff_droppout: Dropout probability in the linear feed-forward layers
        dtype: Data type for the layers.
        device: Device for the layers.
    """

    def __init__(
        self,
        k_l,
        in_len: int,
        n_heads: int,
        key_len: int,
        attn_dropout: float,
        pos_dropout: float,
        ff_dropout: float,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()
        self.norm = Norm("layer", in_len)
        self.mha = EnformerAttention(k_l, 
            dim=in_len,
            heads=n_heads,
            dim_key=key_len,
            dim_value=in_len // n_heads,
            dropout=attn_dropout,
            pos_dropout=pos_dropout,
            num_rel_pos_features=in_len // n_heads,
            use_tf_gamma=False, device = device, dtype=dtype,
        ).to(device=device, dtype=dtype)
        self.dropout = Dropout(ff_dropout)
        self.ffn = FeedForwardBlock(k_l, 
            in_len=in_len,
            dropout=ff_dropout,
            act_func="relu",
            dtype=dtype,
            device=device,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x_input = x
        x = self.norm(x)
        x = self.mha(x)
        x = self.dropout(x)
        x = torch.add(x_input, x)
        ffn_input = x
        x = self.ffn(x)
        x = torch.add(ffn_input, x)
        return x


class EnformerTransformerTower(nn.Module):
    """
    Transformer tower for enformer model

    Args:
        in_channels: Number of channels in the input
        n_blocks: Number of stacked transformer blocks
        n_heads: Number of attention heads
        n_pos_features: Number of positional embedding features
        key_len: Length of the key vectors
        value_len: Length of the value vectors.
        pos_dropout: Dropout probability in the positional embeddings
        attn_dropout: Dropout probability in the output layer
        ff_droppout: Dropout probability in the linear feed-forward layers
        device: Device for the layers.
        dtype: Data type for the layers.
    """

    def __init__(
        self,
        k_l, 
        in_channels: int,
        n_blocks: int,
        n_heads: int,
        key_len: int,
        attn_dropout: float,
        pos_dropout: float,
        ff_dropout: float,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                EnformerTransformerBlock(k_l, 
                    in_len=in_channels,
                    n_heads=n_heads,
                    key_len=key_len,
                    attn_dropout=attn_dropout,
                    pos_dropout=pos_dropout,
                    ff_dropout=ff_dropout,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 1)
        return x


class EnformerTrunk(nn.Module):
    """
    Enformer model architecture.

    Args:
        n_conv: Number of convolutional/pooling blocks
        channels: Number of output channels for the convolutional tower
        n_transformers: Number of stacked transformer blocks
        n_heads: Number of attention heads
        key_len: Length of the key vectors
        value_len: Length of the value vectors.
        pos_dropout: Dropout probability in the positional embeddings
        attn_dropout: Dropout probability in the output layer
        ff_droppout: Dropout probability in the linear feed-forward layers
        crop_len: Number of positions to crop at either end of the output
        dtype: Data type for the layers.
        device: Device for the layers.
    """

    def __init__(
        self,
        k_l, 
        # Conv
        n_conv: int = 7,
        channels: int = 1536,
        # Transformer
        n_transformers: int = 11,
        n_heads: int = 8,
        key_len: int = 64,
        attn_dropout: float = 0.05,
        pos_dropout: float = 0.01,
        ff_dropout: float = 0.4,
        # Crop
        crop_len: int = 0,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()

        self.conv_tower = EnformerConvTower(n_blocks=n_conv, out_channels=channels)
        self.transformer_tower = EnformerTransformerTower(k_l, 
            in_channels=channels,
            n_blocks=n_transformers,
            n_heads=n_heads,
            key_len=key_len,
            attn_dropout=attn_dropout,
            pos_dropout=pos_dropout,
            ff_dropout=ff_dropout,
            dtype=dtype,
            device=device,
        )
        self.pointwise_conv = ConvBlock( 
            in_channels=channels,
            out_channels=channels * 2,
            kernel_size=1,
            act_func="gelu_enformer",
            dropout=ff_dropout // 8,
            order="NACDR",
            dtype=dtype,
            device=device,
        )
        self.act = Activation("gelu_enformer")
        self.crop = Crop(crop_len)

    def forward(self, x):
        x = self.conv_tower(x)
        x = self.transformer_tower(x)
        x = self.pointwise_conv(x)
        x = self.act(x)
        x = self.crop(x)
        return x