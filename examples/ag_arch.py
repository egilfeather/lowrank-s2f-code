"""
The AG model architecture and its required classes.

Code modified from https://github.com/genomicsxai/alphagenome-pytorch/
Copyright 2026 genomicsxai

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
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




class LoraLinear(nn.Module):
    def __init__(self, k, in_features, out_features, bias = True, device=None, dtype= None):
        super().__init__()
        # self.in_features = layer.in_features
        # self.out_features = layer.out_features
        
        # self.w = layer.weight.detach().cpu().numpy()
        # self.b = None if layer.bias is None else layer.bias.detach().cpu().numpy()
        # self.out_dim, self.in_dim = self.w.shape

        # Clamp k to valid range
        # max_k = min(self.in_dim, self.out_dim) - 1  # svds requires k < min(A.shape)
        # if k > max_k:
        #     print(f"[LoraLinear] Requested rank {k} too large, using k={max_k}")
        #     k = max_k
        # if k <= 0:
        #     raise ValueError(f"[LoraLinear] Cannot apply LoRA: k={k} <= 0 for layer {layer}")
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

        # # Initialize weights
        # w11, w12, b = self.make_layer()
        # self.loraw11.weight.data = w11.clone()
        # self.loraw12.weight.data = w12.clone()

        # if b is not None:
        #     self.loraw12.bias.data = b.clone()

        
        # del self.w
        # del self.b

    def make_layer(self):
        U, S, Vt = svds(self.w, k=self.k)
        Uk = U[:,-self.k:].copy()
        Sk = S[-self.k:].copy()
        Vtk = Vt[-self.k:,:].copy()

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


from typing import Optional, Tuple, Union
from pathlib import Path
import warnings

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from alphagenome_pytorch.config import DtypePolicy
from alphagenome_pytorch.named_outputs import NamedOutputs, TrackMetadataCatalog
from alphagenome_pytorch.utils.splicing import generate_splice_site_positions
import torch
import torch.nn as nn
import torch.nn.functional as F
#Literal
from typing import Literal
import math



_SOFT_CLIP_VALUE = 10.0
TRUNK_DIM = 1536
EMBEDDING_128BP_DIM = 3072
DECODER_DIM = 768
PAIR_EMBEDDING_DIM = 128
CONTACT_MAPS_OUTPUT_TRACKS = 28
NUM_SPLICE_TISSUES = 367
SPLICE_USAGE_OUTPUT_TRACKS = NUM_SPLICE_TISSUES * 2


def predictions_scaling(
    x: torch.Tensor,
    track_means: torch.Tensor,
    resolution: int,
    apply_squashing: bool,
    soft_clip_value: float = _SOFT_CLIP_VALUE,
    channels_last: bool = True,
) -> torch.Tensor:
    """Scales predictions to experimental data scale.

    Matches JAX: alphagenome_research.model.heads.predictions_scaling

    Args:
        x: Model predictions - NLC (B, S, C) if channels_last else NCL (B, C, S)
        track_means: Mean values per track (B, C)
        resolution: Bin resolution (1 or 128)
        apply_squashing: Whether to apply power law expansion (for RNA-seq)
        soft_clip_value: Value for soft clipping
        channels_last: If True, x is NLC. If False, x is NCL.

    Returns:
        Scaled predictions in experimental data space (same format as input)
    """
    # Soft clip: where x > soft_clip_value, apply quadratic expansion
    x = torch.where(
        x > soft_clip_value,
        (x + soft_clip_value) ** 2 / (4 * soft_clip_value),
        x,
    )

    # Apply squashing inverse (power law expansion) for RNA-seq type heads
    if apply_squashing:
        x = torch.pow(x, 1.0 / 0.75)

    # Scale by track means and resolution
    if channels_last:
        # NLC: track_means (B, C) → (B, 1, C)
        x = x * (track_means[:, None, :] * resolution)
    else:
        # NCL: track_means (B, C) → (B, C, 1)
        x = x * (track_means[:, :, None] * resolution)

    return x


def targets_scaling(
    x: torch.Tensor,
    track_means: torch.Tensor,
    resolution: int,
    apply_squashing: bool,
    soft_clip_value: float = _SOFT_CLIP_VALUE,
    channels_last: bool = True,
) -> torch.Tensor:
    """Scales targets from experimental data to model prediction space.

    Inverse of predictions_scaling. Used to scale targets before loss computation.
    Matches JAX: alphagenome_research.model.heads.targets_scaling

    Args:
        x: Targets in experimental space - NLC (B, S, C) if channels_last else NCL (B, C, S)
        track_means: Per-track scaling factors (B, C)
        resolution: Resolution multiplier (1 or 128)
        apply_squashing: Apply power law compression (only for RNA-seq)
        soft_clip_value: Value for soft clipping
        channels_last: If True, x is NLC. If False, x is NCL.

    Returns:
        Targets in model space (same format as input)
    """
    # Step 1: Normalize by track means and resolution
    if channels_last:
        # NLC: track_means (B, C) → (B, 1, C)
        x = x / (track_means[:, None, :] * resolution + 1e-8)
    else:
        # NCL: track_means (B, C) → (B, C, 1)
        x = x / (track_means[:, :, None] * resolution + 1e-8)

    # Step 2: Apply power law compression (RNA-seq only)
    if apply_squashing:
        x = torch.pow(x, 0.75)

    # Step 3: Soft clipping (inverse of quadratic expansion)
    x = torch.where(
        x > soft_clip_value,
        2.0 * torch.sqrt(x * soft_clip_value) - soft_clip_value,
        x,
    )

    return x


class MultiOrganismLinear(nn.Module): # NEED TO IMPLEMENT NON STANDARD LINEAR WITH K FOR LORA
    """Linear layer with organism-specific weights. Expects NLC format (B, S, C).

    Used for non-sequence operations like ContactMapsHead on pair activations.
    JAX: alphagenome_research.model.heads._MultiOrganismLinear
    """
    def __init__(
        self,
        k, 
        in_features,
        out_features,
        num_organisms=2,
        init_scheme: Literal['truncated_normal', 'uniform'] = 'truncated_normal',
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_organisms = num_organisms
        self._init_scheme = init_scheme

        # We store weights as (num_organisms, in_features, out_features)
        # Note: PyTorch nn.Linear stores (out_features, in_features).
        # But here we are doing custom einsum logic anyway.
        # Let's stick to JAX shape for easier mapping, then transpose if needed for efficiency.
        # JAX shape: (num_organisms, in, out).
        self.weight = nn.Parameter(torch.empty(num_organisms, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(num_organisms, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.in_features)

        if self._init_scheme == 'truncated_normal':
            # Match JAX: TruncatedNormal for weights, zeros for bias
            nn.init.trunc_normal_(self.weight, std=stdv)
            nn.init.zeros_(self.bias)
        else:  # 'uniform'
            # Legacy PyTorch-style uniform initialization
            self.weight.data.uniform_(-stdv, stdv)
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, organism_index):
        # x: (B, S, in_features) - NLC format
        w = self.weight[organism_index]  # (B, In, Out)
        b = self.bias[organism_index]    # (B, Out)

        input_dtype = x.dtype
        out = torch.bmm(x.float(), w.float()).to(input_dtype)
        return out + b.unsqueeze(1)


class MultiOrganismConv1d(nn.Module): # no linear
    """Organism-specific 1x1 conv for NCL format (B, C, S).

    Equivalent to JAX _MultiOrganismLinear which operates on NLC format.
    Using Conv1d avoids transpose overhead when data is already NCL.
    """

    def __init__(self, in_channels, out_channels, num_organisms=2,
                 init_scheme: Literal['truncated_normal', 'uniform'] = 'truncated_normal'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_organisms = num_organisms
        self._init_scheme = init_scheme

        # Weight: (num_organisms, out_channels, in_channels) - Conv1d convention
        self.weight = nn.Parameter(torch.empty(num_organisms, out_channels, in_channels))
        self.bias = nn.Parameter(torch.empty(num_organisms, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.in_channels)

        if self._init_scheme == 'truncated_normal':
            nn.init.trunc_normal_(self.weight, std=stdv)
            nn.init.zeros_(self.bias)
        else:  # 'uniform'
            self.weight.data.uniform_(-stdv, stdv)
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, organism_index):
        # x: (B, C_in, S) - NCL format
        w = self.weight[organism_index]  # (B, C_out, C_in)
        b = self.bias[organism_index]    # (B, C_out)

        # Batched 1x1 conv via einsum: (B, C_in, S) @ (B, C_out, C_in).T -> (B, C_out, S)
        input_dtype = x.dtype
        out = torch.einsum('bcs,boc->bos', x.float(), w.float()).to(input_dtype)
        return out + b.unsqueeze(2)  # bias: (B, C_out, 1)

class GenomeTracksHead(nn.Module): # no linear
    """Predicts genome tracks at multiple resolutions.

    Internal computation is NCL for Conv1d efficiency.
    Outputs NLC format (B, S, T) to match JAX reference.

    Matches JAX: alphagenome_research.model.heads.GenomeTracksHead
    """

    def __init__(
        self,
        in_channels,
        num_tracks,
        resolutions=(1, 128),
        num_organisms=2,
        apply_squashing=False,
        track_means=None,
        init_scheme: Literal['truncated_normal', 'uniform'] = 'truncated_normal',
    ):
        super().__init__()
        self.num_tracks = num_tracks
        self.resolutions = sorted(resolutions)
        self.num_organisms = num_organisms
        self.apply_squashing = apply_squashing

        # in_channels controls input dim per requested resolution:
        # - None: default AlphaGenome dims (1bp->TRUNK_DIM, 128bp->EMBEDDING_128BP_DIM)
        # - int: same dim for all requested resolutions (e.g., encoder_only=TRUNK_DIM)
        # - dict[int, int]: explicit per-resolution override
        # - tuple/list: positional dims aligned with sorted resolutions
        if in_channels is None:
            resolved_in_channels = {1: TRUNK_DIM, 128: EMBEDDING_128BP_DIM}
        elif isinstance(in_channels, int):
            resolved_in_channels = {res: in_channels for res in self.resolutions}
        elif isinstance(in_channels, dict):
            missing_resolutions = [res for res in self.resolutions if res not in in_channels]
            if missing_resolutions:
                raise ValueError(
                    f"in_channels dict missing resolutions: {missing_resolutions}; "
                    f"required resolutions={self.resolutions}"
                )
            resolved_in_channels = {
                res: int(in_channels[res]) for res in self.resolutions
            }
        elif isinstance(in_channels, (tuple, list)):
            if len(in_channels) != len(self.resolutions):
                raise ValueError(
                    f"in_channels tuple/list length ({len(in_channels)}) must match "
                    f"resolutions length ({len(self.resolutions)})"
                )
            resolved_in_channels = {
                res: int(ch) for res, ch in zip(self.resolutions, in_channels)
            }
        else:
            raise TypeError(
                "in_channels must be None, int, dict[int, int], tuple[int, ...], "
                f"or list[int], got {type(in_channels).__name__}"
            )

        # Track means: (num_organisms, num_tracks)
        # Replace NaN with 0 to prevent gradient poisoning.
        if track_means is not None:
            sanitized_means = torch.nan_to_num(track_means, nan=0.0)
            self.register_buffer('track_means', sanitized_means)
        else:
            self.register_buffer('track_means', torch.ones(num_organisms, num_tracks))

        self.convs = nn.ModuleDict()
        self.residual_scales = nn.ParameterDict()

        for res in self.resolutions:
            res_str = str(res)
            dim = resolved_in_channels[res]

            self.convs[res_str] = MultiOrganismConv1d(dim, num_tracks, num_organisms, init_scheme=init_scheme)

            # learnt_scale: (num_organisms, num_tracks)
            self.residual_scales[res_str] = nn.Parameter(torch.ones(num_organisms, num_tracks))

    def _predict(self, x, organism_index, res_str):
        """Raw model prediction (in model space)."""
        # x: (B, C, S) - NCL format
        x = self.convs[res_str](x, organism_index)  # (B, T, S)

        # Residual Scale: (B, T) → (B, T, 1) for NCL broadcast
        scale = self.residual_scales[res_str][organism_index]

        # Softplus: softplus(x) * softplus(scale)
        x = F.softplus(x) * F.softplus(scale.unsqueeze(2))

        return x

    def unscale(self, x, organism_index, resolution, channels_last=True):
        """Unscales predictions to experimental data scale."""
        track_means = self.track_means[organism_index]  # (B, num_tracks)
        return predictions_scaling(
            x,
            track_means=track_means,
            resolution=resolution,
            apply_squashing=self.apply_squashing,
            channels_last=channels_last,
        )

    def scale(self, x, organism_index, resolution, channels_last=True):
        """Scales targets from experimental to model prediction space.

        Args:
            x: Targets in experimental space.
               NLC (B, S, T) if channels_last else NCL (B, T, S)
            organism_index: Organism indices (B,)
            resolution: Resolution (1 or 128)
            channels_last: If True, x is NLC. If False, x is NCL.

        Returns:
            Targets in model space (same format as input)
        """
        track_means = self.track_means[organism_index]  # (B, num_tracks)
        return targets_scaling(
            x,
            track_means=track_means,
            resolution=resolution,
            apply_squashing=self.apply_squashing,
            channels_last=channels_last,
        )

    def forward(self, embeddings_dict, organism_index, return_scaled=False, channels_last=True):
        """Returns predictions in experimental or model scale.

        Args:
            embeddings_dict: Dict mapping resolution to embeddings (B, C, S) NCL
            organism_index: Organism indices (B,)
            return_scaled: If True, return model space (for loss).
                           If False, return experimental space (for inference).
            channels_last: Output format.
                - True (default): NLC format (B, S, T) - user-friendly, matches JAX
                - False: NCL format (B, T, S) - for training efficiency (0 transposes)

        Returns:
            Dict mapping resolution to predictions in specified format
        """
        outputs = {}
        for res in self.resolutions:
            if res not in embeddings_dict:
                continue
            res_str = str(res)
            emb = embeddings_dict[res]  # (B, C, S) NCL

            # Get raw predictions (model space) - NCL internally
            scaled_pred = self._predict(emb, organism_index, res_str)  # (B, T, S) NCL

            # Transpose to NLC if channels_last
            if channels_last:
                scaled_pred = scaled_pred.transpose(1, 2)  # (B, S, T)

            if return_scaled:
                outputs[res] = scaled_pred
            else:
                outputs[res] = self.unscale(scaled_pred, organism_index, res, channels_last)

        return outputs


class ContactMapsHead(nn.Module): # has k for lora
    """
    Predicts contact maps from pairwise embeddings.
    JAX: alphagenome_research.model.heads.ContactMapsHead
    """
    def __init__(
        self, 
        k, 
        in_features=PAIR_EMBEDDING_DIM,
        num_tracks=CONTACT_MAPS_OUTPUT_TRACKS,
        num_organisms=2,
    ):
        super().__init__()
        self.num_tracks = num_tracks
        self.num_organisms = num_organisms
        self.linear = MultiOrganismLinear(k, in_features, num_tracks, num_organisms)

    def forward(self, pair_embeddings, organism_index, channels_last=True):
        # pair_embeddings: (B, S, S, D) where D=128
        # organism_index: (B,)
        B, S1, S2, D = pair_embeddings.shape

        # Reshape for MultiOrganismLinear: (B, S*S, D)
        x = pair_embeddings.view(B, S1 * S2, D)

        # Apply linear: (B, S*S, num_tracks)
        x = self.linear(x, organism_index)

        # Reshape back: (B, S, S, num_tracks)
        x = x.view(B, S1, S2, self.num_tracks)

        if not channels_last:
            x = x.permute(0, 3, 1, 2).contiguous()  # (B, T, S, S)

        return x

class SpliceSitesClassificationHead(nn.Module): # no linear
    """Predicts splice site classification.

    Internal computation is NCL for Conv1d efficiency.
    Outputs NLC format (B, S, 5) to match JAX reference.

    Classes: Donor+, Acceptor+, Donor-, Acceptor-, Not a splice site
    JAX: alphagenome_research.model.heads.SpliceSitesClassificationHead
    """

    def __init__(self, in_channels=TRUNK_DIM, num_organisms=2):
        super().__init__()
        self.num_organisms = num_organisms
        self.conv = MultiOrganismConv1d(
            in_channels=in_channels,
            out_channels=5,  # 5 classes
            num_organisms=num_organisms
        )

    def forward(self, embeddings_1bp, organism_index, channels_last=True):
        # embeddings_1bp: (B, C, S) - NCL format (internal)
        logits_ncl = self.conv(embeddings_1bp, organism_index)  # (B, 5, S)

        if channels_last:
            # Transpose to NLC: (B, 5, S) -> (B, S, 5)
            logits = logits_ncl.transpose(1, 2)
            # Softmax over classes (dim=-1 in NLC)
            probs = F.softmax(logits, dim=-1)
        else:
            logits = logits_ncl
            # Softmax over classes (dim=1 in NCL)
            probs = F.softmax(logits, dim=1)

        return {
            "logits": logits,
            "probs": probs,
        }

class SpliceSitesUsageHead(nn.Module): # no linear
    """Predicts splice site usage.

    Internal computation is NCL for Conv1d efficiency.
    Outputs NLC format (B, S, T) to match JAX reference.

    Outputs proportion of RNA using each splice site.
    JAX: alphagenome_research.model.heads.SpliceSitesUsageHead
    """

    def __init__(
        self,
        in_channels=TRUNK_DIM,
        num_output_tracks=SPLICE_USAGE_OUTPUT_TRACKS,
        num_organisms=2,
        num_tracks_per_organism=None,
    ):
        super().__init__()
        self.num_organisms = num_organisms
        self.num_output_tracks = num_output_tracks

        # keep a fixed output width and use per-organism masks to
        # ignore padded channels in loss/metrics.
        if num_tracks_per_organism is None:
            num_tracks_per_organism = [num_output_tracks] * num_organisms
        if len(num_tracks_per_organism) != num_organisms:
            raise ValueError(
                f"num_tracks_per_organism length ({len(num_tracks_per_organism)}) "
                f"must equal num_organisms ({num_organisms})"
            )

        for org_idx, tracks in enumerate(num_tracks_per_organism):
            if tracks < 0 or tracks > num_output_tracks:
                raise ValueError(
                    f"num_tracks_per_organism[{org_idx}]={tracks} must be in "
                    f"[0, {num_output_tracks}]"
                )

        # Computed on-the-fly from config, not learned - exclude from state_dict
        track_mask = torch.arange(num_output_tracks)[None, :] < torch.tensor(
            list(num_tracks_per_organism),
            dtype=torch.long,
        )[:, None]
        self.register_buffer('track_mask', track_mask, persistent=False)

        self.conv = MultiOrganismConv1d(
            in_channels=in_channels,
            out_channels=num_output_tracks,  # NUM_SPLICE_TISSUES * 2 strands
            num_organisms=num_organisms
        )

    def forward(self, embeddings_1bp, organism_index, channels_last=True):
        # embeddings_1bp: (B, C, S) - NCL format (internal)
        logits_ncl = self.conv(embeddings_1bp, organism_index)  # (B, T, S)

        if channels_last:
            # Transpose to NLC: (B, T, S) -> (B, S, T)
            logits = logits_ncl.transpose(1, 2)
            mask = self.track_mask[organism_index][:, None, :]
        else:
            logits = logits_ncl
            mask = self.track_mask[organism_index][:, :, None]

        predictions = torch.sigmoid(logits)

        return {
            "logits": logits,
            "predictions": predictions,
            "track_mask": mask,
        }

class SpliceSitesJunctionHead(nn.Module): # no linear
    """Predicts splice junction read counts. Expects NCL format (B, C, S).

    JAX: alphagenome_research.model.heads.SpliceSitesJunctionHead
    """
    def __init__(
        self,
        in_channels=TRUNK_DIM,
        hidden_dim=DECODER_DIM,
        num_tissues=NUM_SPLICE_TISSUES,
        num_organisms=2,
        num_tracks_per_organism=None,
    ):
        super().__init__()
        self._num_organisms = num_organisms
        self._num_tissues = num_tissues
        self._in_channels = in_channels
        self._hidden_dim = hidden_dim
        self._max_position_encoding_distance = int(2**20)

        # Precompute tissue mask per organism (matches JAX get_multi_organism_track_mask).
        # If num_tracks_per_organism is not specified, all organisms use num_tissues (no masking).
        if num_tracks_per_organism is None:
            num_tracks_per_organism = [num_tissues] * num_organisms
        if len(num_tracks_per_organism) != num_organisms:
            raise ValueError(
                f"num_tracks_per_organism length ({len(num_tracks_per_organism)}) "
                f"must equal num_organisms ({num_organisms})"
            )
        for org_idx, tracks in enumerate(num_tracks_per_organism):
            if tracks < 0 or tracks > num_tissues:
                raise ValueError(
                    f"num_tracks_per_organism[{org_idx}]={tracks} must be in "
                    f"[0, {num_tissues}]"
                )

        # Computed on-the-fly from config, not learned - exclude from state_dict
        tissue_mask = torch.arange(num_tissues)[None, :] < torch.tensor(
            list(num_tracks_per_organism),
            dtype=torch.long,
        )[:, None]
        self.register_buffer('tissue_mask', tissue_mask, persistent=False)

        self.conv = MultiOrganismConv1d(
            in_channels=self._in_channels,
            out_channels=self._hidden_dim,
            num_organisms=self._num_organisms
        )

        def make_rope_params():
            return nn.Parameter(torch.zeros(
                self._num_organisms, 2, self._num_tissues, self._hidden_dim
            ))

        self.rope_params = nn.ParameterDict({
            "pos_donor": make_rope_params(),
            "pos_acceptor": make_rope_params(),
            "neg_donor": make_rope_params(),
            "neg_acceptor": make_rope_params(),
        })

    def forward(self, embeddings_1bp, organism_index, channels_last=True, **kwargs):
        """
        Args:
            embeddings_1bp: (B, C, S) - NCL format
            organism_index: (B,)
            splice_site_positions: (B, 4, P) - required kwarg

        Returns:
            Dict with pred_counts (B, P, P, 2*T), positions, mask
        """
        splice_site_positions = kwargs.get("splice_site_positions", None)
        if splice_site_positions is None:
            raise ValueError("splice_site_positions is required")

        def _predict(embeddings_1bp, splice_site_positions, organism_index):
            # embeddings_1bp: (B, C, S), splice_site_positions: (B, 4, P)
            assert splice_site_positions.shape[1] == 4
            pos_donor_idx = splice_site_positions[:, 0, :]
            pos_acceptor_idx = splice_site_positions[:, 1, :]
            neg_donor_idx = splice_site_positions[:, 2, :]
            neg_acceptor_idx = splice_site_positions[:, 3, :]

            # Project: (B, C, S) → (B, H, S)
            splice_site_logits = self.conv(embeddings_1bp, organism_index)

            def _index_embeddings(embedding, indices):
                """Select embeddings at positions. embedding: (B, H, S), indices: (B, P)"""
                B, H, S = embedding.shape
                batch_idx = torch.arange(B, device=embedding.device).unsqueeze(1)
                # Index along S dimension: embedding[b, :, indices[b, p]] → (B, P, H)
                # PyTorch advanced indexing: broadcast indices give leading dims, : gives trailing
                return embedding[batch_idx, :, indices]  # (B, P, H)

            def _apply_rope(embedding, indices, params, organism_index):
                x = _index_embeddings(embedding, indices)  # (B, P, H)
                batch_params = params[organism_index]  # (B, 2, T, H)
                scale = batch_params[:, [0], :, :]
                offset = batch_params[:, [1], :, :]
                x = scale * x[:, :, None, :] + offset  # (B, P, T, H)
                return apply_rope(
                    x, indices,
                    max_position=self._max_position_encoding_distance,
                    inplace=True,
                )

            pos_donor_logits = _apply_rope(
                splice_site_logits, pos_donor_idx,
                self.rope_params["pos_donor"], organism_index
            )
            pos_acceptor_logits = _apply_rope(
                splice_site_logits, pos_acceptor_idx,
                self.rope_params["pos_acceptor"], organism_index
            )
            neg_donor_logits = _apply_rope(
                splice_site_logits, neg_donor_idx,
                self.rope_params["neg_donor"], organism_index
            )
            neg_acceptor_logits = _apply_rope(
                splice_site_logits, neg_acceptor_idx,
                self.rope_params["neg_acceptor"], organism_index
            )

            pos_counts = F.softplus(torch.einsum(
                "bdth,bath->bdat", pos_donor_logits, pos_acceptor_logits
            ))
            neg_counts = F.softplus(torch.einsum(
                "bdth,bath->bdat", neg_donor_logits, neg_acceptor_logits
            ))

            pos_mask = torch.einsum("bd,ba->bda", pos_donor_idx >= 0, pos_acceptor_idx >= 0)
            neg_mask = torch.einsum("bd,ba->bda", neg_donor_idx >= 0, neg_acceptor_idx >= 0)

            tissue_mask = self.tissue_mask[organism_index]

            pos_mask = pos_mask[:, :, :, None] * tissue_mask[:, None, None, :]
            neg_mask = neg_mask[:, :, :, None] * tissue_mask[:, None, None, :]

            splice_junction_mask = torch.cat([pos_mask, neg_mask], dim=-1)
            pred_counts = torch.cat([pos_counts, neg_counts], dim=-1)
            pred_counts = torch.where(splice_junction_mask, pred_counts, 0.0)

            return pred_counts, splice_junction_mask

        pred_counts, splice_junction_mask = _predict(
            embeddings_1bp, splice_site_positions, organism_index
        )
        return {
            "pred_counts": pred_counts,
            "splice_site_positions": splice_site_positions,
            "splice_junction_mask": splice_junction_mask,
        }

class OutputEmbedder(nn.Module): # no linear
    """Output embedder using Conv1d for NCL format (B, C, S).

    Matches JAX `alphagenome_research.model.embeddings.OutputEmbedder`.

    Logic:
    1. Conv1d projection to output channels.
    2. Optional skip connection addition (with projection if needed).
    3. Add Organism Embedding.
    4. Norm + GELU.
    """

    def __init__(self, in_channels, out_channels, num_organisms=2):
        super().__init__()
        self.num_organisms = num_organisms
        self.out_channels = out_channels

        # Use Conv1d(k=1) instead of Linear - same math, native NCL
        # For 128bp: Input 1536 -> Output 3072
        # For 1bp: Input 768 -> Output 1536
        self.project_in = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        # Skip projection - set externally if needed (e.g., for 1bp embedder)
        self.project_skip = None

        self.organism_embed = nn.Embedding(num_organisms, out_channels)
        self.norm = RMSBatchNorm(channels=out_channels)

    def forward(self, x, organism_index, skip_x=None, channels_last=False):
        # x: (B, C, S) - NCL format

        # Project main input
        x_proj = self.project_in(x)

        if skip_x is not None and self.project_skip is not None:
            # skip_x: (B, C_skip, S_skip)
            s_proj = self.project_skip(skip_x)

            # Upsample sequence if needed (dim 2 in NCL)
            repeat_factor = x_proj.shape[2] // s_proj.shape[2]
            if repeat_factor > 1:
                s_proj = s_proj.repeat_interleave(repeat_factor, dim=2)

            x_proj = x_proj + s_proj

        # Apply norm
        out = self.norm(x_proj)

        # Add organism embedding: (B, C) → (B, C, 1) for NCL broadcast
        emb = self.organism_embed(organism_index).unsqueeze(2)
        out = out + emb
        
        out = gelu(out)

        if channels_last:
            # (B, C, S) -> (B, S, C)
            out = out.transpose(1, 2)
            
        return out

class OutputPair(nn.Module): # no linear
    """Output embedder for pair activations (B, S, S, D).

    Note: Pair activations use a different format than sequence data.
    LayerNorm operates over the last dimension (features).
    """

    def __init__(self, dim=128, num_organisms=2):
        super().__init__()
        self.num_organisms = num_organisms
        self.organism_embed = nn.Embedding(num_organisms, dim)
        self.norm = LayerNorm(normalized_shape=dim, rms_norm=True)

    def forward(self, x, organism_index):
        # x: (B, S, S, D) - pair activations
        # Symmetrize
        x = (x + x.transpose(1, 2)) / 2.0

        # Apply norm, then add organism embedding, then gelu
        x = self.norm(x)

        emb = self.organism_embed(organism_index)  # (B, D)
        x = x + emb[:, None, None, :]

        return gelu(x)
_MAX_RELATIVE_DISTANCE = 8192


def _apply_rope_inplace(x, cos_theta, sin_theta):
    """Memory-efficient in-place RoPE application.

    Only allocates 0.5x extra memory (vs 2x in standard implementation).
    Modifies x in-place and returns it.

    Args:
        x: Input tensor (B, S, H, C)
        cos_theta: Cosine of rotation angles (B, S, 1, C)
        sin_theta: Sine of rotation angles (B, S, 1, C)

    Returns:
        x modified in-place with RoPE applied
    """
    # Clone even positions before overwriting (0.5x memory overhead)
    x_even = x[..., ::2].clone()

    # Compute and write new even values in-place
    # RoPE formula for even indices: x_even * cos - x_odd * sin
    x[..., ::2] = x_even * cos_theta[..., ::2] - x[..., 1::2] * sin_theta[..., ::2]

    # Compute and write new odd values in-place (uses saved x_even)
    # RoPE formula for odd indices: x_even * sin + x_odd * cos
    x[..., 1::2] = x_even * sin_theta[..., 1::2] + x[..., 1::2] * cos_theta[..., 1::2]

    return x


def apply_rope(x, positions=None, max_position=_MAX_RELATIVE_DISTANCE, inplace=False):
    """Applies Rotary Position Embeddings to the input tensor.

    Matches JAX: alphagenome_research.model.attention.apply_rope

    All computations use the input dtype (x.dtype), matching JAX behavior.
    When using DtypePolicy.mixed_precision(), this means RoPE computes in bfloat16.
    When using DtypePolicy.full_float32(), this means RoPE computes in float32.

    Args:
        x: Input tensor (B, S, H, C)
        positions: Optional position indices (B, S)
        max_position: Maximum position for frequency calculation
        inplace: If True, use memory-efficient in-place implementation.
                 Reduces memory overhead from ~2x to ~0.5x but modifies x in-place.

    Returns:
        Tensor with RoPE applied. If inplace=True, returns the same tensor (modified).
    """
    # x: (B, S, H, C)
    B, S, H, C = x.shape
    compute_dtype = x.dtype  # Match JAX: use input dtype for all RoPE ops

    if positions is None:
        positions = torch.arange(S, device=x.device, dtype=compute_dtype).unsqueeze(0)  # (1, S)
    elif positions.dtype != compute_dtype:
        positions = positions.to(compute_dtype)

    num_freq = C // 2
    # JAX geomspace equivalent: geomspace(1, max_position - num_freq + 1, num_freq)
    base_freqs = torch.logspace(
        math.log10(1), math.log10(max_position - num_freq + 1),
        steps=num_freq, base=10, device=x.device, dtype=compute_dtype
    )
    denom = torch.arange(num_freq, device=x.device, dtype=compute_dtype) + base_freqs
    inv_freq = 1.0 / denom

    theta = torch.einsum('bs,f->bsf', positions, inv_freq)
    theta = torch.repeat_interleave(theta, 2, dim=-1).unsqueeze(2)  # (B, S, 1, C)

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    if inplace:
        return _apply_rope_inplace(x, cos_theta, sin_theta)
    else:
        x_rotated = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(start_dim=-2)
        return x * cos_theta + x_rotated * sin_theta

def _shift(x, query_length, key_length):
    """Shifts the diagonal of a 2D array, PyTorch equivalent."""
    # x: (..., query_length, query_length + key_length)
    shape = x.shape
    batch_shape = shape[:-2]
    n_rows = shape[-2]
    n_diags = shape[-1]
    
    # Reshape to (..., n_diags, n_rows)
    x = x.view(*batch_shape, n_diags, n_rows)
    # Drop first row (originally first diag)
    x = x[..., 1:, :] 
    # Reshape back to (..., n_rows, n_diags - 1)
    x = x.view(*batch_shape, n_rows, n_diags - 1)
    # Return first key_length columns
    return x[..., :key_length]

def _central_mask_features(distances, feature_size, seq_length):
    """Positional features using exponentially-spaced central mask.

    Matches JAX: alphagenome_research.model.attention._central_mask_features

    JAX formula:
        center_widths = jnp.arange(feature_size) + jnp.geomspace(
            1, seq_length - feature_size + 1, feature_size, endpoint=False
        )
    """
    device = distances.device
    dtype = torch.float32

    # Compute geomspace(1, seq_length - feature_size + 1, feature_size, endpoint=False)
    # geomspace with endpoint=False: values[i] = start * (end/start)^(i/n)
    start = 1.0
    end = float(seq_length - feature_size + 1)

    log_start = math.log(start)
    log_end = math.log(end)
    log_step = (log_end - log_start) / feature_size  # endpoint=False

    exponents = torch.arange(feature_size, device=device, dtype=dtype) * log_step
    geomspace_values = torch.exp(torch.tensor(log_start, device=device, dtype=dtype) + exponents)

    # JAX: center_widths = jnp.arange(feature_size) + jnp.geomspace(...)
    center_widths = torch.arange(feature_size, device=device, dtype=dtype) + geomspace_values

    # center_widths: (feature_size,)
    # distances: (...)
    # Output: (..., feature_size)
    return (center_widths > distances.unsqueeze(-1)).to(dtype)

class MHABlock(nn.Module): # has k for lora
    """Multi-Head Attention block.

    Matches JAX: alphagenome_research.model.attention.MHABlock

    JAX uses precision=BF16_BF16_F32 for attention, meaning:
    - Inputs in bfloat16
    - Accumulation in float32
    - Output cast back to input dtype
    """
    def __init__(self, k, d_model):
        super().__init__()
        self.norm = RMSBatchNorm(d_model, channels_last=True)
        self.q_proj = LoraLinear(k, d_model, 8 * 128, bias=False)
        self.norm_q = LayerNorm(128)
        self.k_proj = LoraLinear(k, d_model, 128, bias=False)
        self.norm_k = LayerNorm(128)
        self.v_proj = LoraLinear(k, d_model, 192, bias=False)
        self.norm_v = LayerNorm(192)
        self.final_norm = RMSBatchNorm(d_model, channels_last=True)
        self.linear_embedding = LoraLinear(k, 8 * 192, d_model)

    def forward(self, x, attention_bias, compute_dtype=None):
        B, S, D = x.shape
        if compute_dtype is None:
            compute_dtype = x.dtype

        # Cast to compute dtype
        x = x.to(compute_dtype)

        h = self.norm(x)

        q = self.norm_q(self.q_proj(h).view(B, S, 8, 128))
        k = self.norm_k(self.k_proj(h).view(B, S, 1, 128))
        v = self.norm_v(self.v_proj(h).view(B, S, 1, 192))

        q = apply_rope(q, inplace=True)
        k = apply_rope(k, inplace=True)

        q_t = q.permute(0, 2, 1, 3)  # (B, 8, S, C)
        k_t = k.permute(0, 2, 1, 3)  # (B, 1, S, C)

        # Attention logits: bf16 matmul then cast to f32 (matches JAX BF16_BF16_F32)
        # JAX uses precision=BF16_BF16_F32: bf16 inputs, f32 accumulation, f32 output
        att = torch.matmul(q_t, k_t.transpose(-2, -1)).float()  # (B, 8, S, S)
        att = att / math.sqrt(128.0)

        if attention_bias is not None:
            att = att + attention_bias.float()

        logits_soft_cap = 5.0
        att = torch.tanh(att / logits_soft_cap) * logits_soft_cap

        attn_weights = F.softmax(att, dim=-1)

        # Value projection: bf16 matmul then cast back to compute dtype
        v_t = v.permute(0, 2, 1, 3)
        y = torch.matmul(attn_weights.to(compute_dtype), v_t).float()  # (B, 8, S, 192)
        y = y.to(compute_dtype)
        y = y.permute(0, 2, 1, 3).reshape(B, S, -1)

        y = self.linear_embedding(y)
        return self.final_norm(y)

class MLPBlock(nn.Module): # has k for lora
    def __init__(self, k, d_model):
        super().__init__()
        self.norm = RMSBatchNorm(d_model, channels_last=True)
        self.fc1 = LoraLinear(k, d_model, d_model * 2)
        self.fc2 = LoraLinear(k, d_model * 2, d_model)
        self.final_norm = RMSBatchNorm(d_model, channels_last=True)

    def forward(self, x):
        h = self.norm(x)
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        return self.final_norm(h)

class AttentionBiasBlock(nn.Module): # has k for lora
    def __init__(self, k, pair_dim):
        super().__init__()
        self.norm = RMSBatchNorm(pair_dim, channels_last=True)
        self.proj = LoraLinear(k, pair_dim, 8, bias=False)

    def forward(self, x):
        # x: (B, s, s, D)
        h = F.gelu(self.norm(x))
        h = self.proj(h) # (B, s, s, 8)
        # Repeat 16x16
        h = torch.repeat_interleave(h, 16, dim=1)
        h = torch.repeat_interleave(h, 16, dim=2)
        return h.permute(0, 3, 1, 2) # (B, 8, S, S)

class SequenceToPairBlock(nn.Module): # has k for lora
    def __init__(self, k, d_model, pair_dim=128):
        super().__init__()
        self.d_model = d_model
        
        # 32 heads * 128 dim = 4096 params for q/k internal?
        # JAX uses hardcoded 32*128.
        self.num_heads = 32
        self.head_dim = 128
        
        self.pool = Pool1d(kernel_size=16, stride=16, method='mean')
        self.norm_seq2pair = LayerNorm(d_model, rms_norm=True)
        
        self.linear_q = LoraLinear(k, d_model, self.num_heads * self.head_dim, bias=False)
        self.linear_k = LoraLinear(k, d_model, self.num_heads * self.head_dim, bias=False)
        
        # Relative positions features -> 2*32 -> ...
        self.linear_pos_features = LoraLinear(k, 2 * self.num_heads, self.num_heads * self.head_dim)
        
        self.q_r_bias = nn.Parameter(torch.zeros(1, 1, self.num_heads, self.head_dim))
        self.k_r_bias = nn.Parameter(torch.zeros(1, 1, self.num_heads, self.head_dim))
        
        self.linear_y_q = LoraLinear(k, d_model, self.head_dim, bias=False)
        self.linear_y_k = LoraLinear(k, d_model, self.head_dim, bias=False)
        
        self.linear_pair = LoraLinear(k, self.num_heads, self.head_dim) 

    def forward(self, x):
        # x: (B, S, D) - NLC format
        # Pool1d expects NCL, so transpose around pool call
        x_pooled = self.pool(x.transpose(1, 2)).transpose(1, 2)
        x_norm = self.norm_seq2pair(x_pooled)
        
        B, S_prime, _ = x_norm.shape
        
        q = self.linear_q(x_norm).view(B, S_prime, self.num_heads, self.head_dim)
        k = self.linear_k(x_norm).view(B, S_prime, self.num_heads, self.head_dim)
        
        # Relative positions (computed in float32 for precision, then cast to model dtype)
        range_vec = torch.arange(-S_prime, S_prime, device=x.device, dtype=torch.float32)
        pos_feat = _central_mask_features(torch.abs(range_vec), self.num_heads, _MAX_RELATIVE_DISTANCE // 16)
        sign = torch.sign(range_vec).unsqueeze(-1)
        pos_feat = torch.cat([pos_feat, sign * pos_feat], dim=-1) # (2S', 64)
        pos_feat = pos_feat.to(x.dtype)  # Match model dtype

        pos_encoding = self.linear_pos_features(pos_feat).view(2 * S_prime, self.num_heads, self.head_dim)
        
        term_q = torch.einsum('bqhc,phc->bhqp', q + self.q_r_bias, pos_encoding)
        term_k = torch.einsum('bkhc,phc->bhkp', k + self.k_r_bias, pos_encoding)
        
        rel_q_a = _shift(term_q, S_prime, S_prime)
        rel_k_a = _shift(term_k, S_prime, S_prime)
        
        rel_q_a = rel_q_a.permute(0, 2, 3, 1) # (B, S', S', H)
        rel_k_a = rel_k_a.permute(0, 3, 2, 1) # (B, S', S', H) from bhkp -> bpkh logic
        
        a = torch.einsum('bqhc,bkhc->bqkh', q, k) # (B, S', S', H)
        a = a + 0.5 * (rel_q_a + rel_k_a)
        
        # y branches
        x_gelu = F.gelu(x_norm)
        y_q = self.linear_y_q(x_gelu)
        y_k = self.linear_y_k(x_gelu)
        
        pair_act = self.linear_pair(a) + y_q.unsqueeze(2) + y_k.unsqueeze(1)
        return pair_act

class RowAttentionBlock(nn.Module): # has k for lora
    """Self-attention block applied along rows of pairwise representations.

    Matches JAX: alphagenome_research.model.attention.RowAttentionBlock

    JAX uses precision=BF16_BF16_F32 for einsum operations.
    """
    def __init__(self, k, pair_dim=128):
        super().__init__()
        self.norm = LayerNorm(pair_dim, rms_norm=True)
        self.linear_q = LoraLinear(k, pair_dim, pair_dim, bias=False)
        self.linear_k = LoraLinear(k, pair_dim, pair_dim, bias=False)
        self.linear_v = LoraLinear(k, pair_dim, pair_dim)

    def forward(self, x, compute_dtype=None):
        if compute_dtype is None:
            compute_dtype = x.dtype
        x = x.to(compute_dtype)

        h = self.norm(x)
        q = self.linear_q(h)
        k = self.linear_k(h)
        v = self.linear_v(h)

        # Attention: bf16 einsum then cast to f32 (matches JAX BF16_BF16_F32)
        scale = 1.0 / math.sqrt(128.0)
        attn = torch.einsum('bpqf,bpkf->bpqk', q, k).float() * scale
        attn = F.softmax(attn, dim=-1)

        # Value projection: bf16 einsum then cast back
        out = torch.einsum('bpqk,bpkf->bpqf', attn.to(compute_dtype), v).float()
        return out.to(compute_dtype)

class PairMLPBlock(nn.Module): # has k for lora
    def __init__(self, k, pair_dim=128):
        super().__init__()
        self.norm = LayerNorm(pair_dim, rms_norm=True)
        self.linear1 = LoraLinear(k, pair_dim, 2 * pair_dim)
        self.linear2 = LoraLinear(k, 2 * pair_dim, pair_dim)
        
    def forward(self, x):
        h = self.norm(x)
        h = self.linear1(h)
        h = F.relu(h)
        h = self.linear2(h)
        return h

class PairUpdateBlock(nn.Module): # has k for lora
    def __init__(self, k, d_model, pair_dim=128):
        super().__init__()
        self.seq2pair = SequenceToPairBlock(k, d_model, pair_dim)
        self.row_attn = RowAttentionBlock(k, pair_dim)
        self.pair_mlp = PairMLPBlock(k, pair_dim)

    def forward(self, x, pair_rep, compute_dtype=None):
        # x: (B, S, D)
        # pair_rep: (B, S/16, S/16, F)

        y = self.seq2pair(x)

        if pair_rep is None:
            pair_rep = y
        else:
            pair_rep = pair_rep + y

        pair_rep = pair_rep + self.row_attn(pair_rep, compute_dtype=compute_dtype)
        pair_rep = pair_rep + self.pair_mlp(pair_rep)

        return pair_rep


def gelu(x):
    """GELU using JAX's custom approximation: sigmoid(1.702 * x) * x

    Matches JAX: alphagenome_research.model.layers.gelu
    JAX explicitly converts coefficient to match input dtype.
    """
    coef = torch.tensor(1.702, dtype=x.dtype, device=x.device)
    return torch.sigmoid(coef * x) * x

class Pool1d(nn.Module): # no linear 
    """1D pooling with SAME padding. Expects NCL input (B, C, S).

    Matches JAX: alphagenome_research.model.layers.pool
    JAX uses padding='SAME' which pads input to ensure output_size = ceil(input_size / stride).
    """
    def __init__(self, kernel_size: int, stride: int = None, method: str = 'max'):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, S) - NCL format, no transpose needed
        input_size = x.shape[-1]
        output_size = (input_size + self.stride - 1) // self.stride  # ceil division
        pad_total = max((output_size - 1) * self.stride + self.kernel_size - input_size, 0)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        if pad_total > 0:
            x = F.pad(x, (pad_left, pad_right))

        if self.method == 'max':
            return F.max_pool1d(x, kernel_size=self.kernel_size, stride=self.stride)
        elif self.method in ['avg', 'mean']:
            return F.avg_pool1d(x, kernel_size=self.kernel_size, stride=self.stride)
        else:
            raise NotImplementedError(f"Pooling method {self.method} not implemented")

class RMSBatchNorm(nn.Module): # no linear
    """RMS Batch Normalization supporting both channels-first and channels-last formats.

    Normalizes over the channel dimension using stored running statistics.
    Matches JAX: alphagenome_research.model.layers.RMSBatchNorm

    Args:
        num_features: Number of channels.
        channels: Alias for num_features.
        eps: Small constant for numerical stability.
        channels_last: If True, expects (B, S, C) format. If False, expects (B, C, S).
                       Default False (channels-first, matching PyTorch conv conventions).
    """
    def __init__(self, num_features: int = 0, channels: int = 0, eps: float = 1e-5, channels_last: bool = False):
        super().__init__()
        num_features = num_features or channels
        if num_features == 0:
            raise ValueError("Must provide num_features or channels")
        self.num_features = num_features
        self.eps = eps
        self.channels_last = channels_last

        # Always store parameters as (C,) - standard PyTorch convention
        # Reshape for broadcasting happens in forward()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # JAX casts the inverse std dev to input dtype BEFORE multiplying by scale
        inv = self.weight * torch.rsqrt(self.running_var + self.eps).to(x.dtype)
        if self.channels_last:
            # NLC format (B, S, C) - parameters broadcast from the right
            return x * inv + self.bias
        else:
            # NCL format (B, C, S) - reshape for broadcasting
            return x * inv.view(1, -1, 1) + self.bias.view(1, -1, 1)

class LayerNorm(nn.Module): # no linear
    """Layer Normalization with optional RMSNorm mode (centering=False).

    Expects NLC format (B, S, C) - used by TransformerTower.
    Normalizes over the last dimension(s).

    Matches JAX: alphagenome_research.model.layers.LayerNorm
    JAX computes variance in float32 for numerical stability, then casts back.
    """
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True, rms_norm: bool = False):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = getattr(normalized_shape, 'tuple', lambda: normalized_shape)()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.rms_norm = rms_norm

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        dims = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))

        if self.rms_norm:
            # RMSNorm: x / sqrt(mean(x^2) + eps)
            # JAX computes variance in float32 for stability
            variance = torch.mean(x.float() ** 2, dim=dims, keepdim=True)
            inv = torch.rsqrt(variance + self.eps).to(input_dtype)
            x_norm = x * inv
        else:
            # Standard LayerNorm with centering
            # JAX: mean and variance both computed in float32
            mean = torch.mean(x.float(), dim=dims, keepdim=True)
            x_centered = x - mean.to(input_dtype)
            variance = torch.mean(x_centered.float() ** 2, dim=dims, keepdim=True)
            inv = torch.rsqrt(variance + self.eps).to(input_dtype)
            x_norm = x_centered * inv

        if self.elementwise_affine:
            return x_norm * self.weight + self.bias
        return x_norm
    
class StandardizedConv1d(nn.Conv1d):
    """1D Convolution with weight standardization and learned scaling.

    Expects NCL format (B, C, S).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation, groups=groups, bias=bias)
        # JAX uses padding='SAME'. PyTorch 'same' padding requires specific setup or manual padding.
        # We will handle padding in forward to match 'SAME' behavior roughly.
        self.pad_mode = padding
        
        # Scale parameter
        self.scale = nn.Parameter(torch.ones(out_channels, 1, 1))

    def forward(self, x):
        # x: (B, C, S) - NCL format
        
        # Weight standardization
        # JAX: w -= mean(w, axis=(0, 1)) -> (kernel_width, input_channels)
        # PyTorch weight: (out_channels, in_channels, kernel_width)
        # We want to standardize over (in_channels, kernel_width) corresponding to fan-in?
        # JAX shape: (width, input_channels, output_channels). Mean axis (0, 1) means mean over width and input_channels.
        # PyTorch equivalent: mean over (1, 2).
        
        w = self.weight
        mean = w.mean(dim=(1, 2), keepdim=True)
        var = w.var(dim=(1, 2), keepdim=True, unbiased=False) 
        
        fan_in = self.in_channels * self.kernel_size[0]
        scale_factor = torch.rsqrt(torch.maximum(var * fan_in, torch.tensor(1e-4, device=w.device, dtype=w.dtype))) * self.scale
        
        w_standardized = (w - mean) * scale_factor
        
        # Padding 'SAME' manually if needed, or use functional
        # For even kernel sizes, 'same' padding is asymmetric. JAX/TF usually pad more on the right.
        if self.pad_mode == 'same':
            # Padding formulation:
            # Note: this formula is valid for stride=1 only (the only stride used in this model).
            pad_total = self.kernel_size[0] - 1
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x = F.pad(x, (pad_left, pad_right))
            
        return F.conv1d(x, w_standardized, self.bias, self.stride, 0, self.dilation, self.groups)

class ConvBlock(nn.Module): # no linear
    """Convolution block operating on NCL format (B, C, S)."""

    def __init__(self, in_channels, out_channels, kernel_size, name=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.norm = RMSBatchNorm(in_channels)

        if kernel_size == 1:
            # Use Conv1d(k=1) instead of Linear - same math, native NCL
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv = StandardizedConv1d(in_channels, out_channels, kernel_size, padding='same')

    def forward(self, x):
        # x: (B, C, S) - NCL format, no transposes needed
        return self.conv(gelu(self.norm(x)))

class DnaEmbedder(nn.Module): # no linear
    """Embeds one-hot DNA to feature space. Expects NCL format (B, 4, S)."""

    def __init__(self):
        super().__init__()
        # JAX: Conv1D(768, 15) -> Input 4 channels (one-hot)
        # Then + ConvBlock(768, 5)
        self.conv1 = nn.Conv1d(4, 768, kernel_size=15, padding='same')
        self.block = ConvBlock(768, 768, kernel_size=5)

    def forward(self, x):
        # x: (B, 4, S) - NCL format, no transposes needed
        out = self.conv1(x)
        return out + self.block(out)

class DownResBlock(nn.Module): # no linear
    """Downsampling residual block. Expects NCL format (B, C, S)."""

    def __init__(self, in_channels, name=None):
        super().__init__()
        self.out_channels_int = in_channels + 128
        self.block1 = ConvBlock(in_channels, self.out_channels_int, kernel_size=5)
        self.block2 = ConvBlock(self.out_channels_int, self.out_channels_int, kernel_size=5)

    def forward(self, x):
        # x: (B, C, S) - NCL format
        out = self.block1(x)

        # Residual connection with channel padding
        # F.pad pads from last dim backwards: (left_S, right_S, left_C, right_C)
        # We want to pad channels (dim 1), so: (0, 0, 0, 128)
        x_padded = F.pad(x, (0, 0, 0, 128))

        out = out + x_padded
        return out + self.block2(out)

class UpResBlock(nn.Module): # no linear
    """Upsampling residual block with skip connection. Expects NCL format (B, C, S)."""

    def __init__(self, in_channels, skip_channels):
        super().__init__()
        self.conv_in = ConvBlock(in_channels, skip_channels, kernel_size=5)
        self.residual_scale = nn.Parameter(torch.ones(1))
        self.pointwise = ConvBlock(skip_channels, skip_channels, kernel_size=1)
        self.conv_out = ConvBlock(skip_channels, skip_channels, kernel_size=5)

    def forward(self, x, unet_skip):
        # x: (B, C, S) - NCL format
        # unet_skip: (B, C_skip, S*2) - skip has 2x sequence length

        # 1. First block + slice channels to match skip
        # Channels are dim 1 in NCL: x[:, :skip_channels, :]
        out = self.conv_in(x) + x[:, :unet_skip.shape[1], :]

        # 2. Upsample sequence (dim 2 in NCL)
        out = torch.repeat_interleave(out, repeats=2, dim=2)

        out = out * self.residual_scale

        # 3. Add skip connection
        out = out + self.pointwise(unet_skip)

        # 4. Final block
        return out + self.conv_out(out)
    

class SequenceEncoder(nn.Module): # no linear
    """Encodes DNA sequence to trunk representation. Outputs NCL format (B, C, S)."""

    def __init__(self):
        super().__init__()
        self.gradient_checkpointing = False
        self.dna_embedder = DnaEmbedder() # no linear
        self.pool = Pool1d(kernel_size=2)

        self.down_blocks = nn.ModuleList()
        in_channels = 768  # Initial output from embedder

        # 6 blocks: bin sizes 2, 4, 8, 16, 32, 64
        self.bin_sizes = [2, 4, 8, 16, 32, 64]
        for _ in self.bin_sizes:
            self.down_blocks.append(DownResBlock(in_channels))
            in_channels += 128

    def forward(self, x):
        # x input: (B, S, 4) from user - NLC format
        x = x.transpose(1, 2)  # → (B, 4, S) NCL format

        intermediates = {}
        x = self.dna_embedder(x)
        intermediates['bin_size_1'] = x
        x = self.pool(x)

        for i, block in enumerate(self.down_blocks):
            if self.gradient_checkpointing and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            bin_size = self.bin_sizes[i]
            intermediates[f'bin_size_{bin_size}'] = x
            x = self.pool(x)

        # x: (B, 1536, 1024), intermediates: all NCL
        return x, intermediates

class SequenceDecoder(nn.Module): # no linear
    """Decodes trunk to full resolution. Operates on NCL format (B, C, S)."""

    def __init__(self):
        super().__init__()
        self.gradient_checkpointing = False

        # bin sizes: 64, 32, 16, 8, 4, 2, 1
        self.bin_sizes = [64, 32, 16, 8, 4, 2, 1]

        # Channel sizes from encoder:
        # 1: 768, 2: 896, 4: 1024, 8: 1152, 16: 1280, 32: 1408, 64: 1536
        self.up_blocks = nn.ModuleList()
        current_channels = 1536

        skip_channels_map = {
            64: 1536, 32: 1408, 16: 1280, 8: 1152, 4: 1024, 2: 896, 1: 768
        }

        for bin_size in self.bin_sizes:
            skip_ch = skip_channels_map[bin_size]
            self.up_blocks.append(UpResBlock(
                in_channels=current_channels, skip_channels=skip_ch
            ))
            current_channels = skip_ch

    def forward(self, x, intermediates):
        # x: (B, C, S) - NCL format
        for i, bin_size in enumerate(self.bin_sizes):
            block = self.up_blocks[i]
            skip = intermediates.pop(f'bin_size_{bin_size}')
            if self.gradient_checkpointing and torch.is_grad_enabled():
                x = checkpoint(block, x, skip, use_reentrant=False)
            else:
                x = block(x, skip)
            del skip
        return x  # (B, 768, S) - NCL format

class TransformerTower(nn.Module): # has k for lora
    """Transformer tower. Operates on NLC format (B, S, C) - native for attention."""

    def __init__(self, k, d_model):
        super().__init__()
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList()
        # 9 blocks
        for i in range(9):
            is_even = (i % 2 == 0)
            pair_update = PairUpdateBlock(k, d_model) if is_even else None

            # Layer components
            attn_bias = AttentionBiasBlock(k, pair_dim=128)
            mha = MHABlock(k, d_model)
            mlp = MLPBlock(k, d_model)

            self.blocks.append(nn.ModuleDict({
                'pair_update': pair_update,
                'attn_bias': attn_bias,
                'mha': mha,
                'mlp': mlp
            }))

    def _forward_block(self, block, x, pair_x, compute_dtype):
        if block['pair_update'] is not None:
            pair_x = block['pair_update'](x, pair_x, compute_dtype=compute_dtype)
        mha_bias = block['attn_bias'](pair_x)
        x = x + block['mha'](x, mha_bias, compute_dtype=compute_dtype)
        x = x + block['mlp'](x)
        return x, pair_x

    def forward(self, x, compute_dtype=None):
        # x: (B, S, D)
        pair_x = None

        for block in self.blocks:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                x, pair_x = checkpoint(
                    self._forward_block, block, x, pair_x, compute_dtype,
                    use_reentrant=False,
                )
            else:
                x, pair_x = self._forward_block(block, x, pair_x, compute_dtype)

        return x, pair_x

class AlphaGenome(nn.Module):
    """Main AlphaGenome model for genomic sequence analysis.

    Matches JAX: alphagenome_research.model.model.AlphaGenome

    The model predicts various genomic tracks (ATAC, DNASE, CAGE, etc.) and
    contact maps from DNA sequences.

    Note that by default we include the splicing heads.

    Args:
        k: LLRA Rank for linear layers. If "full", uses full-rank linear layers (no llra).
        num_organisms: Number of organisms (default 2: human, mouse).
        dtype_policy: DtypePolicy controlling precision for params/compute/output.
                      Use DtypePolicy.full_float32() for safe defaults (works everywhere).
                      Use DtypePolicy.mixed_precision() for JAX-matching bfloat16 compute.
                      Defaults to DtypePolicy.full_float32() if not specified.
        track_means_dict: Optional dict mapping head names to track_means tensors.
                          Not needed if loading weights from convert_weights.py, which
                          bundles track_means into the weights file.
        gradient_checkpointing: If True, enable gradient checkpointing in the
                                encoder and decoder to reduce memory usage during
                                training at the cost of additional compute.

    Example:
        from alphagenome_pytorch import AlphaGenome
        from alphagenome_pytorch.config import DtypePolicy

        # Load pretrained model (recommended):
        model = AlphaGenome.from_pretrained('model.pth', device='cuda')

        # Or using load_state_dict directly:
        model = AlphaGenome()
        model.load_state_dict(torch.load('model.pth', weights_only=True))
        model.cuda()

        # JAX-matching mixed precision (bfloat16 compute):
        model = AlphaGenome.from_pretrained(
            'model.pth',
            dtype_policy=DtypePolicy.mixed_precision(),
        )
    """
    def __init__(
        self,
        k: Union[int, str] = "full",
        num_organisms: int = 2,
        dtype_policy: Optional[DtypePolicy] = None,
        track_means_dict: Optional[dict] = None,
        gradient_checkpointing: bool = False,
    ):
        """Initialize AlphaGenome model."""
        super().__init__()
        self.num_organisms = num_organisms
        track_means_dict = track_means_dict or {}

        # Set dtype policy (default: full float32, works everywhere)
        self.dtype_policy = dtype_policy if dtype_policy is not None else DtypePolicy.default()

        self.encoder = SequenceEncoder()
        self.encoder.gradient_checkpointing = gradient_checkpointing

        # Architecture dimension constants
        TRUNK_DIM = 1536           # Encoder output / transformer dimension
        EMBEDDING_128BP_DIM = 3072 # 128bp output embedder dimension
        DECODER_DIM = 768          # Decoder output / 1bp embedder input dimension

        self.organism_embed = nn.Embedding(num_organisms, TRUNK_DIM)
        
        self.tower = TransformerTower(k= k, d_model=TRUNK_DIM)
        self.tower.gradient_checkpointing = gradient_checkpointing
        self.decoder = SequenceDecoder()
        self.decoder.gradient_checkpointing = gradient_checkpointing

        # Output Embedders - all NCL format
        # Trunk (1536) -> 128bp Embeddings (3072)
        self.embedder_128bp = OutputEmbedder(
            in_channels=TRUNK_DIM,
            out_channels=EMBEDDING_128BP_DIM,
            num_organisms=num_organisms,
        )

        # Decoder (768) + Skip (3072) -> 1bp Embeddings (1536)
        self.embedder_1bp = OutputEmbedder(
            in_channels=DECODER_DIM,
            out_channels=TRUNK_DIM,
            num_organisms=num_organisms,
        )
        # Skip projection: Conv1d for NCL format
        self.embedder_1bp.project_skip = nn.Conv1d(
            EMBEDDING_128BP_DIM,
            TRUNK_DIM,
            kernel_size=1,
            bias=False,
        )
        
        # Pair Embedder
        self.embedder_pair = OutputPair(dim=128, num_organisms=num_organisms)
        
        # Heads
        # We replace placeholders with specific named heads matching JAX reference.
        # Resolutions: Most use [1, 128], some use only [128].
        
        self.heads = nn.ModuleDict()

        # Embedding dimensions per resolution (single source of truth)
        _EMBEDDING_DIMS = {1: TRUNK_DIM, 128: EMBEDDING_128BP_DIM}

        # Helper to simplify head creation
        def add_head(name, num_tracks, resolutions=(1, 128), apply_squashing=False):
            # Get track_means from dict if provided
            track_means = track_means_dict.get(name, None)
            self.heads[name] = GenomeTracksHead(
                in_channels=_EMBEDDING_DIMS,
                num_tracks=num_tracks,
                resolutions=resolutions,
                num_organisms=num_organisms,
                apply_squashing=apply_squashing,
                track_means=track_means,
            )

        # Standard Heads Config (from reference heads.py)
        # apply_squashing is True only for RNA_SEQ
        add_head('atac', 256, [1, 128], apply_squashing=False)
        add_head('dnase', 384, [1, 128], apply_squashing=False)
        add_head('procap', 128, [1, 128], apply_squashing=False)
        add_head('cage', 640, [1, 128], apply_squashing=False)
        add_head('rna_seq', 768, [1, 128], apply_squashing=True)
        add_head('chip_tf', 1664, [128], apply_squashing=False)
        add_head('chip_histone', 1152, [128], apply_squashing=False)

        # Contact Maps Head (from pair embeddings)
        self.contact_maps_head = ContactMapsHead(
            k, 
            in_features=PAIR_EMBEDDING_DIM,
            num_tracks=CONTACT_MAPS_OUTPUT_TRACKS,
            num_organisms=num_organisms,
        )

        # Splice heads have different structure (Single resolution usually or specific logic)
        # reference: SpliceSitesClassificationHead, SpliceSitesUsageHead, SpliceSitesJunctionHead
        if num_organisms == 1:
            splice_usage_tracks_per_organism = (734,)
            splice_junction_tracks_per_organism = (367,)
        elif num_organisms == 2:
            # Human/mouse defaults matching the current pretrained setup.
            splice_usage_tracks_per_organism = (734, 180)
            splice_junction_tracks_per_organism = (367, 367)
        else:
            warnings.warn(
                "AlphaGenome currently only supports num_organisms in {1, 2}. "
                "For now, splicing heads use hardcoded human/mouse track configs."
            )

        self.splice_sites_classification_head = SpliceSitesClassificationHead(
            in_channels=TRUNK_DIM, num_organisms=num_organisms
        )
        self.splice_sites_usage_head = SpliceSitesUsageHead(
            in_channels=TRUNK_DIM,
            num_output_tracks=max(splice_usage_tracks_per_organism),
            num_organisms=num_organisms,
            num_tracks_per_organism=splice_usage_tracks_per_organism,
        )
        self.splice_sites_junction_head = SpliceSitesJunctionHead(
            in_channels=TRUNK_DIM,
            hidden_dim=DECODER_DIM,
            num_tissues=max(splice_junction_tracks_per_organism),
            num_organisms=num_organisms,
            num_tracks_per_organism=splice_junction_tracks_per_organism,
        )

        self._track_metadata_catalog: TrackMetadataCatalog | None = None

        # Convert model parameters to params_dtype
        # JAX keeps params in float32 even when computing in bfloat16
        if self.dtype_policy.params_dtype != torch.float32:
            self.to(self.dtype_policy.params_dtype)

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        dtype_policy: Optional[DtypePolicy] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> "AlphaGenome":
        """Load a pretrained AlphaGenome model from a weights file.

        Args:
            path: Path to the weights file (.pth) created by convert_weights.py.
            dtype_policy: DtypePolicy for precision control. Defaults to DtypePolicy.full_float32().
                          Use DtypePolicy.mixed_precision() for JAX-matching bfloat16 compute.
            device: Device to load the model onto ('cuda', 'cpu', etc.). If None, loads to CPU.
            **kwargs: Additional arguments passed to AlphaGenome constructor
                      (e.g., num_organisms, gradient_checkpointing).

        Returns:
            AlphaGenome model with loaded weights.

        Note:
            This method is backward compatible with older weights files that don't
            include track_means buffers. If track_means are missing, a warning is
            issued and default values (zeros) are used. For proper output scaling
            with older weights, load track_means separately using load_track_means().

        Example:
            # Load model to GPU with default settings (float32):
            model = AlphaGenome.from_pretrained('model.pth', device='cuda')

            # Load with JAX-matching mixed precision (bfloat16 compute):
            model = AlphaGenome.from_pretrained(
                'model.pth',
                dtype_policy=DtypePolicy.mixed_precision(),
                device='cuda:0',
            )
        """
        if dtype_policy is None:
            dtype_policy = DtypePolicy.default()

        # Create model and move to target device first for efficient weight loading.
        # This allows loading state_dict directly to the target device, avoiding
        # cross-device transfers during load_state_dict.
        model = cls(dtype_policy=dtype_policy, **kwargs)
        if device:
            model.to(device)

        # Load state dict directly to the device where model lives
        map_location = device if device else 'cpu'
        if Path(path).suffix == '.safetensors':
            from safetensors.torch import load_file
            state_dict = load_file(path, device=str(map_location))
        else:
            state_dict = torch.load(path, map_location=map_location, weights_only=True)

        # Use strict=False to allow loading older weights without track_means,
        # but validate the result to catch other issues
        result = model.load_state_dict(state_dict, strict=False)

        # Check for unexpected keys (always an error - indicates architecture mismatch)
        if result.unexpected_keys:
            raise RuntimeError(
                f"Unexpected keys in state_dict: {result.unexpected_keys}. "
                "This may indicate a version mismatch between the weights file "
                "and the model architecture."
            )

        # Check for missing keys - allow track_means but warn, error on others
        if result.missing_keys:
            track_means_keys = [k for k in result.missing_keys if 'track_means' in k]
            other_missing = [k for k in result.missing_keys if 'track_means' not in k]

            if other_missing:
                raise RuntimeError(
                    f"Missing keys in state_dict: {other_missing}. "
                    "This may indicate a version mismatch between the weights file "
                    "and the model architecture."
                )

            if track_means_keys:
                warnings.warn(
                    f"Weights file is missing track_means buffers ({len(track_means_keys)} keys). "
                    "Using default values (zeros). For proper output scaling, either: "
                    "(1) use a newer weights file with bundled track_means, or "
                    "(2) load track_means separately using model.load_track_means().",
                    UserWarning,
                    stacklevel=2,
                )

        return model

    def set_track_metadata_catalog(self, catalog: TrackMetadataCatalog) -> None:
        """Attach a metadata catalog used by named output views."""
        self._track_metadata_catalog = catalog

    def load_track_metadata(
        self,
        metadata_path: str | Path,
        *,
        default_organism: int = 0,
        default_output_name: str | None = None,
    ) -> TrackMetadataCatalog:
        """Load track metadata (parquet/csv/tsv) and attach it to the model."""
        catalog = TrackMetadataCatalog.from_file(
            metadata_path,
            default_organism=default_organism,
            default_output_name=default_output_name,
        )
        self._track_metadata_catalog = catalog
        return catalog

    def named_outputs(
        self,
        outputs: dict,
        *,
        organism: int | str | torch.Tensor | None = None,
        strict_metadata: bool = False,
        metadata_catalog: TrackMetadataCatalog | None = None,
        channels_last: bool = True,
        include_padding: bool = False,
    ) -> NamedOutputs:
        """Wrap raw model outputs with metadata-aware named views.

        Args:
            outputs: Raw model output dict.
            organism: Organism index or name.
            strict_metadata: If True, raise on missing/mismatched metadata.
            metadata_catalog: Override the model's attached catalog.
            channels_last: If True, track axis is last dimension.
            include_padding: If True, keep padding tracks. If False
                (default), padding tracks are stripped.
        """
        catalog = metadata_catalog if metadata_catalog is not None else self._track_metadata_catalog
        if catalog is None and not include_padding:
            catalog = TrackMetadataCatalog.load_builtin()
            self._track_metadata_catalog = catalog
        return NamedOutputs.from_raw(
            outputs,
            organism=organism,
            catalog=catalog,
            strict_metadata=strict_metadata,
            channels_last=channels_last,
            include_padding=include_padding,
        )

    @classmethod
    def from_delta(
        cls,
        delta_path: Union[str, Path],
        base_path: Union[str, Path],
        dtype_policy: Optional[DtypePolicy] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> "AlphaGenome":
        """Load a finetuned AlphaGenome model from delta weights and base weights.

        This is the simplest way to load a finetuned model. It reconstructs the
        full model from a small delta weights file (adapters + heads) and the
        base pretrained weights.

        Args:
            delta_path: Path to the delta weights file (.safetensors or .pth)
                created by ``export_delta_weights()``.
            base_path: Path to the base pretrained weights file (.pth or
                .safetensors) created by ``convert_weights.py``.
            dtype_policy: DtypePolicy for precision control. Defaults to
                DtypePolicy.full_float32().
            device: Device to load the model onto ('cuda', 'cpu', etc.).
                If None, loads to CPU.
            **kwargs: Additional arguments passed to AlphaGenome constructor
                (e.g., num_organisms, gradient_checkpointing).

        Returns:
            AlphaGenome model with base weights, adapters, and finetuned heads.

        Example:
            >>> model = AlphaGenome.from_delta(
            ...     'colleague_lora.safetensors',
            ...     'alphagenome_pretrained.pth',
            ...     device='cuda',
            ... )
        """
        from alphagenome_pytorch.extensions.finetuning.checkpointing import (
            load_delta_config,
            load_delta_weights,
        )
        from alphagenome_pytorch.extensions.finetuning.transfer import (
            load_trunk,
            prepare_for_transfer,
        )

        if dtype_policy is None:
            dtype_policy = DtypePolicy.default()

        # 1. Create base model and load pretrained trunk
        model = cls(dtype_policy=dtype_policy, **kwargs)
        model = load_trunk(model, base_path, exclude_heads=True)

        # 2. Read config and set up adapters/heads
        config = load_delta_config(delta_path)
        model = prepare_for_transfer(model, config)

        # 3. Load delta weights
        load_delta_weights(model, delta_path)

        # 4. Move to target device
        if device:
            model.to(device)

        return model

    def _compute_embeddings_ncl(self, dna_sequence, organism_index, resolutions=None):
        """Internal method to compute embeddings in NCL format.

        Returns:
            Tuple of (embeddings_1bp, embeddings_128bp, embeddings_pair, need_1bp)
            where sequence embeddings are in NCL format (B, C, S).
        """
        # Cast input to compute dtype
        dna_sequence = self.dtype_policy.cast_to_compute(dna_sequence)

        # ===== ENCODER (NCL) =====
        trunk, intermediates = self.encoder(dna_sequence)  # trunk: (B, 1536, 1024)

        # ===== NCL → NLC for Transformer =====
        trunk = trunk.transpose(1, 2)  # → (B, 1024, 1536)

        # Add organism embedding (NLC format)
        org_emb = self.organism_embed(organism_index).unsqueeze(1)  # (B, 1, 1536)
        trunk = trunk + org_emb

        # ===== TRANSFORMER (NLC) =====
        trunk, pair_activations = self.tower(trunk, compute_dtype=self.dtype_policy.compute_dtype)
        # trunk: (B, 1024, 1536) NLC

        # ===== NLC → NCL for Decoder/Embedders =====
        trunk_ncl = trunk.transpose(1, 2)  # → (B, 1536, 1024)

        # Determine which resolutions are needed
        need_1bp = resolutions is None or 1 in resolutions

        # ===== OUTPUT EMBEDDINGS (NCL format) =====
        # 128bp Embeddings - always computed
        embeddings_128bp = self.embedder_128bp(
            trunk_ncl, organism_index, channels_last=False
        )  # (B, 3072, 1024)

        # 1bp Embeddings (from decoder + skip) - skip if not needed
        if need_1bp:
            decoded_x = self.decoder(trunk_ncl, intermediates)  # (B, 768, 131072)
            embeddings_1bp = self.embedder_1bp(
                decoded_x, organism_index, skip_x=embeddings_128bp, channels_last=False
            )  # (B, 1536, 131072)
        else:
            embeddings_1bp = None
            del intermediates  # Free memory from encoder skip connections

        # Pair Embeddings (B, S, S, D) - different format, not NCL
        embeddings_pair = self.embedder_pair(pair_activations, organism_index)

        return embeddings_1bp, embeddings_128bp, embeddings_pair, need_1bp

    def encode(
        self,
        dna_sequence,
        organism_index,
        resolutions=None,
        channels_last=True,
    ):
        """Extract embeddings without running prediction heads.

        This method runs the encoder, transformer, decoder, and output embedders
        to produce embeddings that can be used with custom heads for fine-tuning.

        Args:
            dna_sequence: One-hot encoded DNA sequence (B, S, 4) - NLC input format.
            organism_index: Organism index per batch (B,). 0=human, 1=mouse.
            resolutions: Tuple of resolutions to compute, e.g. (1, 128) or (128,).
                         If None, computes all resolutions. When 1bp is not needed,
                         the expensive decoder is skipped for faster computation.
            channels_last: Output format for sequence embeddings.
                - True (default): NLC format (B, S, C) - user-friendly, matches JAX
                - False: NCL format (B, C, S) - efficient for Conv1d heads

        Returns:
            Dict with keys:
                - 'embeddings_1bp': (B, S, 1536) or (B, 1536, S) at 1bp resolution.
                  Only present if 1 is in resolutions (or resolutions is None).
                - 'embeddings_128bp': (B, S//128, 3072) or (B, 3072, S//128) at 128bp.
                - 'embeddings_pair': (B, S//2048, S//2048, 128) pair embeddings.

        Example:
            # Get embeddings for fine-tuning with a custom head
            model = AlphaGenome.from_pretrained('model.pth', device='cuda')
            model.eval()

            # Freeze backbone
            for param in model.parameters():
                param.requires_grad = False

            # Get embeddings (128bp only for efficiency)
            with torch.no_grad():
                emb = model.encode(dna_seq, organism_idx, resolutions=(128,))

            # Use with custom head (NCL format for Conv1d)
            emb = model.encode(dna_seq, organism_idx, channels_last=False)
            custom_output = my_conv_head(emb['embeddings_128bp'])
        """
        embeddings_1bp, embeddings_128bp, embeddings_pair, need_1bp = \
            self._compute_embeddings_ncl(dna_sequence, organism_index, resolutions)

        # Build output dict with requested format
        # Use contiguous() after transpose to ensure memory layout is optimal
        # for downstream operations (Conv1d, CUDA kernels, etc.)
        outputs = {}

        if channels_last:
            if need_1bp:
                outputs['embeddings_1bp'] = embeddings_1bp.transpose(1, 2).contiguous()
            outputs['embeddings_128bp'] = embeddings_128bp.transpose(1, 2).contiguous()
        else:
            if need_1bp:
                outputs['embeddings_1bp'] = embeddings_1bp
            outputs['embeddings_128bp'] = embeddings_128bp

        outputs['embeddings_pair'] = embeddings_pair

        return self._cast_outputs(outputs)

    def forward(
        self,
        dna_sequence,
        organism_index,
        *,
        splice_site_positions=None,
        return_embeddings=False,
        return_scaled_predictions=False,
        resolutions=None,
        heads: Optional[Tuple[str, ...]] = None,
        channels_last=True,
        embeddings_only=False,
        encoder_only=False,
    ):
        """Forward pass through the model.

        Args:
            dna_sequence: One-hot encoded DNA sequence (B, S, 4) - NLC input
            organism_index: Organism index per batch (B,). 0=human, 1=mouse.
            splice_site_positions: Optional pre-computed splice site positions
                (B, 4, K). If provided, skips internal Top-K selection.
            return_embeddings: If True, include embeddings in output.
            return_scaled_predictions: If True, return model space (for loss).
                                       If False, return experimental space (for inference).
            resolutions: Tuple of resolutions to compute, e.g. (1, 128) or (128,).
                         If None, computes all resolutions.
            heads: Tuple of head names to compute, e.g. ('atac',) or ('atac', 'dnase').
                   If None, computes all heads. Use this to skip expensive unused heads
                   during inference.
            channels_last: Format for embeddings and head outputs.
                - True (default): NLC format (B, S, C) - user-friendly, matches JAX
                - False: NCL format (B, C, S) - for training efficiency (0 transposes)
            embeddings_only: If True, skip all head computation and only return
                embeddings. Useful for fine-tuning where only a custom head is used.
                Implies return_embeddings=True.
            encoder_only: If True, run only the CNN encoder (skip organism embedding,
                transformer, decoder, and heads). Returns raw encoder output in
                ``{"encoder_output": tensor}`` where tensor has shape (B, S//128, 1536).

        Returns:
            Dict of predictions from each head. Keys are head names
            (atac, dnase, cage, etc.), values are dicts mapping
            resolution (1 or 128) to prediction tensors.
            If return_embeddings is True, also contains 'embeddings_1bp' and 'embeddings_128bp'.
            If encoder_only is True, returns ``{"encoder_output": tensor}`` only.

        Raises:
            ValueError: If unknown head names are specified in ``heads``.
        """
        if encoder_only:
            # Return raw CNN encoder output before organism embedding and transformer.
            trunk, _intermediates = self.encoder(dna_sequence)
            outputs = {"encoder_output": trunk}
            return self._cast_outputs(outputs)

        # Validate heads parameter
        if heads is not None:
            # Build set of all valid head names
            valid_heads = set(self.heads.keys())
            if self.contact_maps_head is not None:
                valid_heads.add('contact_maps')
            if self.splice_sites_classification_head is not None:
                valid_heads.add('splice_sites')
            if self.splice_sites_usage_head is not None:
                valid_heads.add('splice_site_usage')
            if self.splice_sites_junction_head is not None:
                valid_heads.add('splice_junctions')

            # Check for unknown heads
            unknown_heads = set(heads) - valid_heads
            if unknown_heads:
                raise ValueError(
                    f"Unknown head names: {sorted(unknown_heads)}. "
                    f"Available heads: {sorted(valid_heads)}"
                )
            head_set = set(heads)
        else:
            head_set = None

        # Compute embeddings (NCL format internally)
        embeddings_1bp, embeddings_128bp, embeddings_pair, need_1bp = \
            self._compute_embeddings_ncl(dna_sequence, organism_index, resolutions)

        if need_1bp:
            embeddings_dict = {1: embeddings_1bp, 128: embeddings_128bp}
        else:
            embeddings_dict = {128: embeddings_128bp}

        # ===== HEADS =====
        outputs = {}

        if not embeddings_only:
            for name, head in self.heads.items():
                if head_set is not None and name not in head_set:
                    continue
                outputs[name] = head(
                    embeddings_dict, organism_index,
                    return_scaled=return_scaled_predictions,
                    channels_last=channels_last,
                )

            # Contact Maps
            if self.contact_maps_head is not None:
                if head_set is None or 'contact_maps' in head_set:
                    outputs['contact_maps'] = self.contact_maps_head(
                        embeddings_pair, organism_index, channels_last=channels_last
                    )

            # Splice predictions (require 1bp embeddings)
            need_splice = head_set is None or any(
                k in head_set for k in ('splice_sites', 'splice_site_usage', 'splice_junctions')
            )
            if need_1bp and need_splice:
                # Also compute classification when junction needs it for position generation
                classification_output = None
                need_classification = (
                    head_set is None
                    or 'splice_sites' in head_set
                    or (
                        'splice_junctions' in head_set
                        and splice_site_positions is None
                    )
                )
                if self.splice_sites_classification_head is not None and need_classification:
                    classification_output = self.splice_sites_classification_head(
                        embeddings_1bp, organism_index, channels_last=channels_last
                    )
                    if head_set is None or 'splice_sites' in head_set:
                        outputs['splice_sites'] = classification_output
                if self.splice_sites_usage_head is not None:
                    if head_set is None or 'splice_site_usage' in head_set:
                        outputs['splice_site_usage'] = self.splice_sites_usage_head(
                            embeddings_1bp, organism_index, channels_last=channels_last
                        )

                if self.splice_sites_junction_head is not None:
                    if head_set is None or 'splice_junctions' in head_set:
                        # Use provided positions if given, otherwise generate from classification
                        if splice_site_positions is not None:
                            top_k_positions = splice_site_positions
                        else:
                            if classification_output is None:
                                raise ValueError(
                                    "splice_junctions requires either splice_site_positions "
                                    "or an available splice_sites classification head"
                                )
                            # probs: (B, S, 5) NLC - already correct format for generate_splice_site_positions
                            splice_site_probs = classification_output['probs']

                            # If NCL (channels_last=False), transpose back to NLC for generate_splice_site_positions
                            if not channels_last:
                                splice_site_probs = splice_site_probs.transpose(1, 2)

                            top_k_positions = generate_splice_site_positions(
                                ref=splice_site_probs,
                                alt=None,
                                true_splice_sites=None,
                                k=512,
                                pad_to_length=512,
                                threshold=0.1,
                            )
                        outputs['splice_junctions'] = self.splice_sites_junction_head(
                            embeddings_1bp,
                            organism_index,
                            channels_last=channels_last,
                            splice_site_positions=top_k_positions,
                        )

        if return_embeddings or embeddings_only:
            if channels_last:
                if need_1bp:
                    outputs['embeddings_1bp'] = embeddings_1bp.transpose(1, 2).contiguous()
                outputs['embeddings_128bp'] = embeddings_128bp.transpose(1, 2).contiguous()
            else:
                if need_1bp:
                    outputs['embeddings_1bp'] = embeddings_1bp
                outputs['embeddings_128bp'] = embeddings_128bp

        return self._cast_outputs(outputs)

    def _cast_outputs(self, outputs):
        """Recursively cast all output tensors to output_dtype."""
        if torch.is_tensor(outputs):
            return self.dtype_policy.cast_to_output(outputs)
        if isinstance(outputs, dict):
            return {k: self._cast_outputs(v) for k, v in outputs.items()}
        if isinstance(outputs, (list, tuple)):
            casted = [self._cast_outputs(v) for v in outputs]
            return type(outputs)(casted)
        return outputs

    @staticmethod
    def _upcast_outputs(outputs):
        """Recursively upcast low-precision floating-point tensors to float32.

        Matches JAX's tensor_utils.upcast_floating: only upcasts floating-point
        types smaller than float32 (bfloat16, float16). Leaves int tensors and
        float32+ tensors unchanged.
        """
        if torch.is_tensor(outputs):
            if outputs.is_floating_point() and outputs.dtype in (torch.bfloat16, torch.float16):
                return outputs.float()
            return outputs
        if isinstance(outputs, dict):
            return {k: AlphaGenome._upcast_outputs(v) for k, v in outputs.items()}
        if isinstance(outputs, (list, tuple)):
            upcasted = [AlphaGenome._upcast_outputs(v) for v in outputs]
            return type(outputs)(upcasted)
        return outputs

    @torch.no_grad()
    def predict(
        self,
        dna_sequence: torch.Tensor,
        organism_index: Union[torch.Tensor, int],
        named_outputs: bool = False,
        strict_metadata: bool = False,
        include_padding: bool = False,
        **kwargs,
    ) -> dict | NamedOutputs:
        """Inference-mode forward pass with automatic dtype handling.

        Wraps forward() with:
        - torch.no_grad() for memory efficiency
        - torch.autocast for mixed-precision compute
        - Float32 upcasting of all outputs

        This matches the JAX reference's inference behavior, where outputs are
        upcast to float32 via _upcast_single_batch_predictions before being
        returned to the user.

        Args:
            dna_sequence: One-hot encoded DNA sequence (B, S, 4). Can be any
                float dtype — autocast handles weight/input casting.
            organism_index: Organism index per batch (B,). 0=human, 1=mouse.
            named_outputs: If True, return a ``NamedOutputs`` wrapper instead
                of a raw dict.
            strict_metadata: If True, raise when metadata is missing or
                mismatched for named outputs. If False, fallback placeholders
                are used.
            include_padding: If True, keep padding tracks in named outputs.
                If False (default), padding tracks are stripped. Only applies
                when ``named_outputs=True``.
            **kwargs: Additional arguments passed to forward()
                (e.g., return_embeddings, resolutions).

        Returns:
            Dict of predictions with all floating-point tensors in float32, or
            ``NamedOutputs`` when ``named_outputs=True``.
        """
        device_type = "cuda" if dna_sequence.is_cuda else "cpu"
        use_amp = self.dtype_policy.compute_dtype != torch.float32

        # Handle integer organism_index by converting to tensor of shape (B,)
        # forward() expects a tensor for embedding lookups.
        if isinstance(organism_index, int):
            batch_size = dna_sequence.shape[0]
            organism_index = torch.full(
                (batch_size,),
                organism_index,
                dtype=torch.long,
                device=dna_sequence.device
            )

        with torch.autocast(device_type=device_type, dtype=self.dtype_policy.compute_dtype, enabled=use_amp):
            outputs = self.forward(dna_sequence, organism_index, **kwargs)

        upcast_outputs = self._upcast_outputs(outputs)
        if named_outputs:
            return self.named_outputs(
                upcast_outputs,
                organism=organism_index,
                strict_metadata=strict_metadata,
                channels_last=kwargs.get("channels_last", True),
                include_padding=include_padding,
            )
        return upcast_outputs