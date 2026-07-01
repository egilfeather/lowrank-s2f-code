#!/usr/bin/env python3
"""
ag_static_quant.py
==================
Static (FX) int8 quantization of AlphaGenome into FOUR fixed-variant models:

    in4kb_res128     : 4 kb input,  128 bp out only
    in4kb_res1-128   : 4 kb input,  1 bp + 128 bp out
    in1Mb_res128     : 1 Mb input,  128 bp out only
    in1Mb_res1-128   : 1 Mb input,  1 bp + 128 bp out

All NON-splice heads are computed in every variant. The splice heads
(classification / usage / junction) are EXCLUDED: the junction head does a
data-dependent top-k that can't be made into a static graph, and excluding all
three keeps the traced graph clean.

Why a separate model per variant: the `need_1bp` branch gates the decoder + 1bp
embedder. FX bakes whichever path it traces into a static graph, so "128 only"
and "1+128" are genuinely different graphs / different quantized weights.

What actually becomes int8: FX + fbgemm quantize `nn.Linear` (the transformer
LoRA linears — the prize) and plain `nn.Conv1d` (e.g. the DNA embedder's k=15
conv). It does NOT quantize the custom `StandardizedConv1d` / `MultiOrganism*`
ops (weight-standardized convs and hand-written einsums have no fbgemm kernel),
so the decoder-heavy 1bp path stays largely float. That matches the cost
analysis: int8 buys you the transformer in all four variants, less of the
decoder.

Usage:
    python ag_static_quant.py --ranks 1 8 64 512 --outdir ag_static_weights
"""

import os
import argparse
import importlib

import torch
import torch.nn as nn

from torch.ao.quantization import get_default_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
try:
    from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
except Exception:  # older torch
    from torch.quantization.fx.custom_config import PrepareCustomConfig

AG_ARCH = os.environ.get("AG_ARCH_MODULE", "ag_arch_quant")
ag = importlib.import_module(AG_ARCH)
AlphaGenome = ag.AlphaGenome

# ── The four variants: name -> (input_len, resolutions) ───────────────────────
VARIANTS = {
    "in4kb_res128":   (4096,      (128,)),
    "in4kb_res1-128": (4096,      (1, 128)),
    "in1Mb_res128":   (1_048_576, (128,)),
    "in1Mb_res1-128": (1_048_576, (1, 128)),
}

# ── Modules FX must NOT trace into. ───────────────────────────────────────────
def _leaf_module_classes():
    names = [
        "Pool1d", "RMSBatchNorm", "LayerNorm", "StandardizedConv1d",
        "MultiOrganismConv1d", "MultiOrganismLinear",
        "DownResBlock", "UpResBlock", "OutputEmbedder", "OutputPair",
        "SequenceToPairBlock",
        "GenomeTracksHead", "ContactMapsHead",
        "SpliceSitesClassificationHead", "SpliceSitesUsageHead",
        "SpliceSitesJunctionHead",
    ]
    out = []
    for n in names:
        cls = getattr(ag, n, None)
        if cls is not None:
            out.append(cls)
    return out


class AGStaticVariant(nn.Module):
    """Traceable, fixed-variant view of AlphaGenome.

    `need_1bp` is resolved from `resolutions` at construction, so the decoder
    branch is a Python constant the tracer bakes in. Splice heads are omitted.
    Returns a dict of non-splice head outputs (+ contact maps).
    """

    def __init__(self, model: nn.Module, resolutions):
        super().__init__()
        self.model = model
        self.resolutions = tuple(sorted(resolutions))
        self.need_1bp = 1 in self.resolutions

    def forward(self, dna_sequence, organism_index):
        m = self.model
        # dna_sequence = m.dtype_policy.cast_to_compute(dna_sequence)

        # Encoder (traced -> dna_embedder.conv1 quantizes; DownResBlocks are leaf)
        trunk, intermediates = m.encoder(dna_sequence)
        trunk = trunk.transpose(1, 2)
        trunk = trunk + m.organism_embed(organism_index).unsqueeze(1)

        # Transformer tower (traced -> LoRA linears quantize; seq2pair is leaf)
        trunk, pair_activations = m.tower(
            trunk, compute_dtype=m.dtype_policy.compute_dtype
        )
        trunk_ncl = trunk.transpose(1, 2)

        # Output embedders (leaf modules; run in float)
        embeddings_128bp = m.embedder_128bp(trunk_ncl, organism_index, channels_last=False)
        if self.need_1bp:
            decoded_x = m.decoder(trunk_ncl, intermediates)
            embeddings_1bp = m.embedder_1bp(
                decoded_x, organism_index, skip_x=embeddings_128bp, channels_last=False
            )
            embeddings_dict = {1: embeddings_1bp, 128: embeddings_128bp}
        else:
            embeddings_dict = {128: embeddings_128bp}
        embeddings_pair = m.embedder_pair(pair_activations, organism_index)

        # Non-splice heads (leaf modules; float). All heads computed.
        outputs = {}
        for name, head in m.heads.items():
            outputs[name] = head(
                embeddings_dict, organism_index,
                return_scaled=False, channels_last=True,
            )
        outputs["contact_maps"] = m.contact_maps_head(
            embeddings_pair, organism_index, channels_last=True
        )
        return outputs


def make_onehot(input_len, dtype=torch.float32, seed=0):
    """Random valid one-hot DNA sequence (1, L, 4) NLC + organism index."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, 4, (1, input_len), generator=g)
    x = torch.zeros(1, input_len, 4, dtype=dtype)
    x.scatter_(2, idx.unsqueeze(-1), 1.0)
    organism = torch.tensor([0], dtype=torch.long)
    return x, organism


def _qconfig_mapping():
    return QConfigMapping().set_global(get_default_qconfig("fbgemm"))


def _prepare_custom_config():
    return PrepareCustomConfig().set_non_traceable_module_classes(_leaf_module_classes())


def build_float_variant(rank, resolutions):
    """Float AGStaticVariant for a given LoRA rank + resolution set."""
    k = "full" if rank == "full" else rank
    model = AlphaGenome(k=k)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return AGStaticVariant(model, resolutions).eval()


def build_skeleton(rank, input_len, resolutions):
    """A converted (int8-structured) GraphModule with default scales, ready for
    load_state_dict(). Used by the timing script to reload saved weights."""
    variant = build_float_variant(rank, resolutions)
    example = make_onehot(input_len, dtype=torch.float32)
    prepared = prepare_fx(
        variant, _qconfig_mapping(), example_inputs=example,
        prepare_custom_config=_prepare_custom_config(),
    )
    return convert_fx(prepared)


def quantize_variant(rank, input_len, resolutions, n_calib=8):
    """Build, calibrate (on random one-hot), and convert one variant to int8."""
    variant = build_float_variant(rank, resolutions)
    example = make_onehot(input_len, dtype=torch.float32)

    prepared = prepare_fx(
        variant, _qconfig_mapping(), example_inputs=example,
        prepare_custom_config=_prepare_custom_config(),
    )

    # Calibration. Random one-hot is fine for a TIMING benchmark (int8 kernels
    # are selected the same way regardless of calibration data). For accuracy
    # work, replace this with real sequences.
    with torch.no_grad():
        for s in range(n_calib):
            x, org = make_onehot(input_len, dtype=torch.float32, seed=1000 + s)
            prepared(x, org)

    return convert_fx(prepared)


def quantize_all(ranks, outdir, n_calib=8):
    os.makedirs(outdir, exist_ok=True)
    for rank in ranks:
        for vname, (input_len, resolutions) in VARIANTS.items():
            print(f"[quantize] rank={rank} variant={vname} "
                  f"(len={input_len:,}, res={resolutions})", flush=True)
            qmodel = quantize_variant(rank, input_len, resolutions, n_calib=n_calib)
            path = os.path.join(outdir, f"ag_static_{vname}_lr{rank}.pth")
            torch.save(qmodel.state_dict(), path)
            print(f"          saved -> {path}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Static int8 quantization of AlphaGenome (4 variants)")
    p.add_argument("--ranks", nargs="+", default=["1", "8", "64", "512"],
                   help="LoRA ranks to quantize (ints or 'full').")
    p.add_argument("--outdir", default="ag_static_weights")
    p.add_argument("--n-calib", type=int, default=8)
    args = p.parse_args()

    ranks = [r if r == "full" else int(r) for r in args.ranks]
    quantize_all(ranks, args.outdir, n_calib=args.n_calib)


if __name__ == "__main__":
    main()
