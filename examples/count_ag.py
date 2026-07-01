#!/usr/bin/env python3
"""
ag_quantize.py  —  Quantize and benchmark AlphaGenome LoRA models.

Organism handling
-----------------
organism_index only selects slices of fixed weight tensors (nn.Embedding,
MultiOrganismLinear, MultiOrganismConv1d).  The compute graph is identical
for human (0) and mouse (1), so a SINGLE quantized model covers both organisms.
We calibrate with alternating human/mouse batches so the activation statistics
(min/max ranges for int8 clamping) reflect both organisms' distributions.

Sequence-length handling
------------------------
FX static quantization traces the model once and bakes the input shape into
the graph.  AlphaGenome's TransformerTower computes length-dependent attention
biases (torch.arange(S), _central_mask_features, relative position matrices),
which become compile-time constants under FX tracing.  The OutputEmbedder also
has a length-dependent repeat_interleave factor.  This means a model quantized
at length L is NOT valid for any other length.

Strategy: produce one quantized checkpoint per supported input length.

Pass --seq_len <length> on the command line, or omit for the default (1048578).
To produce all length variants in one run, pass --seq_len all.
"""

import os
import sys
import csv
import torch
import pandas as pd
import numpy as np
from ag_arch_quant import AlphaGenome
from torch.ao.quantization import get_default_qconfig, QConfigMapping, quantize_fx
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---- Config ----

ALL_RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, "full"]

# Supported input lengths.  Each produces a separate quantized checkpoint.
# All are (N * 131072) + 2 to preserve the encoder's stride structure.
SUPPORTED_SEQ_LENS = {
    "1Mb":    1_048_576,
    "500kb":    524_288,
    "4kb":        4_096,
}

# Organism indices for calibration (0 = human, 1 = mouse)
ALL_ORGANISMS = [0, 1]

NUM_ORGANISMS = 2   # must match the checkpoint
NUM_CALIB_SEQS = 16  # total calibration sequences (split evenly across organisms)

KEY_REPLACEMENTS = [
    ("proj.weight",                "proj.layer.weight"),
    ("linear_pair.weight",         "linear_pair.layer.weight"),
    ("linear_pair.bias",           "linear_pair.layer.bias"),
    ("linear_pos_features.weight", "linear_pos_features.layer.weight"),
    ("linear_pos_features.bias",   "linear_pos_features.layer.bias"),
    ("linear_y_q.weight",          "linear_y_q.layer.weight"),
    ("linear_y_k.weight",          "linear_y_k.layer.weight"),
    ("linear_q.weight",            "linear_q.layer.weight"),
    ("linear_k.weight",            "linear_k.layer.weight"),
    ("linear_v.weight",            "linear_v.layer.weight"),
    ("linear_v.bias",              "linear_v.layer.bias"),
    ("linear1.weight",             "linear1.layer.weight"),
    ("linear1.bias",               "linear1.layer.bias"),
    ("linear2.weight",             "linear2.layer.weight"),
    ("linear2.bias",               "linear2.layer.bias"),
]


# ---- Weight loading ----

def remap_state_dict(state_dict: dict) -> dict:
    """Drop LoRA keys and remap layer names to current arch."""
    new_sd = {}
    for key, value in state_dict.items():
        if ".lora." in key:
            continue
        new_key = key
        for old, new in KEY_REPLACEMENTS:
            new_key = new_key.replace(old, new)
        new_sd[new_key] = value
    return new_sd


def initialize_model(rank, device="cpu"):
    """Build AlphaGenome, load checkpoint, return eval model on device."""
    model = AlphaGenome(k=rank, num_organisms=NUM_ORGANISMS)
    ckpt = f"ag_lora_weights/ag_lora_lr{rank}.pth"
    state_dict = torch.load(ckpt, weights_only=True)
    model.load_state_dict(remap_state_dict(state_dict), strict=True)
    model.eval()
    return model.to(device)


# ---- Dataset ----

class SeqDataset(torch.utils.data.Dataset):
    """
    Generates (one_hot, organism_index) pairs.
    Organisms alternate: 0, 1, 0, 1, … so calibration covers both equally.
    """

    def __init__(self, num_seqs: int, seq_len: int):
        self.num_seqs = num_seqs
        self.seq_len  = seq_len

    def __len__(self):
        return self.num_seqs

    def __getitem__(self, idx):
        rand_idx = np.random.randint(0, 4, size=(self.seq_len,))
        one_hot  = np.zeros((4, self.seq_len), dtype=np.float32)
        one_hot[rand_idx, np.arange(self.seq_len)] = 1.0
        one_hot  = np.ascontiguousarray(one_hot.T)  # (4, S) → (S, 4) for model input
        organism = idx % NUM_ORGANISMS   # alternate 0, 1, 0, 1, …
        return torch.from_numpy(one_hot), torch.tensor(organism, dtype=torch.long)


def make_loader(num_seqs: int, seq_len: int) -> DataLoader:
    return DataLoader(
        SeqDataset(num_seqs, seq_len),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )


# ---- Parameter counting ----

def get_model_size(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_linear_params(model: torch.nn.Module) -> int:
    return sum(
        p.numel()
        for m in model.modules()
        if isinstance(m, torch.nn.Linear)
        for p in m.parameters()
    )


# ---- Quantization ----

class _ForwardWrapper(torch.nn.Module):
    """Thin wrapper exposing forward_for_quantization as the traced entry point."""
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, dna_sequence: torch.Tensor, organism_index: torch.Tensor):
        return self.model.forward_for_quantization(dna_sequence, organism_index)



def quantize_and_save(
    model: torch.nn.Module,
    outfile: str,
    rank,
    seq_len: int,
    seq_len_name: str,
    device: str = "cpu",
):
    """
    Quantize model for a fixed seq_len with multi-organism calibration,
    evaluate quality metrics, save checkpoint and TSV row.

    One quantized file is produced per (rank, seq_len) combination.
    The same file is valid for both organisms at that sequence length.
    """
    model = model.to(device).eval()
    # Freeze all input-length-dependent computations (Pool1d padding,
    # RoPE cos/sin tables, SequenceToPairBlock positional encodings)
    # so FX traces a fully static graph with no symbolic shape arithmetic.
    model.prepare_for_quantization(seq_len)
    wrapped = _ForwardWrapper(model).to(device).eval()

    # --- Calibration ---
    calib_loader = make_loader(NUM_CALIB_SEQS, seq_len)
    sample_seq, sample_org = next(iter(calib_loader))
    sample_seq = sample_seq.to(device)
    sample_org = sample_org.to(device)

    # Warm-up run (also verifies the model works at this length)
    with torch.no_grad():
        _ = wrapped(sample_seq, sample_org)

    qconfig         = get_default_qconfig("fbgemm")
    qconfig_mapping = QConfigMapping().set_global(qconfig)

    # All shape-dependent computations are frozen as constant buffers by
    # prepare_for_quantization(), so standard prepare_fx traces cleanly.
    prepared_model = quantize_fx.prepare_fx(
        wrapped, qconfig_mapping, (sample_seq, sample_org)
    )

    print(f"  Calibrating (seq_len={seq_len}, {NUM_CALIB_SEQS} seqs, "
          f"organisms alternating 0/1) ...")
    with torch.no_grad():
        for seq, org in tqdm(calib_loader, desc=f"  Calib rank={rank} len={seq_len_name}"):
            prepared_model(seq.to(device), org.to(device))

    model_quantized = quantize_fx.convert_fx(prepared_model)

    # --- Evaluation ---
    print(f"  Evaluating quantized model ...")
    eval_loader = make_loader(NUM_CALIB_SEQS, seq_len)
    rmaes, pcors, scors, maes = [], [], [], []

    # Evaluate separately per organism so metrics are organism-aware
    with torch.no_grad():
        for seq, org in eval_loader:
            seq = seq.to(device)
            org = org.to(device)

            # Flatten all head outputs into a single vector for correlation metrics
            raw_y    = wrapped(seq, org)
            raw_yhat = model_quantized(seq, org)

            y    = torch.cat([v.flatten() for v in _iter_tensors(raw_y)])
            yhat = torch.cat([v.flatten() for v in _iter_tensors(raw_yhat)])

            corr = torch.corrcoef(torch.stack([y, yhat]))
            pcors.append(float(corr[0, 1]))

            scorr, _ = spearmanr(y.cpu().numpy(), yhat.cpu().numpy())
            scors.append(float(scorr))

            mae = torch.mean(torch.abs(yhat - y))
            maes.append(float(mae))
            rmaes.append(float(mae / (torch.mean(torch.abs(y)) + 1e-8)))

    print(f"  Quantization Metrics — rank={rank}, seq_len={seq_len_name}:")
    print(f"    Relative MAE : {np.nanmean(rmaes):.4f} ± {np.std(rmaes):.4f}")
    print(f"    MAE          : {np.nanmean(maes):.4f}  ± {np.std(maes):.4f}")
    print(f"    Pearson      : {np.nanmean(pcors):.4f} ± {np.std(pcors):.4f}")
    print(f"    Spearman     : {np.nanmean(scors):.4f} ± {np.std(scors):.4f}")

    metrics = {
        "rank":          rank,
        "seq_len":       seq_len_name,
        "seq_len_bp":    seq_len,
        "rmae_mean":     round(np.nanmean(rmaes), 4),
        "rmae_std":      round(np.std(rmaes),     4),
        "mae_mean":      round(np.nanmean(maes),  4),
        "mae_std":       round(np.std(maes),      4),
        "pearson_mean":  round(np.nanmean(pcors), 4),
        "pearson_std":   round(np.std(pcors),     4),
        "spearman_mean": round(np.nanmean(scors), 4),
        "spearman_std":  round(np.std(scors),     4),
    }

    tsv_path   = "ag_quantization_metrics.tsv"
    file_exists = os.path.isfile(tsv_path)
    with open(tsv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys(), delimiter="\t")
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
    print(f"  Metrics appended to {tsv_path}")

    torch.save(model_quantized.state_dict(), outfile)
    print(f"  Saved quantized weights: {outfile}")
    return model_quantized


def _iter_tensors(obj):
    """Recursively yield all tensors from a nested dict/list/tuple."""
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_tensors(v)


# ---- Main entry point ----

def save_model_sizes(
    rank_index: int,
    quant: bool,
    seq_len_name: str = "1Mb",
    outpath: str = "benchmark_ag_model_sizes.tsv",
):
    """
    For the given rank_index (1–11) and seq_len_name:
      quant=False  → report param counts, append to TSV (seq_len irrelevant)
      quant=True   → quantize at that seq_len, save weights + metrics
    """
    if not (1 <= rank_index <= len(ALL_RANKS)):
        raise ValueError(f"rank_index must be between 1 and {len(ALL_RANKS)}, got {rank_index}")
    if seq_len_name not in SUPPORTED_SEQ_LENS:
        raise ValueError(f"seq_len must be one of {list(SUPPORTED_SEQ_LENS)}, got {seq_len_name!r}")

    rank    = ALL_RANKS[rank_index - 1]
    seq_len = SUPPORTED_SEQ_LENS[seq_len_name]
    device  = "cpu"

    print(f"\n[INFO] rank={rank}, quant={quant}, seq_len={seq_len_name} ({seq_len:,} bp)")
    model = initialize_model(rank, device=device)

    if quant:
        # One checkpoint per (rank, seq_len) — valid for both organisms at that length
        outfile = f"ag_lora_weights/ag_lora_lr{rank}_len{seq_len_name}_quant.pth"
        print(f"[INFO] Quantizing AlphaGenome rank={rank}, seq_len={seq_len_name} ...")
        quantize_and_save(
            model, outfile=outfile, rank=rank,
            seq_len=seq_len, seq_len_name=seq_len_name, device=device,
        )
        print("[INFO] Quantization complete.")

    else:
        # Param counts are independent of seq_len and organism — report once
        n_params        = get_model_size(model)
        n_linear_params = get_linear_params(model)

        row = {
            "model":           "alphagenome",
            "rank":            rank,
            "quant":           quant,
            "n_params":        n_params,
            "n_linear_params": n_linear_params,
        }
        print(f"[INFO] n_params={n_params:,}  n_linear_params={n_linear_params:,}")

        df = pd.DataFrame([row])
        if os.path.exists(outpath):
            df.to_csv(outpath, sep="\t", mode="a", index=False, header=False)
        else:
            df.to_csv(outpath, sep="\t", index=False)
        print(f"[INFO] Results appended to {outpath}")
        return df


def main():
    """
    Usage:
      python ag_quantize.py [rank_index] [quant] [seq_len]

    rank_index : integer 1–11  (maps to 1,2,4,8,16,32,64,128,256,512,full)
                 omit or "all" to run all ranks
    quant      : True | False  (default: False)
    seq_len    : 1Mb | 500kb | 4kb | all  (default: 1Mb)
                 "all" produces one quantized checkpoint per supported length.
                 Ignored when quant=False (param counts don't depend on length).

    NOTE on seq_len variants
    ------------------------
    FX quantization bakes sequence length into the graph, so a model quantized
    at length "1Mb" (1,048,576 bp) cannot be used for "500kb" (524,288 bp) or
    "4kb" (4,096 bp) inputs.  Use seq_len all to produce all variants.
    Organism (human=0, mouse=1) is handled by a single model per length because
    organism_index only selects weight slices — the graph is identical for both.

    Examples:
      python ag_quantize.py                        # all ranks, quant=False
      python ag_quantize.py 3                      # rank=4, quant=False
      python ag_quantize.py all True 1Mb           # all ranks, quant=True, 1Mb
      python ag_quantize.py 5 True all             # rank=16, quant=True, all lengths
      python ag_quantize.py all True all           # everything
    """
    argv = sys.argv[1:]

    # rank_index
    if not argv or argv[0].lower() == "all":
        rank_indices = list(range(1, len(ALL_RANKS) + 1))
    else:
        try:
            rank_indices = [int(argv[0])]
        except ValueError:
            print("Error: rank_index must be an integer 1–11, 'all', or omitted.")
            sys.exit(1)

    # quant
    quant = argv[1].lower() == "true" if len(argv) > 1 else False

    # seq_len
    seq_len_arg = argv[2].lower() if len(argv) > 2 else "1Mb"
    if seq_len_arg == "all":
        seq_len_names = list(SUPPORTED_SEQ_LENS.keys())
    elif seq_len_arg in SUPPORTED_SEQ_LENS:
        seq_len_names = [seq_len_arg]
    else:
        print(f"Error: seq_len must be one of {list(SUPPORTED_SEQ_LENS)} or 'all'.")
        sys.exit(1)

    print(f"\n[RUNNING] ranks={[ALL_RANKS[i-1] for i in rank_indices]}, "
          f"quant={quant}, seq_lens={seq_len_names}")

    for rank_index in rank_indices:
        if quant:
            for seq_len_name in seq_len_names:
                save_model_sizes(rank_index=rank_index, quant=True, seq_len_name=seq_len_name)
        else:
            # param counts don't depend on seq_len — run once per rank
            save_model_sizes(rank_index=rank_index, quant=False)


if __name__ == "__main__":
    main()