#!/usr/bin/env python
import time
import json


from tqdm import tqdm
from scipy.stats import pearsonr

from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch
import os
import grelu.resources
import time
from alphagenome_pytorch import AlphaGenome
from torch import Tensor, nn
from scipy.sparse.linalg import svds




import copy


class SeqDataset(Dataset):
    def __init__(self, sequences):
        # sequences: numpy array of shape [N, 4, L] or [N, L, 4]
        self.sequences = torch.tensor(sequences, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

def aggregate_predictions(pred_list):
    """
    Concatenate a list of per-sample prediction dicts along batch dim.
    Handles nested dicts and dicts-of-dicts with tensor values.
    """
    def cat_tensors(tensors):
        return torch.cat(tensors, dim=0)

    def merge(vals):
        # vals: list of values from each sample, one per batch item
        if isinstance(vals[0], torch.Tensor):
            return cat_tensors(vals)
        elif isinstance(vals[0], dict):
            return {k: merge([v[k] for v in vals]) for k in vals[0]}
        else:
            return vals  # fallback: just keep as list

    keys = pred_list[0].keys()
    return {k: merge([p[k] for p in pred_list]) for k in keys}


def pearson_vectorized(p1, p2):
    """
    Compute per-track Pearson r without a Python loop over T.

    p1, p2: (B, T, C) float32 arrays
    Returns: (T,) array of Pearson r values

    Strategy: treat the B*C values for each track as the population.
    Transpose to (T, B*C), mean-centre, then compute r in one shot.
    """
    # (B, T, C) -> (T, B*C)
    T = p1.shape[1]
    a = p1.transpose(1, 0, 2).reshape(T, -1).astype(np.float64)
    b = p2.transpose(1, 0, 2).reshape(T, -1).astype(np.float64)

    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)

    num = (a * b).sum(axis=1)
    denom = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    # avoid division by zero for constant tracks
    denom = np.where(denom == 0, np.nan, denom)
    return num / denom  # (T,)


def benchmark_model(model, loader, baseline_preds_batched, linear_rank, input_shape, device="cuda", runs=5, outfile="borzoi_benchmark_results.json"):
    save_dir = "./ag_lora_weights/"
    os.makedirs(save_dir, exist_ok=True)

    new_model = convert_to_lora_proper(model, linear_rank=linear_rank)
    new_model.eval()
    new_model.to(device)

    if linear_rank == 1000000000:
        linear_rank = "full"

    lora_fname = f"{save_dir}ag_lora_lr{linear_rank}.pth"
    state_dict = new_model.state_dict()
    new_state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}
    torch.save(new_state_dict, lora_fname)
    print(f"Saved LoRA weights: {lora_fname}")

    total_params = sum(p.numel() for p in new_model.parameters() if p.requires_grad)

    def get_comparable_heads(pred_dict):
        heads = []
        for head, val in pred_dict.items():
            if isinstance(val, torch.Tensor):
                heads.append((head, None))
            elif isinstance(val, dict):
                for k, v in val.items():
                    if isinstance(v, torch.Tensor):
                        heads.append((head, k))
        return heads

    comparable_heads = get_comparable_heads(baseline_preds_batched)

    accum = {}
    num_batches = 0
    organism_index = torch.tensor([0], dtype=torch.long)

    for batch_idx, batch_seqs in tqdm(enumerate(loader), desc=f"Evaluating LoRA rank {linear_rank}"):
        batch_seqs = batch_seqs.float().to(device)

        with torch.no_grad():
            preds2_dict = new_model(batch_seqs, organism_index)

        batch_seqs = batch_seqs.cpu()

        for (head, res) in comparable_heads:
            if res is None:
                p1 = baseline_preds_batched[head][batch_idx:batch_idx+1]
                p2 = preds2_dict[head]
            else:
                p1 = baseline_preds_batched[head][res][batch_idx:batch_idx+1]
                p2 = preds2_dict[head][res]

            if isinstance(p1, torch.Tensor):
                p1 = p1.numpy()
            if isinstance(p2, torch.Tensor):
                p2 = p2.detach().cpu().numpy()

            # Flatten to (B, T, C) — handles 3D, 4D, etc.
            p1 = p1.reshape(p1.shape[0], p1.shape[1], -1)
            p2 = p2.reshape(p2.shape[0], p2.shape[1], -1)

            T = p1.shape[1]
            key = (head, res)
            if key not in accum:
                accum[key] = {"pearson": np.zeros(T), "mse": np.zeros(T)}

            # Vectorized: no Python loop over T
            accum[key]["pearson"] += pearson_vectorized(p1, p2)
            accum[key]["mse"] += ((p1 - p2) ** 2).mean(axis=(0, 2))  # mean over B and C, shape (T,)

        num_batches += 1

    print(f"Completed evaluation for LoRA rank {linear_rank} over {num_batches} batches.")

    # Summarize — track-weighted global mean (extend) and per-head mean
    per_head_results = {}
    all_pearsons = []  # one value per track across all heads → track-weighted global mean
    all_mses = []

    for (head, res), vals in accum.items():
        per_track_pearsons = vals["pearson"] / num_batches  # (T,)
        per_track_mses = vals["mse"] / num_batches
        label = f"{head}_{res}" if res is not None else head
        per_head_results[label] = {
            "mean_pearson": float(np.nanmean(per_track_pearsons)),
            "mean_mse": float(np.nanmean(per_track_mses)),
        }
        all_pearsons.extend(per_track_pearsons.tolist())  # track-weighted
        all_mses.extend(per_track_mses.tolist())

    result = {
        "linear_rank": linear_rank,
        "n_blocks": "mha",
        "total_params": total_params,
        "mean_pearson_overall": float(np.nanmean(all_pearsons)),
        "mean_mse_overall": float(np.nanmean(all_mses)),
        "per_head": per_head_results,
    }

    # Append result to log file (fixed: was reading/writing inverted)
    if os.path.exists(outfile):
        with open(outfile, "r") as f:
            existing = json.load(f)
        existing.append(result)
        with open(outfile, "w") as f:
            json.dump(existing, f, indent=4)
    else:
        with open(outfile, "w") as f:
            json.dump([result], f, indent=4)

    print(f"Logged benchmark: {result}")




def convert_to_lora_proper(model, linear_rank=16):
    new_model = copy.deepcopy(model)  # clone entire model

    def replace_layers(module):
        for name, child in module.named_children():
            # if name == "mha":
            #     continue  
            replace_layers(child)
            if isinstance(child, nn.Linear):
                max_k = min(child.in_features, child.out_features)
                if linear_rank < max_k:
                    setattr(module, name, LoraLinear(child, k=linear_rank))
            elif isinstance(child, nn.Conv1d):
                setattr(module, name, WrappedConv1d(child))

    replace_layers(new_model)
    return new_model


class LoraLinear(nn.Module):
    def __init__(self, layer: nn.Linear, k: int = 16):
        super().__init__()
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        
        self.w = layer.weight.detach().cpu().numpy()
        self.b = None if layer.bias is None else layer.bias.detach().cpu().numpy()
        self.out_dim, self.in_dim = self.w.shape

        # Clamp k to valid range
        max_k = min(self.in_dim, self.out_dim) - 1  # svds requires k < min(A.shape)
        if k > max_k:
            print(f"[LoraLinear] Requested rank {k} too large, using k={max_k}")
            k = max_k
        if k <= 0:
            raise ValueError(f"[LoraLinear] Cannot apply LoRA: k={k} <= 0 for layer {layer}")
        self.k = k

        # Initialize weights
        w11, w12, b = self.make_layer()

        self.loraw11 = nn.Linear(self.in_dim, self.k, bias=False)
        if b is not None:
            self.loraw12 = nn.Linear(self.k, self.out_dim)
        else:
            self.loraw12 = nn.Linear(self.k, self.out_dim, bias=False)

        self.loraw11.weight.data = w11.clone()
        self.loraw12.weight.data = w12.clone()

        if b is not None:
            self.loraw12.bias.data = b.clone()

        del self.w
        del self.b

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
        x = self.loraw11(x)
        x = self.loraw12(x)
        return x


class WrappedConv1d(nn.Module):
    def __init__(self, layer: nn.Conv1d):
        super().__init__()
        assert isinstance(layer, nn.Conv1d), "Expected nn.Conv1d layer"
        self.skip_lora = True
        self.layer = layer
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def main():
    model = AlphaGenome.from_pretrained("model_all_folds.safetensors")

    # import bed file, pad range of start and end from 4096 to 1048576
    test_df = pd.read_csv("GRCh38_cCREs_4kb.bed", sep="\t", header=None, names=["chrom", "start", "end"])
    test_df["start"] = test_df["start"] - (1048576 - 4096) // 2
    test_df["end"] = test_df["end"] + (1048576 - 4096) // 2

    test_df = test_df[test_df['start'] >= 1].reset_index(drop=True)
    #get first 8 sequences
    test_df = test_df.iloc[:8]
    input_seqs = grelu.sequence.format.convert_input_type(
        test_df,
        output_type="strings",
        genome="hg38"
    )
    #make onehot encoding of input_seqs, reshape to [b, 4, L]
    input_seqs = grelu.sequence.format.convert_input_type(
        input_seqs,
        output_type="one_hot",   
        genome="hg38"
    )   

    #reshape from [b, 4, L] to [b, L, 4]
    input_seqs = np.transpose(input_seqs, (0, 2, 1))
    dataset = SeqDataset(input_seqs)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    baseline_preds = []
    model.eval()
    model.to("cpu")
    organism_index = torch.tensor([0], dtype=torch.long).to("cpu")
    for batch_seqs in tqdm(loader, desc="Baseline predictions"):
        batch_seqs = batch_seqs.float().to("cpu") # ensure correct dtype
        with torch.no_grad():
            preds = model(batch_seqs, organism_index)
        batch_seqs = batch_seqs.cpu()
        baseline_preds.append(preds)
    baseline_preds_batched = aggregate_predictions(baseline_preds)

    # Example sweep
    linear_ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 1000000000] # rename 1000000000 to full
    input_shape = (1, 1048576, 4)  # batch=4

    for lr in linear_ranks:
        benchmark_model(model, loader, baseline_preds_batched, linear_rank=lr, input_shape=input_shape, device="cpu", runs=1, outfile="ag_benchmark_results_mha_2.json")

if __name__ == "__main__":
    main()