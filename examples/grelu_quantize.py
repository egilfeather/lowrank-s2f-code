#!/usr/bin/env python3
import os, sys
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from borzoi_lora_arch_mha import BorzoiModel, EnformerModel
import grelu.resources
from torch.ao.quantization import get_default_qconfig, QConfigMapping, quantize_fx
from scipy.stats import spearmanr
from pathlib import Path
import hashlib
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import time

MODEL_PARAMS = {
    "borzoi": {
            "input_len": 524288,
            "bin_length": 32,
            "model_class": BorzoiModel,
            "kwargs": {"n_tasks": 7611, "crop_len": 5120, "final_act_func": "softplus", "final_pool_func": None},
            "gm12878": [1288, 1345],
            "microglia": [2052],
            "smc" : [2097], 
            "spi1": [2405],
            "pai_mask": "exon", #"gene",
            "pai_metric": "SAR",

    },
    "enformer": {
            "input_len": 196608,
            "bin_length": 128,
            "model_class": EnformerModel,
            "kwargs": {"n_tasks": 5313, "crop_len": 320, "final_act_func": "softplus", "final_pool_func": None},
            "gm12878": [12, 69],
            "microglia": [41, 131, 392, 508, 517],
            "smc" : [82], 
            "spi1": [907],
            "pai_mask": "all",
            "pai_metric": "SAD",
    }

    }


def initialize_models(k_l: int, quant: bool, model_name = "borzoi", full = False):

    if quant == True:
        dev = "cpu"
    else:
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if full != True:

        model = MODEL_PARAMS[model_name]["model_class"](k_l =k_l, device = dev, **MODEL_PARAMS[model_name]["kwargs"])
        # model = BorzoiModel(k_l =k_l, k_c = k_c, n_tasks = 7611, crop_len=5120, final_act_func="softplus", final_pool_func=None)
        state_dict = torch.load(f'{model_name}_lora_weights/{model_name}_lora_lr{k_l}.pth', weights_only=True)

        model.load_state_dict(state_dict, strict = True)
    else:
        if model_name == "borzoi":
            model = torch.load("./full_models/borzoi_human_rep0.pt")
        elif model_name == "enformer":
            model = torch.load("./full_models/enformer_human.pt")
        else:
            model = None

  
    return model

def one_hot_encode_dna(seqs):
    """One-hot encode a list of DNA sequences (A,C,G,T)."""
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    N = len(seqs)
    L = len(seqs[0])
    
    # Create tensor of zeros
    one_hot = torch.zeros((N, 4, L), dtype=torch.float32)
    
    # Fill positions
    for i, seq in enumerate(seqs):
        for j, base in enumerate(seq):
            if base in base_to_idx:  # ignore Ns or ambiguous bases
                one_hot[i, base_to_idx[base], j] = 1.0
    return one_hot

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, input_len, num_seqs):
        self.input_len = input_len
        self.num_seqs = num_seqs
        self.bases = 4
    
    def __len__(self):
        return self.num_seqs
    def __getitem__(self, idx):

        rand_indices = np.random.randint(0, self.bases, size=(1, self.input_len))
        one_hot_seqs = np.zeros((1, self.bases, self.input_len), dtype=np.float32)
        one_hot_seqs[0, rand_indices, np.arange(self.input_len)] = 1.0
        embedding = torch.from_numpy(one_hot_seqs).to('cpu')
        return embedding



def quantize_and_save(model: torch.nn.Module, model_name: str, outfile: str, device="cpu", rank = None):
    """Quantize a LoRA Borzoi model and save weights + metrics."""
    seq_count = 16
    device = "cpu"
    seq_len = MODEL_PARAMS[model_name]["input_len"]

    dataset = SeqDataset(input_len=seq_len, num_seqs=seq_count)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print("loaded data")
    model = model.to(device)
    # model = model.model
    model.eval()

    sample_batch = next(iter(loader))
    sample_batch = sample_batch[0].to(device)
    _ = model(sample_batch)
    # --- FX Quantization ---
    qconfig = get_default_qconfig("fbgemm")
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    prepared_model = quantize_fx.prepare_fx(model, qconfig_mapping, sample_batch)
    print("prepping model")
#    with torch.no_grad():
#        for batch in tqdm(loader, desc="Prepping quant model"):
#            prepared_model(batch[0])
    model_quantized = quantize_fx.convert_fx(prepared_model)

    print("Evaluating quantized model ...")
    # --- Evaluate metrics ---
    dataset_2 = SeqDataset(input_len=seq_len, num_seqs=seq_count)
    loader_2 = DataLoader(dataset_2, batch_size=1, shuffle=False, num_workers=0)
    rmaes, pcors, scors, maes = [], [], [], []
    with torch.no_grad():
        for batch in loader_2:
            y = model(batch[0]).flatten()
            yhat = model_quantized(batch[0]).flatten()

            corr = torch.corrcoef(torch.stack([y, yhat]))
            pcors.append(float(corr[0,1]))

            scorr, _ = spearmanr(y.cpu().numpy(), yhat.cpu().numpy())
            scors.append(float(scorr))

            mae = torch.mean(torch.abs(yhat - y))
            maes.append(float(mae))
            rmaes.append(float(mae / torch.mean(torch.abs(y))))

    # Print metrics
    print(f"Quantization Metrics for {outfile}:")
    print(f"  Relative MAE: {np.nanmean(rmaes):.4f} ± {np.std(rmaes):.4f}")
    print(f"  MAE: {np.nanmean(maes):.4f} ± {np.std(maes):.4f}")
    print(f"  Pearson: {np.nanmean(pcors):.4f} ± {np.std(pcors):.4f}")
    print(f"  Spearman: {np.nanmean(scors):.4f} ± {np.std(scors):.4f}")

    metrics = {
        "rank": rank,
        "rmae_mean": round(np.nanmean(rmaes), 4),
        "rmae_std": round(np.std(rmaes), 4),
        "mae_mean": round(np.nanmean(maes), 4),
        "mae_std": round(np.std(maes), 4),
        "pearson_mean": round(np.nanmean(pcors), 4),
        "pearson_std": round(np.std(pcors), 4),
        "spearman_mean": round(np.nanmean(scors), 4),
        "spearman_std": round(np.std(scors), 4),
    }

    file_exists = os.path.isfile(f"{model_name}_quantization_metrics.tsv")
    with open(f"{model_name}_quantization_metrics.tsv", "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys(), delimiter="\t")
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

            print(f"Results saved to {model_name}_quantization_metrics.tsv")

    torch.save(model_quantized.state_dict(), outfile)
    print(f"Saved quantized weights: {outfile}")
    return model_quantized

def get_linear_params(model: torch.nn.Module):
    """Return total number of parameters in Linear layers recursively."""
    total = 0
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            total += sum(p.numel() for p in module.parameters())
        else:
            total += get_linear_params(module)
    return total

def get_model_size(model: torch.nn.Module):
    """Return total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())

def save_model_sizes(quant, name, outpath="benchmark_model_sizes.tsv", rank_index=None):
    """Generate TSV of model sizes for all ranks and configs."""
    all_ranks = [1,2,4,8,16,32,64,128,256,512, "full"]
    if rank_index is None or not (1 <= rank_index <= len(all_ranks)):
        raise ValueError(f"rank_index must be between 1 and {len(all_ranks)}")

    rank = all_ranks[rank_index - 1]
    if quant == True:
        device = "cpu"
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = []
    print(f"\n[INFO] Starting model: {name}, quant={quant}, rank={rank}")

    model = initialize_models(k_l=rank, quant=quant, model_name=name)

    if quant:
        outfile = f"{name}_lora_weights/{name}_lora_lr{rank}_quant.pth"
        print(f"[INFO] Quantizing {name} model rank {rank} ...")
        model = quantize_and_save(model, model_name=name, outfile=outfile, device=device, rank =rank)
        print("[INFO] Quantization complete.")
    else:
        n_params = get_model_size(model)
        n_linear_params = get_linear_params(model)
        results.append({
            "model": name,
            "rank": rank,
            "quant": quant,
            "n_params": n_params,
            "n_linear_params": n_linear_params
        })
        print(f"[INFO] Rank={rank}, quant={quant} -> n_params={n_params}")

        # Append or create results file
        df = pd.DataFrame(results)
        if os.path.exists(outpath):
            df.to_csv(outpath, sep="\t", mode="a", index=False, header=False)
        else:
            df.to_csv(outpath, sep="\t", index=False)
        print(f"[INFO] Results appended to {outpath}")
        return df


def main():
    """
    Usage:
      python grelu_quantize.py <rank_index> [model_name] [quant]
    Example:
      python grelu_quantize.py 3 borzoi False
      python grelu_quantize.py 5 enformer True
    """
    if len(sys.argv) < 2:
        print("Usage: python grelu_quantize.py <rank_index> [model_name] [quant]")
        sys.exit(1)

    try:
        rank_index = int(sys.argv[1])
    except ValueError:
        print("Error: rank_index must be an integer from 1 to 11.")
        sys.exit(1)

    model_name = sys.argv[2] if len(sys.argv) > 2 else "enformer"
    quant = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else True

    outpath = f"benchmark_{model_name}_model_sizes.tsv"

    print(f"\n[RUNNING] rank_index={rank_index}, model={model_name}, quant={quant}")
    save_model_sizes(quant=quant, name=model_name, outpath=outpath, rank_index=rank_index)


if __name__ == "__main__":
    main()


