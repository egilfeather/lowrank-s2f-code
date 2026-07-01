#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import gc
import pickle
import tempfile
import threading
import queue
import time
from typing import Literal, Optional

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
from tqdm import tqdm

from borzoi_lora_arch_mha import BorzoiModel, EnformerModel
from ag_arch import AlphaGenome
from alphagenome_pytorch import AlphaGenome as AlphaGenomePT
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from sei_lora.dataloaders import VariantDataset, SeqDataLoader, SeqDataset, VariantDataLoader
import seimodel as sm
import seillra as sl

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
RANKS       = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
N_SAMPLES   = 200
SEED        = 42
BED         = "./GRCh38_cCREs_4kb.bed"
FASTA       = "../resources/hg38_UCSC.fa"
OUTPUT_PKL  = "ag_human_pearson_corrs_shuffle.pkl"
MODEL_NAME  = "ag"
DEVICE      = torch.device("cuda")

MODE_KEYS = {
    "atac":              {"type": "multi_res", "resolutions": [1, 128]},
    "dnase":             {"type": "multi_res", "resolutions": [1, 128]},
    "procap":            {"type": "multi_res", "resolutions": [1, 128]},
    "cage":              {"type": "multi_res", "resolutions": [1, 128]},
    "rna_seq":           {"type": "multi_res", "resolutions": [1, 128]},
    "chip_tf":           {"type": "multi_res", "resolutions": [128]},
    "chip_histone":      {"type": "multi_res", "resolutions": [128]},
    "contact_maps":      {"type": "raw"},
    "splice_sites":      {"type": "subkey", "subkeys": ["logits", "probs"]},
    "splice_site_usage": {"type": "subkey", "subkeys": ["logits", "predictions"]},
}

MODEL_PARAMS = {
    "ag": {
        "input_len":   1048576,
        "bin_length":  1,
        "model_class": AlphaGenome,
        "kwargs":      {},
    },
    "borzoi": {
        "input_len":  524288,
        "bin_length": 32,
        "model_class": BorzoiModel,
        "kwargs": {"n_tasks": 7611, "crop_len": 5120,
                   "final_act_func": "softplus", "final_pool_func": None},
    },
    "enformer": {
        "input_len":  196608,
        "bin_length": 128,
        "model_class": EnformerModel,
        "kwargs": {"n_tasks": 5313, "crop_len": 320,
                   "final_act_func": "softplus", "final_pool_func": None},
    },
}

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
def free_gpu():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def print_gpu_mem(label=""):
    alloc   = torch.cuda.memory_allocated()  / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[GPU] {label:30s}  alloc={alloc:.2f}GB  reserved={reserved:.2f}GB")


def initialize_model(model_name, rank=None, full=False):
    """Load one model, all grads disabled."""
    dev = DEVICE

    if full:
        if model_name == "ag":
            model = AlphaGenomePT.from_pretrained("model_all_folds.safetensors")
        elif model_name == "borzoi":
            model = torch.load("./full_models/borzoi_human_rep0.pt")
        elif model_name == "enformer":
            model = torch.load("./full_models/enformer_human.pt")
        elif model_name == "sei":
            model = SeiWrapper(k=None, ft=None, projection=False, mode="sequence", device=dev)
        else:
            model = None
    else:
        if model_name not in MODEL_PARAMS:
            model = sl.SeiLoraWrapper(k=rank, projection=False, mode="sequence", device=dev)
        else:
            model = MODEL_PARAMS[model_name]["model_class"](
                k=rank, **MODEL_PARAMS[model_name]["kwargs"]
            )
            state_dict = torch.load(
                f'{model_name}_lora_weights/{model_name}_lora_lr{rank}.pth',
                weights_only=True
            )
            new_state_dict = {}
            for key, value in state_dict.items():
                if ".lora." in key:
                    continue
                new_key = key
                for old, new in [
                    ("proj.weight",                  "proj.layer.weight"),
                    ("linear_pair.weight",            "linear_pair.layer.weight"),
                    ("linear_pair.bias",              "linear_pair.layer.bias"),
                    ("linear_pos_features.weight",    "linear_pos_features.layer.weight"),
                    ("linear_pos_features.bias",      "linear_pos_features.layer.bias"),
                    ("linear_y_q.weight",             "linear_y_q.layer.weight"),
                    ("linear_y_k.weight",             "linear_y_k.layer.weight"),
                    ("linear_q.weight",               "linear_q.layer.weight"),
                    ("linear_k.weight",               "linear_k.layer.weight"),
                    ("linear_v.weight",               "linear_v.layer.weight"),
                    ("linear_v.bias",                 "linear_v.layer.bias"),
                    ("linear1.weight",                "linear1.layer.weight"),
                    ("linear1.bias",                  "linear1.layer.bias"),
                    ("linear2.weight",                "linear2.layer.weight"),
                    ("linear2.bias",                  "linear2.layer.bias"),
                ]:
                    new_key = new_key.replace(old, new)
                new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict, strict=True)

    # disable all grads — saves ~memory and prevents autograd graph buildup
    for p in model.parameters():
        p.requires_grad_(False)

    model.eval()
    return model


def extract_arrays(out):
    """Flatten one sequence's model output into {key: float16 numpy array}."""
    arrays = {}
    for mode, cfg in MODE_KEYS.items():
        if mode not in out:
            continue
        if cfg["type"] == "multi_res":
            for res in cfg["resolutions"]:
                if res in out[mode]:
                    arrays[f"{mode}_res{res}"] = (
                        out[mode][res].cpu().to(torch.float16).numpy().flatten()
                    )
        elif cfg["type"] == "raw":
            arrays[mode] = out[mode].cpu().to(torch.float16).numpy().flatten()
        elif cfg["type"] == "subkey":
            for sk in cfg["subkeys"]:
                if sk in out[mode]:
                    arrays[f"{mode}_{sk}"] = (
                        out[mode][sk].cpu().to(torch.float16).numpy().flatten()
                    )
    return arrays


# ----------------------------------------------------------------
# Async pickle writer
# Runs in a background thread so GPU never waits for disk I/O
# ----------------------------------------------------------------
class AsyncPickleWriter:
    def __init__(self, path):
        self.path = path
        self._q   = queue.Queue(maxsize=8)   # backpressure: max 8 items buffered
        self._t   = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def _worker(self):
        with open(self.path, "ab") as f:
            while True:
                item = self._q.get()
                if item is None:        # sentinel
                    break
                pickle.dump(item, f)
                f.flush()
                self._q.task_done()

    def write(self, obj):
        self._q.put(obj)

    def close(self):
        self._q.put(None)
        self._t.join()


def make_loader(dataset):
    g = torch.Generator()
    g.manual_seed(SEED)
    return SeqDataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        n_samples=N_SAMPLES,
        generator=g,
    )


def run_inference_to_file(model, dataset, desc, tmp_path):
    """
    GPU inference with async background pickle writes.
    GPU extracts + transfers while disk writer runs in parallel.
    """
    organism_index = torch.tensor([0], dtype=torch.long, device=DEVICE)
    writer = AsyncPickleWriter(tmp_path)
    loader = make_loader(dataset)

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            data, _ = batch
            data = data.to(DEVICE, non_blocking=True).permute(0, 2, 1)
            out  = model(data, organism_index, resolutions=[1, 128])

            # extract to CPU float16 immediately — frees GPU memory
            arrays = extract_arrays(out)
            del out, data

            # hand off to background thread; GPU continues to next batch
            writer.write(arrays)

    writer.close()
    free_gpu()


def iter_pickle_file(path):
    """Lazy iterator over sequentially-written pickle records."""
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


# ----------------------------------------------------------------
# Correlation computation
# ----------------------------------------------------------------
def compute_correlations(full_tmp, lora_tmp, output_keys):
    per_seq    = {k: [] for k in output_keys}
    lora_means = {k: [] for k in output_keys}
    full_means = {k: [] for k in output_keys}

    for full_d, lora_d in zip(iter_pickle_file(full_tmp), iter_pickle_file(lora_tmp)):
        for k in output_keys:
            fa = full_d[k].astype(np.float32)
            la = lora_d[k].astype(np.float32)

            if fa.std() > 0 and la.std() > 0:
                per_seq[k].append(np.corrcoef(fa, la)[0, 1])
            else:
                per_seq[k].append(np.nan)

            full_means[k].append(fa.mean())
            lora_means[k].append(la.mean())

    across = {}
    for k in output_keys:
        fm = np.array(full_means[k])
        lm = np.array(lora_means[k])
        if fm.std() > 0 and lm.std() > 0:
            across[k], _ = spearmanr(fm, lm)
        else:
            across[k] = np.nan

    return per_seq, across


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def calculate_correlation(model_name):
    input_len = MODEL_PARAMS[model_name]["input_len"] if model_name in MODEL_PARAMS else 4096
    dataset   = SeqDataset(
        file_path=BED, scores_path=False, fasta_path=FASTA,
        window_size=input_len, mode="test", test_chrom=["chr8", "chr9"]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        full_tmp = os.path.join(tmpdir, "full.pkl")

        # ---- full model ----
        print_gpu_mem("before full model load")
        full_model = initialize_model(model_name, full=True)
        full_model.to(DEVICE)
        print_gpu_mem("after full model load")

        run_inference_to_file(full_model, dataset, "Full model", full_tmp)
        del full_model
        free_gpu()
        print_gpu_mem("after full model deleted")

        # get output keys from first record
        output_keys = list(next(iter_pickle_file(full_tmp)).keys())
        print(f"Output keys: {output_keys}")

        # ---- lora ranks ----
        per_seq_corrs    = {}
        across_seq_corrs = {}

        for r in RANKS:
            lora_tmp = os.path.join(tmpdir, f"rank_{r}.pkl")

            print_gpu_mem(f"before rank {r} load")
            lora_model = initialize_model(model_name, rank=r, full=False)
            lora_model.to(DEVICE)
            print_gpu_mem(f"after rank {r} load")

            run_inference_to_file(lora_model, dataset, f"LoRA rank {r}", lora_tmp)
            del lora_model
            free_gpu()
            print_gpu_mem(f"after rank {r} deleted")

            per_seq, across = compute_correlations(full_tmp, lora_tmp, output_keys)
            per_seq_corrs[r]    = per_seq
            across_seq_corrs[r] = across

            print(f"\nRank {r} summary:")
            print(f"  Per-seq Pearson:     { {k: f'{np.nanmean(v):.4f}' for k, v in per_seq.items()} }")
            print(f"  Across-seq Spearman: { {k: f'{across[k]:.4f}' for k in output_keys} }")

            os.remove(lora_tmp)

        # tmpdir + full.pkl cleaned up automatically

    return {"per_seq": per_seq_corrs, "across_seq": across_seq_corrs}


def main():
    results = calculate_correlation(MODEL_NAME)
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved to {OUTPUT_PKL}")


if __name__ == "__main__":
    main()