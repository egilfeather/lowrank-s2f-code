#!/usr/bin/env python3
import os, sys
import time
from typing import Literal, Optional
import torch
import torch.nn as nn
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
import seimodel as sm
import seillra as sl
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from sei_lora.dataloaders import VariantDataset, SeqDataLoader, SeqDataset, VariantDataLoader
import pickle


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

class SeiWrapper(nn.Module):
    def __init__(self, k: int, ft: Optional[str] = None, projection: bool = True, mode: Literal["sequence", "variant"] = "sequence", device: str = "cpu"):
        super().__init__()
        self.device = device
        self.mode = mode
        self.projection = projection
        self.head = sm.get_sei_head().load_weights()
        self.trunk = sm.get_sei_trunk().load_weights()
        
        if self.projection:
            self.proj = sm.get_sei_projection().load_weights()
            self.proj.set_mode(mode)
        self.device = device

    def set_mode(self, mode):
        if mode != "sequence" and mode != "variant":
            print(f"Mode options are: \'sequence\' or \'variant\'. Keeping current mode as {mode}")
        else:
            if self.projection:
                self.proj.set_mode(mode)
            self.mode = mode
    def forward(self, x):
        """
        Forward pass: computes output for both original and reversed input
        and averages the results. This is fed into the projector.
        """
        if self.projection:
            if self.proj.mode == "variant":
                x_r, x_a = x
                for_x_r = self.trunk(x_r)
                for_x_r = self.head(for_x_r)

                rev_x_r = torch.flip(x_r, dims=[1, 2])
                rev_x_r = self.trunk(rev_x_r)
                rev_x_r = self.head(rev_x_r)

                out_r = (for_x_r + rev_x_r) / 2


                for_x_a = self.trunk(x_a)
                for_x_a = self.head(for_x_a)

                rev_x_a = torch.flip(x_a, dims=[1, 2])
                rev_x_a = self.trunk(rev_x_a)
                rev_x_a = self.head(rev_x_a)

                out_a = (for_x_a + rev_x_a) / 2

                out = (out_r, out_a)
                out = self.proj(out)
            else: ## default to sequence
                for_x = self.trunk(x)
                for_x = self.head(for_x)

                rev_x = torch.flip(x, dims=[1, 2])
                rev_x = self.trunk(rev_x)
                rev_x = self.head(rev_x)

                out = (for_x + rev_x) / 2
                out = self.proj(out)
        else:
            if self.mode == "variant":
                x_r, x_a = x
                for_x_r = self.trunk(x_r)
                for_x_r = self.head(for_x_r)

                rev_x_r = torch.flip(x_r, dims=[1, 2])
                rev_x_r = self.trunk(rev_x_r)
                rev_x_r = self.head(rev_x_r)

                out_r = (for_x_r + rev_x_r) / 2


                for_x_a = self.trunk(x_a)
                for_x_a = self.head(for_x_a)

                rev_x_a = torch.flip(x_a, dims=[1, 2])
                rev_x_a = self.trunk(rev_x_a)
                rev_x_a = self.head(rev_x_a)

                out_a = (for_x_a + rev_x_a) / 2

                out = (out_r, out_a)
            else:
                for_x = self.trunk(x)
                for_x = self.head(for_x)

                rev_x = torch.flip(x, dims=[1, 2])
                rev_x = self.trunk(rev_x)
                rev_x = self.head(rev_x)

                out = (for_x + rev_x) / 2

        return out

def initialize_models(k_l: int, quant: bool, model_name = "borzoi", full = False):

    if quant == True:
        dev = "cpu"
    else:
        dev = 'cuda:1'
    if dev == "cpu":
        q = "CPU"
    else:
        q = None
    if full != True:
        if model_name not in MODEL_PARAMS:
            model = sl.Sei_LLRA(k=k_l, projection = False, mode = "sequence", quant = q)
        else:
            model = None

    else:
        
        if model_name == "sei":
            model = SeiWrapper(k=None, ft=None, projection=False, mode="sequence", device=dev)

        else:
            model = None

  
    return model

def calculate_correlation(model_name):
    ranks = [1] #,2,4,8,16,32,64,128,256,512]
    seed = 42
    g = torch.Generator()
    g.manual_seed(seed)
    bed = "./GRCh38_cCREs_4kb.bed"
    device = torch.device('cuda:1')
    if model_name not in MODEL_PARAMS:
        input_len = 4096
    else:
        input_len = MODEL_PARAMS[model_name]["input_len"]
    dataset = SeqDataset(file_path=bed, scores_path= False, fasta_path = "../resources/hg38_UCSC.fa", window_size = input_len, mode="test", test_chrom = ["chr8", "chr9"])
    dataloader = SeqDataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=7, n_samples=1000, generator=g)

    pearson_corrs_ranks = {}

    for r in ranks:
        pearson_corrs = []
        gpu_model = initialize_models(k_l = r, quant = False, model_name = model_name, full = False)
        gpu_model.to(device)
        print(gpu_model)
        cpu_model = initialize_models(k_l = r, quant = True, model_name = model_name, full = False)
        cpu_model.to("cpu")
        print(cpu_model)
        progress_bar = tqdm(dataloader, desc=f"Running {model_name} rank {r}")
        for batch in progress_bar:
            data, _ = batch
            data = data.to(device)
            with torch.no_grad():
                out_full = gpu_model(data)
                data = data.to("cpu")
                out_lora = cpu_model(data)
            corr = np.corrcoef(out_full.cpu().detach().numpy().flatten(), out_lora.cpu().detach().numpy().flatten())[0,1]
            pearson_corrs.append(corr)
        pearson_corrs_ranks[r] = pearson_corrs
        print(f"Completed {model_name} rank {r}: Mean Pearson Correlation: {np.mean(pearson_corrs)}")
        del cpu_model
        # Compute correlation between full_model and lora_model
    torch.cuda.empty_cache()
    return pearson_corrs_ranks

def main():
    sei_corrs = calculate_correlation("sei")
    with open("sei_pearson_corrs_shuffle_quant.pkl", "wb") as f:
       pickle.dump(sei_corrs, f)

if __name__ == "__main__":
    main()