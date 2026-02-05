#!/usr/bin/env python 
import os, sys, csv
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sei_lora.dataloaders import VariantDataset, SeqDataLoader, SeqDataset, VariantDataLoader
from sei_lora.score import get_celltype_asssy_specific, get_sequence_class_scores_and_max 
from tqdm import tqdm
import scipy
from sklearn.metrics import average_precision_score, matthews_corrcoef

import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.special import expit
from sklearn.metrics import average_precision_score, matthews_corrcoef, f1_score, roc_auc_score
import seimodel as sm
import seillra as sl
import torch.nn as nn
import torch


import os, sys
from typing import Optional, Literal

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


def initialize_models(rank: int, trained_version: str, quant: bool, full = False):
    if quant == True:
        dev = "cpu"
    else:
        dev = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    if dev == "cpu":
        q = "CPU"
    else:
        q = None
    if not full:
        cp_model_seq = sl.Sei_LLRA(k=rank, projection = False, mode = "sequence", quant = q)
        cp_model_var = sl.Sei_LLRA(k=rank, projection = False, mode = "variant", quant = q)
        sc_model_seq = sl.Sei_LLRA(k=rank, projection = True, mode = "sequence", quant = q)
        sc_model_var = sl.Sei_LLRA(k=rank, projection = True, mode = "variant", quant = q)

        if quant != True:
            cp_model_seq.trunk.load_weights()
            cp_model_var.trunk.load_weights()
            sc_model_seq.trunk.load_weights()
            sc_model_var.trunk.load_weights()
    else:
        cp_model_seq = SeiWrapper(k=rank, ft = trained_version, projection = False, mode = "sequence", device = dev)
        cp_model_var = SeiWrapper(k=rank, ft = trained_version, projection = False, mode = "variant", device = dev)
        sc_model_seq = SeiWrapper(k=rank, ft = trained_version, projection = True, mode = "sequence", device = dev)
        sc_model_var = SeiWrapper(k=rank, ft = trained_version, projection = True, mode = "variant", device = dev)


    return cp_model_seq, cp_model_var, sc_model_seq, sc_model_var




def get_gtex_eqtls_promoter(model, rank, trained_version = ""):
    benchmark_name = "gtex_eqtls_near_promoter"
    vcf_name ="../data/tableS1D_gtex_eqtls.vcf"
    sc_ref, sc_alt, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, sc = True)
    df = pd.read_csv("../data/tableS1D_gtex_eqtls.tsv", header = 0, sep = "\t")
   
    outs = get_over_under_null(sc_ref, sc_alt, vcf, df)
    return outs 

def get_mpra_eqtls_promoter(model, rank, trained_version = ""):
    benchmark_name = "mpra_eqtls_near_promoter"
    vcf_name ="../data/tableS1E_mpra_eqtls.vcf"
    sc_ref, sc_alt, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, sc = True)
    df = pd.read_csv("../data/tableS1E_mpra_eqtls.tsv", header = 0, sep = "\t")
   
    outs = get_over_under_null(sc_ref, sc_alt, vcf, df)
    return outs 

def get_gtex_outliers_promoter(model, rank, trained_version = ""):
    benchmark_name = "gtex_outliers_near_promoter"
    vcf_name ="../data/tableS1A_gtex_outliers.vcf"
    sc_ref, sc_alt, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, sc = True)
    df = pd.read_csv("../data/tableS1A_gtex_outliers.tsv", header = 0, sep = "\t")
   
    outs = get_over_under_null(sc_ref, sc_alt, vcf, df)
    return outs 

def get_cagi5_sat_promoter(model, rank, trained_version = ""):
    benchmark_name = "cagi5_sat_near_promoter"
    vcf_name ="../data/tableS1B_cagi5_saturation.vcf"
    sc_ref, sc_alt, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, sc = True)
    df = pd.read_csv("../data/tableS1B_cagi5_saturation.tsv", header = 0, sep = "\t")
   
    outs = get_over_under_null(sc_ref, sc_alt, vcf, df)
    return outs 

def get_mpra_sat_promoter(model, rank, trained_version = ""):
    benchmark_name = "mpra_sat_near_promoter"
    vcf_name ="../data/tableS1C_mpra_saturation.vcf"
    sc_ref, sc_alt, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, sc = True)
    df = pd.read_csv("../data/tableS1C_mpra_saturation.tsv", header = 0, sep = "\t")
   
    outs = get_over_under_null(sc_ref, sc_alt, vcf, df)
    return outs 

def get_ukbb_proteome_promoter(model, rank, trained_version = ""):
    benchmark_name = "ukbb_proteome_near_promoter"
    vcf_name ="../data/tableS1F_ukbb_proteome.vcf"
    sc_ref, sc_alt, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, sc = True)
    df = pd.read_csv("../data/tableS1F_ukbb_proteome.tsv", header = 0, sep = "\t")
   
    outs = get_over_under_null(sc_ref, sc_alt, vcf, df)
    return outs 

def get_gel_rna_promoter(model, rank, trained_version = ""):
    benchmark_name = "gel_rna_near_promoter"
    vcf_name ="../data/tableS1G_gel_rna.vcf"
    sc_ref, sc_alt, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, sc = True)
    df = pd.read_csv("../data/tableS1G_gel_rna.tsv", header = 0, sep = "\t")
   
    outs = get_over_under_null(sc_ref, sc_alt, vcf, df)
    return outs 

def get_over_under_null(sc_ref, sc_alt, vcf, df):
    sc_diff = sc_alt - sc_ref
    df_pred = pd.DataFrame(vcf, columns=["CHROM", "POS", "NAME", "REF", "ALT"])
    df_pred["POS"] = df_pred["POS"].astype(int)
    df_sc = get_sequence_class_scores_and_max(sc_diff)
    df_pred =  pd.concat([df_pred, df_sc], axis=1)

    df_ou = df[df['consequence'].isin(['over', 'under'])].copy()
    df_combine_ou = df_ou.merge(df_pred, left_on = ["chrom", "pos", "ref", "alt"], right_on=["CHROM", "POS", "REF", "ALT"], how = "inner")
    df_combine_ou = df_combine_ou.drop_duplicates()
    binary_labels_ou = (df_combine_ou['consequence'] == 'over')
    roc_promoter_ou = roc_auc_score(binary_labels_ou, df_combine_ou["Promoter"])

    df_un = df[df['consequence'].isin(['under', 'none'])].copy()
    df_combine_un = df_un.merge(df_pred, left_on = ["chrom", "pos", "ref", "alt"], right_on=["CHROM", "POS", "REF", "ALT"], how = "inner")
    df_combine_un = df_combine_un.drop_duplicates()
    binary_labels_un = (df_combine_un['consequence'] == 'under')
    roc_promoter_un = roc_auc_score(binary_labels_un, -df_combine_un["Promoter"])

    df_on = df[df['consequence'].isin(['over', 'none'])].copy()
    df_combine_on = df_on.merge(df_pred, left_on = ["chrom", "pos", "ref", "alt"], right_on=["CHROM", "POS", "REF", "ALT"], how = "inner")
    df_combine_on = df_combine_on.drop_duplicates()
    binary_labels_on = (df_combine_on['consequence'] == 'over')
    roc_promoter_on = roc_auc_score(binary_labels_on, df_combine_on["Promoter"])


    return roc_promoter_ou, roc_promoter_un, roc_promoter_on 


def get_eu_lcl_caqtls(model, rank, trained_version = ""):
    benchmark_name = "caqtls_eu_GM12878"
    vcf_name ="../data/caqtls.eu.lcls.benchmarking.all.vcf"
    cp_ref, cp_alt, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version)
    df = pd.read_csv("../data/caqtls.eu.lcls.benchmarking.tsv", header = 0, sep = "\t")
    df = df[df["var.isused"]]
    df["log10p"] = np.log10(df["obs.pval"])*-1
   
    diff = cp_alt - cp_ref
    df_pred = pd.DataFrame(vcf, columns=["CHROM", "POS", "NAME", "REF", "ALT"])
    df_pred["POS"] = df_pred["POS"].astype(int)
    df_pred["GM12878_DNase_cp_mean"] = get_celltype_asssy_specific(diff, celltypes = ["GM12878_B_Lymphocyte_Blood"], assays = ["ATAC-seq", "DNase"], strict = True)
    df_pred["Cardiomyocyte_DNase_cp_mean"] = get_celltype_asssy_specific(diff, celltypes = ["Cardiomyocyte"], assays = ["ATAC-seq", "DNase"], strict = True)


    dataf1 = df[df["log10p"]>6]
    dataf2 = df[df["log10p"]<3]
    dataf1.loc[:, "obs.label"] = 1
    dataf2.loc[:, "obs.label"] = 0

    dataf = pd.concat([dataf1, dataf2])
    df_combine = dataf.merge(df_pred, left_on = ["var.chr", "var.pos_hg38"], right_on=["CHROM", "POS"], how = "inner")
    df_combine.to_csv(f"scores/caqtls_eu_lcl_seilora_rank{rank}_quant.tsv.gz", sep="\t", index=False, compression="gzip")


    ap_unsigned = average_precision_score(df_combine["obs.label"], abs(df_combine["GM12878_DNase_cp_mean"]))

    df_combine = df_combine[df_combine["log10p"]>6]
    pearson_signed = scipy.stats.pearsonr(df_combine["GM12878_DNase_cp_mean"],df_combine["obs.beta"])
   
    return pearson_signed , ap_unsigned

def get_yoruba_lcl_dsqtls(model, rank, trained_version = ""):
    benchmark_name = "dsqtls_yoruba_GM12878"
    vcf_name ="../data/dsqtls.yoruba.lcls.benchmarking.all.vcf"
    
    df = pd.read_csv("../data/dsqtls.yoruba.lcls.benchmarking.all.tsv", index_col=False,  header = 0, sep = "\t")
    df = df[df["var.isused"]]
    # df = df[df["obs.label"] ==1]
    
    cp_ref, cp_alt, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version)
    diff = cp_alt - cp_ref
    df_pred = pd.DataFrame(vcf, columns=["CHROM", "POS", "NAME", "REF", "ALT"])
    df_pred["POS"] = df_pred["POS"].astype(int)
    df_pred["GM12878_DNase_cp_mean"] = get_celltype_asssy_specific(diff, celltypes = ["GM12878_B_Lymphocyte_Blood"], assays = ["DNase"], strict = True)

    df_combine = df.merge(df_pred, left_on = ["var.chr", "var.pos_hg38"], right_on=["CHROM", "POS"], how = "inner")
    ap_unsigned = average_precision_score(df_combine["obs.label"], abs(df_combine["GM12878_DNase_cp_mean"]))
    df_combine.to_csv(f"scores/dsqtls_yoruba_lcl_seilora_rank{rank}_quant.tsv.gz", sep="\t", index=False, compression="gzip")

    df_combine = df_combine[df_combine["obs.label"] ==1]
    pearson_signed = scipy.stats.pearsonr(-df_combine["GM12878_DNase_cp_mean"],df_combine["obs.estimate"])

    return pearson_signed, ap_unsigned

def get_afr_lcl_caqtls(model, rank, trained_version = ""):
    benchmark_name = "caqtls_african_GM12878"
    vcf_name ="../data/caqtls.african.lcls.benchmarking.all.vcf"
    cp_ref, cp_alt, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version)
    df = pd.read_csv("../data/caqtls.african.lcls.benchmarking.tsv", header = 0, sep = "\t")
    df = df[df["var.isused"]]
    df = df.dropna(subset=["obs.label"])
    # df["log10p"] = np.log10(df["obs.pval"])*-1
   
    diff = cp_alt - cp_ref
    df_pred = pd.DataFrame(vcf, columns=["CHROM", "POS", "NAME", "REF", "ALT"])
    df_pred["POS"] = df_pred["POS"].astype(int)
    df_pred["GM12878_DNase_cp_mean"] = get_celltype_asssy_specific(diff, celltypes = ["GM12878_B_Lymphocyte_Blood"], assays = ["DNase"], strict = True)

    df_combine = df.merge(df_pred, left_on = ["var.chr", "var.pos_hg38"], right_on=["CHROM", "POS"], how = "inner")
    ap_unsigned = average_precision_score(df_combine["obs.label"], abs(df_combine["GM12878_DNase_cp_mean"]))
    df_combine.to_csv(f"scores/caqtls_afr_lcl_seilora_rank{rank}_quant.tsv.gz", sep="\t", index=False, compression="gzip")

    df_combine = df_combine[df_combine["obs.label"] ==1]
    pearson_signed = scipy.stats.pearsonr(df_combine["GM12878_DNase_cp_mean"],df_combine["obs.beta"])

    return pearson_signed , ap_unsigned

def get_microglia_caqtls(model, rank, trained_version = ""):
    benchmark_name = "caqtls_eu_Microglia"
    vcf_name ="../data/caqtls.microglia.benchmarking.all.vcf"
    cp_ref, cp_alt, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version)
    df = pd.read_csv("../data/caqtls.microglia.benchmarking.tsv", header = 0, sep = "\t")
    df = df[df["var.isused"]]
    df = df.dropna().drop_duplicates()
    diff = cp_alt - cp_ref
    df_pred = pd.DataFrame(vcf, columns=["CHROM", "POS", "NAME", "REF", "ALT"])

    df_pred["POS"] = df_pred["POS"].astype(int)
    df_pred["Macrophage_ATAC_cp_mean"] = get_celltype_asssy_specific(diff, celltypes = ["Macrophage"], assays = ["ATAC-seq", "DNase"], strict = False)

    df_combine = df.merge(df_pred, left_on = ["var.chr", "var.pos_hg38", "var.allele1", "var.allele2"], right_on=["CHROM", "POS", "REF", "ALT"], how = "inner")
    df_combine = df_combine.dropna().drop_duplicates()

    df_combine.to_csv(f"scores/caqtls_eu_microglia_seilora_rank{rank}_quant.tsv.gz", sep="\t", index=False, compression="gzip")
    pearson_signed = scipy.stats.pearsonr(-df_combine["Macrophage_ATAC_cp_mean"],df_combine["obs.Beta"])
    return pearson_signed

def get_smc_caqtls(model, rank, trained_version = ""):
    benchmark_name = "caqtls_eu_SMC"
    vcf_name ="../data/caqtls.smc.benchmarking.all.vcf"
    cp_ref, cp_alt, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version)
    df = pd.read_csv("../data/caqtls.smc.benchmarking.tsv", header = 0, sep = "\t")
    df = df.dropna()
    df = df[df["var.isused"]]
    diff = cp_alt - cp_ref
    df_pred = pd.DataFrame(vcf, columns=["CHROM", "POS", "NAME", "REF", "ALT"])
    df_pred["POS"] = df_pred["POS"].astype(int)
    df_pred["SMC_ATAC_cp_mean"] = get_celltype_asssy_specific(diff, celltypes = ["Smooth_Muscle_Cell_Coronary_artery_smooth_muscle"], assays = ["ATAC-seq"], strict = True)

    df_combine = df.merge(df_pred, left_on = ["var.chr", "var.pos_hg38"], right_on=["CHROM", "POS"], how = "inner")
    df_combine.to_csv(f"scores/caqtls_eu_smc_seilora_rank{rank}_quant.tsv.gz", sep="\t", index=False, compression="gzip")
    pearson_signed = scipy.stats.pearsonr(df_combine["SMC_ATAC_cp_mean"],df_combine["obs.Effect_size"])
    return pearson_signed

def get_spi1_bqtls(model, rank, trained_version = ""):
    benchmark_name = "bqtls_spi1_LCL"
    vcf_name ="../data/bqtls.pu1.lcls.benchmarking.all.vcf"
    cp_ref, cp_alt, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version)
    df = pd.read_csv("../data/bqtls.pu1.lcls.benchmarking.tsv", header = 0, sep = "\t")

    df = df[df["var.isused"]]
    diff = cp_alt - cp_ref

    df_pred = pd.DataFrame(vcf, columns=["CHROM", "POS", "NAME", "REF", "ALT"])
    df_pred["POS"] = df_pred["POS"].astype(int)
    df_pred["GM12878_spi1_cp_mean"] = get_celltype_asssy_specific(diff, celltypes = ["GM12878_B_Lymphocyte_Blood"], assays = ["SPI1"], strict = True)

    df_combine = df.merge(df_pred, left_on = ["var.chr", "var.pos_hg38"], right_on=["CHROM", "POS"], how = "inner")

    df_combine.to_csv(f"scores/bsqtls_eu_spi1_seilora_rank{rank}_quant.tsv.gz", sep="\t", index=False, compression="gzip")
    df_combine = df_combine[df_combine["obs.pval"] < 1e-9]
    pearson_signed = scipy.stats.pearsonr(df_combine["GM12878_spi1_cp_mean"],df_combine["obs.chiplogratio"])

    return pearson_signed


def get_variants(model, vcf, rank, benchmark_name="", trained_version = "", sc = False):
    dataset = VariantDataset(file_path=vcf)
    dataloader = VariantDataLoader(dataset=dataset, batch_size=32, shuffle=False, num_workers=15)
    device = model.device
    model = model.to(device)
    model.eval()

    all_cp_ref = []
    all_cp_alt = []
    all_vcf = []

    progress_bar = tqdm(dataloader, desc=f"Running {rank} {trained_version} {benchmark_name} benchmark")

    for batch in progress_bar:
        ref, alt, vcf = batch
        ref, alt = ref.to(device), alt.to(device)

        cp_outputs = model((ref, alt))  # both are tuples: (refproj, altproj)

    
        all_cp_ref.append(cp_outputs[0].detach().cpu())
        all_cp_alt.append(cp_outputs[1].detach().cpu())
        all_vcf.append(vcf)

        # Accumulate by appending to list
    
    all_cp_ref = torch.cat([t.detach().cpu() for t in all_cp_ref], dim=0).numpy()
    all_cp_alt = torch.cat([t.detach().cpu() for t in all_cp_alt], dim=0).numpy()

    all_vcf = np.concatenate(all_vcf, axis=0)
    model = model.to("cpu")
    torch.cuda.empty_cache()

    return all_cp_ref, all_cp_alt, all_vcf

def get_scores(model, bed, rank, benchmark_name="", trained_version = "", scores = None, sc = False):
    dataset = SeqDataset(file_path=bed, scores_path = scores, fasta_path="../../sei-framework-main/resources/hg38_UCSC.fa",
            mode = "test", val_chrom = "chr10", test_chrom = ["chr8", "chr9"])
    dataloader = SeqDataLoader(dataset=dataset, batch_size=32, shuffle=False, num_workers=15, n_samples=10_000)
    device = model.device

    model = model.to(device)
    model.eval()

    all_cp = []
    all_scores = []

    progress_bar = tqdm(dataloader, desc=f"Running {rank} {trained_version} {benchmark_name} benchmark")

    for batch in progress_bar:
        data, score = batch
        data = data.to(device)

        out = model(data)  # both are tuples: (refproj, altproj)

    
        all_cp.append(out.detach().cpu())
        all_scores.append(score)

        # Accumulate by appending to list
    model = model.to("cpu")
    
    all_cp = torch.cat([t.detach().cpu() for t in all_cp], dim=0).numpy()
    all_scores = np.concatenate(
        [t.detach().cpu().numpy() if torch.is_tensor(t) else t for t in all_scores],
        axis=0
    )
    torch.cuda.empty_cache()
    return all_cp, all_scores

def save_output(rank = 256, trained_version = None, quant = False, full = False):
    if quant:
        q = "quant"
    else:
        q = "no_quant"

    model_name = f"seilora_{rank}_{trained_version}_{q}" 


    cp_seq_mod, cp_var_mod, sc_seq_mod, sc_var_mod = initialize_models(rank = rank, trained_version = trained_version, quant = quant, full = full)

    ## PromoterAI
    # gtex_eqtls = get_gtex_eqtls_promoter(model = sc_var_mod, rank = rank, trained_version = trained_version)
    # mpra_eqtls = get_mpra_eqtls_promoter(model = sc_var_mod, rank = rank, trained_version = trained_version)
    # gtex_outliers = get_gtex_outliers_promoter(model = sc_var_mod, rank = rank, trained_version = trained_version)
    # cagi5_sat = get_cagi5_sat_promoter(model = sc_var_mod, rank = rank, trained_version = trained_version)
    # mpra_sat = get_mpra_sat_promoter(model = sc_var_mod, rank = rank, trained_version = trained_version)
    # ukbb_proteome = get_ukbb_proteome_promoter(model = sc_var_mod, rank = rank, trained_version = trained_version)
    # gel_rna = get_gel_rna_promoter(model = sc_var_mod, rank = rank, trained_version = trained_version)
    
    # # spi1_pearson = get_spi1_bqtls(model = cp_var_mod, rank = rank, trained_version = trained_version)

    # pai_path = "benchmark_pai_all_fixed.tsv"
    # pai_row_dict = {
    #     "model": model_name,
    #     "GTEX_eqtl_OvU_promoter": round(gtex_eqtls[0], 4),
    #     "GTEX_eqtl_outliers_OvU_promoter": round(gtex_outliers[0], 4),
    #     "CAGI5_saturation_OvU_promoter": round(cagi5_sat[0], 4),
    #     "MPRA_saturation_OvU_promoter": round(mpra_sat[0], 4),
    #     "MPRA_eqtl_OvU_promoter": round(mpra_eqtls[0], 4),
    #     "UKBB_proteome_OvU_promoter": round(ukbb_proteome[0], 4),
    #     "Gel_RNA_OvU_promoter": round(gel_rna[0], 4),

    #     "GTEX_eqtl_UvN_promoter": round(gtex_eqtls[1], 4),
    #     "GTEX_eqtl_outliers_UvN_promoter": round(gtex_outliers[1], 4),
    #     "CAGI5_saturation_UvN_promoter": round(cagi5_sat[1], 4),
    #     "MPRA_saturation_UvN_promoter": round(mpra_sat[1], 4),
    #     "MPRA_eqtl_UvN_promoter": round(mpra_eqtls[1], 4),
    #     "UKBB_proteome_UvN_promoter": round(ukbb_proteome[1], 4),
    #     "Gel_RNA_UvN_promoter": round(gel_rna[1], 4),

    #     "GTEX_eqtl_OvN_promoter": round(gtex_eqtls[2], 4),
    #     "GTEX_eqtl_outliers_OvN_promoter":round(gtex_outliers[2], 4),
    #     "CAGI5_saturation_OvN_promoter": round(cagi5_sat[2], 4),
    #     "MPRA_saturation_OvN_promoter": round(mpra_sat[2], 4),
    #     "MPRA_eqtl_OvN_promoter": round(mpra_eqtls[2], 4),
    #     "UKBB_proteome_OvN_promoter": round(ukbb_proteome[2], 4),
    #     "Gel_RNA_OvN_promoter": round(gel_rna[2], 4),

    #     # "EU_spi1_LCL_pearson_signed": round(spi1_pearson.statistic, 4),
    # }
    # file_exists = os.path.isfile(pai_path)
    # with open(pai_path, "a", newline="") as f:
    #         writer = csv.DictWriter(f, fieldnames=pai_row_dict.keys(), delimiter="\t")
    #         if not file_exists:
    #             writer.writeheader()
    #         writer.writerow(pai_row_dict)

    #         print(f"Results saved to {pai_path}")
    # # ChrombpNet

    yoruba_pearson, yoruba_ap = get_yoruba_lcl_dsqtls(model = cp_var_mod, rank = rank, trained_version = trained_version)
    eu_pearson, eu_ap = get_eu_lcl_caqtls(model = cp_var_mod, rank = rank, trained_version = trained_version)

    afr_pearson, afr_ap = get_afr_lcl_caqtls(model = cp_var_mod, rank = rank, trained_version = trained_version)

    microglia_pearson = get_microglia_caqtls(model = cp_var_mod, rank = rank, trained_version = trained_version)
    smc_pearson = get_smc_caqtls(model = cp_var_mod, rank = rank, trained_version = trained_version)
    spi1_pearson = get_spi1_bqtls(model = cp_var_mod, rank = rank, trained_version = trained_version)
    print(microglia_pearson)
    print(spi1_pearson)

    bpn_path = "benchmark_chrombpnet_all_quant.tsv"
    bpn_row_dict = {
        "model": model_name,
        "EU_LCL_pearson_signed": round(eu_pearson.statistic, 4),
        "Yoruba_LCL_pearson_signed": round(yoruba_pearson.statistic, 4),
        "African_LCL_pearson_signed":round(afr_pearson.statistic, 4),
        "EU_Microglia_pearson_signed": round(microglia_pearson.statistic, 4),
        "EU_spi1_LCL_pearson_signed": round(spi1_pearson.statistic, 4),
        "EU_SMC_pearson_signed": round(smc_pearson.statistic, 4),

        "EU_LCL_AP_unsigned": round(eu_ap, 4),
        "Yoruba_AP_LCL_unsigned": round(yoruba_ap, 4),
        "African_AP_unsigned": round(afr_ap, 4)
  
    }

    file_exists = os.path.isfile(bpn_path)
    with open(bpn_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=bpn_row_dict.keys(), delimiter="\t")
            if not file_exists:
                writer.writeheader()
            writer.writerow(bpn_row_dict)

            print(f"Results saved to {bpn_path}")
    

    torch.cuda.empty_cache()

def main():

    #save_output(rank="full", quant = False, full = True)
  
    # save_output(rank=1, quant = False)
    # # save_output(rank=2, quant = False)
    # save_output(rank=4, quant = False)
    # # save_output(rank=8, quant = False)
    # save_output(rank=16, quant = False)
    # # save_output(rank=32, quant = False)
    # save_output(rank=64, quant = False)
    # # save_output(rank=128, quant = False)
    # save_output(rank=256, quant = False)
    # # save_output(rank=512, quant = False)
    # save_output(rank=1024, quant = False)
    # save_output(rank=2048, quant = False)
    # save_output(rank=1, quant = True)
    # save_output(rank=4, quant = True)
    # save_output(rank=16, quant = True)
    # save_output(rank=64, quant = True)
    save_output(rank=256, quant = True)
    save_output(rank=1024, quant = True)
    save_output(rank=2048, quant = True)



    # save_output(rank=2048, quant = True)
  



if __name__ == '__main__':
    main()