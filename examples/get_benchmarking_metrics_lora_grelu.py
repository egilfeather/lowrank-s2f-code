#!/usr/bin/env python 
import os, sys, csv
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import numpy as np
import torch
import torch.nn as nn
# import sei_lora.module
import pandas as pd
from seq_dataloader import VariantDataset, SeqDataLoader, SeqDataset, VariantDataLoader
# from sei_lora.model import get_celltype_asssy_specific, get_sequence_class_scores_and_max #Variant_Prediction_Processor, load_to_anndata
from tqdm import tqdm
import scipy
from sklearn.metrics import average_precision_score, matthews_corrcoef

import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.special import expit
from sklearn.metrics import average_precision_score, matthews_corrcoef, f1_score, roc_auc_score

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import math

# import grelu.resources
# import wandb
# import time
from borzoi_lora_arch_mha import BorzoiModel, EnformerModel
import grelu.resources



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
            "cbpnet_mask": "center",
            "cbpnet_metric": "log2_diff",
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
            "pai_metric": "log2_diff",
            "cbpnet_mask": "center_8",
            "cbpnet_metric": "SAR",

    }

    }

def get_center(tensor: torch.Tensor, center_bp: int, bin_length=1):
    seq_len = tensor.shape[-1]

    # Compute how many bins correspond to center_bp
    center_bins = math.ceil(center_bp / bin_length)

    if center_bins > seq_len:
        raise ValueError(f"center window ({center_bins} bins) exceeds tensor length ({seq_len})")

    start = (seq_len - center_bins) // 2
    end = start + center_bins

    return tensor[..., start:end]

def read_gtf_to_df(gtf_file):
    """
    Read a GTF file and extract protein-coding transcript entries into a DataFrame.
    """
    gtf_cols = [
        "chrom", "source", "feature", "start", "end", "score",
        "strand", "frame", "attribute"
    ]
    df = pd.read_csv(gtf_file, sep="\t", comment="#", names=gtf_cols)
    df = df[(df["feature"] == "transcript") & (df["attribute"].str.contains('gene_type "protein_coding"'))]

    # Simplify chromosome names
    df["chrom"] = df["chrom"].astype(str).str.replace("chr", "", regex=False)

    return df[["chrom", "start", "end"]]

def get_gene(ref_outputs, vcf, genes_df, bin_length):
    """
    Subset model output tensor to overlap the closest gene or transcript.

    Args:
        ref_outputs (torch.Tensor): shape [B, C, L]
        vcf (pd.Series or dict): must have 'chr' and 'pos'
        genes_df (pd.DataFrame): must have ['chrom', 'start', 'end', 'gene_name']
        bin_length (int): number of base pairs per bin (model resolution)

    Returns:
        (torch.Tensor, dict): subset tensor and selected gene metadata
    """
    # print(vcf)
    # print(type(vcf))
    chr_ = str(vcf[0][0]).replace("chr", "")
    pos = int(vcf[0][1])
    B, C, L = ref_outputs.shape

    # Compute genomic span of the model window
    half_bp = (L * bin_length) // 2
    model_start = pos - half_bp
    model_end = pos + half_bp

    # Filter to same chromosome
    # print(genes_df["chrom"].iloc[0])
    # print(chr_)
    chr_genes = genes_df[genes_df["chrom"] == chr_]
    if chr_genes.empty:
        raise ValueError(f"No genes found on chromosome {chr_}")

    # Find overlapping or closest gene
    overlap = chr_genes[(chr_genes["end"] >= model_start) & (chr_genes["start"] <= model_end)].copy() 
    if overlap.empty:
        return get_center(ref_outputs, 1_000, bin_length)
    else:
        overlap["dist"] = np.minimum(
                            (overlap["end"] - pos).abs(),
                            (overlap["start"] - pos).abs()
                        )
        gene_row = overlap.loc[overlap["dist"].idxmin()]

    gene_start, gene_end = int(gene_row["start"]), int(gene_row["end"])

    # Compute bin overlap coordinates
    overlap_start = max(gene_start, model_start)
    overlap_end = min(gene_end, model_end)
    bin_start = max(0, (overlap_start - model_start) // bin_length)
    bin_end = min(L, (overlap_end - model_start) // bin_length)

    # Subset tensor to overlapping bins
    gene_tensor = ref_outputs[:, :, bin_start:bin_end]

    return gene_tensor

def read_gtf_exons(gtf_file):
    """
    Read a GTF file and extract protein-coding exon entries into a DataFrame.
    """
    gtf_cols = [
        "chrom", "source", "feature", "start", "end", "score",
        "strand", "frame", "attribute"
    ]
    df = pd.read_csv(gtf_file, sep="\t", comment="#", names=gtf_cols)

    # Keep only exons from protein-coding genes
    df = df[(df["feature"] == "exon") & (df["attribute"].str.contains('gene_type "protein_coding"'))].copy()

    # Extract transcript_id and gene_name
    df["transcript_id"] = df["attribute"].apply(
        lambda x: x.split('transcript_id "')[1].split('"')[0] if 'transcript_id "' in x else None
    )
    df["gene_name"] = df["attribute"].apply(
        lambda x: x.split('gene_name "')[1].split('"')[0] if 'gene_name "' in x else None
    )

    # Simplify chromosome names
    df["chrom"] = df["chrom"].astype(str).str.replace("chr", "", regex=False)

    return df[["chrom", "start", "end", "gene_name", "transcript_id"]]

def get_exons(ref_outputs, vcf, exons_df, bin_length):
    """
    Subset model output tensor to the bins overlapping exons of the closest transcript.

    Args:
        ref_outputs (torch.Tensor): shape [B, C, L]
        vcf (pd.Series or list of tuples): variant info with ['chr', 'pos']
        exons_df (pd.DataFrame): exon-level annotations with ['chrom', 'start', 'end', 'transcript_id', 'gene_name']
        bin_length (int): number of base pairs per bin

    Returns:
        torch.Tensor: subset of ref_outputs covering all overlapping exon bins
    """
    chr_ = str(vcf[0][0]).replace("chr", "")
    pos = int(vcf[0][1])
    B, C, L = ref_outputs.shape

    # Model window in genomic coordinates
    half_bp = (L * bin_length) // 2
    model_start = pos - half_bp
    model_end = pos + half_bp

    # Filter exons on same chromosome
    chr_exons = exons_df[exons_df["chrom"] == chr_]
    if chr_exons.empty:
        raise ValueError(f"No exons found on chromosome {chr_}")

    # Determine overlapping transcripts
    overlap_transcripts = chr_exons[
        (chr_exons["end"] >= model_start) & (chr_exons["start"] <= model_end)
    ]["transcript_id"].unique()

    if len(overlap_transcripts) == 0:
        # fallback: pick the closest transcript
        chr_exons["dist"] = ((chr_exons["start"] + chr_exons["end"]) // 2 - pos).abs()
        closest_tx = chr_exons.loc[chr_exons["dist"].idxmin()]["transcript_id"]
    else:
        # pick the transcript whose exons are closest to pos
        tx_dist = []
        for tx in overlap_transcripts:
            tx_exons = chr_exons[chr_exons["transcript_id"] == tx]
            dist = np.min(np.minimum((tx_exons["start"] - pos).abs(), (tx_exons["end"] - pos).abs()))
            tx_dist.append((tx, dist))
        closest_tx = min(tx_dist, key=lambda x: x[1])[0]

    # Get exons of the closest transcript
    tx_exons = chr_exons[chr_exons["transcript_id"] == closest_tx]

    # Collect all overlapping bins
    bin_indices = []
    for _, exon in tx_exons.iterrows():
        exon_start = max(exon["start"], model_start)
        exon_end = min(exon["end"], model_end)
        start_bin = max(0, (exon_start - model_start) // bin_length)
        end_bin = min(L, (exon_end - model_start) // bin_length)
        if end_bin > start_bin:
            bin_indices.extend(range(start_bin, end_bin))

    # Remove duplicates and sort
    bin_indices = sorted(set(bin_indices))

    # Subset tensor
    if len(bin_indices) == 0:
        # fallback: return small center window
        center_bins = max(1, min(L, 1_000 // bin_length))
        start = (L - center_bins) // 2
        end = start + center_bins
        return ref_outputs[:, :, start:end]

    gene_tensor = ref_outputs[:, :, bin_indices]
    return gene_tensor


def initialize_models(k_l: int, quant: bool, model_name = "borzoi", full = False):

    if quant == True:
        dev = "cpu"
    else:
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if full != True:

        model = MODEL_PARAMS[model_name]["model_class"](k_l =k_l, device = dev, **MODEL_PARAMS[model_name]["kwargs"])
        # model = BorzoiModel(k_l =k_l, k_c = k_c, n_tasks = 7611, crop_len=5120, final_act_func="softplus", final_pool_func=None)
        state_dict = torch.load(f'{model_name}_lora_weights/{model_name}_lora_lr{k_l}.pth')

        model.load_state_dict(state_dict, strict = True)
    else:
        if model_name == "borzoi":
            model = torch.load("./full_models/borzoi_human_rep0.pt")
        elif model_name == "enformer":
            model = torch.load("./full_models/enformer_human.pt")
        else:
            model = None

  
    return model

def get_cp_sequence_metrics(model, rank, trained_version="", scores_path="", bed_path = ""):
    benchmark_name = "chromatin_profile_pearson"

    # cps = predicted scores, scores = ground truth (binary or continuous)
    cps, scores = get_scores(
        model, bed_path, rank=rank, benchmark_name=benchmark_name,
        trained_version=trained_version, scores=scores_path
    )

    # --- 1. Row-wise similarity metrics (sample-level) ---
    pearson_corrs = []
    spearman_corrs = []
    for y_true, y_pred in zip(scores, cps):
        if np.std(y_true) > 0 and np.std(y_pred) > 0:
            pearson_corrs.append(np.corrcoef(y_true, y_pred)[0, 1])
            spearman_corrs.append(spearmanr(y_true, y_pred).correlation)

    avg_pearson = np.nanmean(pearson_corrs)
    avg_spearman = np.nanmean(spearman_corrs)

    # --- 2. Column-wise prediction metrics (task-level) ---
    n_cols = scores.shape[1]
    ap_scores = []
    auroc_scores = []
    f1_scores = []
    mcc_scores = []

    for j in range(n_cols):
        y_true_col = scores[:, j]
        y_pred_col = cps[:, j]

        if y_true_col.sum() == 0 or y_true_col.sum() == len(y_true_col):
            continue  # skip columns with no positive/negative labels

        y_true_bin = (y_true_col > 0.5).astype(int)

        # AP (same as AUPRC)
        ap_scores.append(average_precision_score(y_true_bin, y_pred_col))

        # AUROC
        try:
            auroc_scores.append(roc_auc_score(y_true_bin, y_pred_col))
        except ValueError:
            pass

        # F1 and MCC (need hard predictions)
        y_pred_bin = (y_pred_col > 0.5).astype(int)
        f1_scores.append(f1_score(y_true_bin, y_pred_bin, zero_division= np.nan))
        if np.std(y_pred_bin) > 0 and np.std(y_true_bin) > 0:
            mcc_scores.append(matthews_corrcoef(y_true_bin, y_pred_bin))

    diagnostics = {
        # Row-wise similarity
        "avg_pearson": float(avg_pearson),
        "avg_spearman": float(avg_spearman),

        # Column-wise accuracy
        "avg_ap": float(np.nanmean(ap_scores)),
        "avg_auprc": float(np.nanmean(ap_scores)),  # identical to AP
        "avg_auroc": float(np.nanmean(auroc_scores)),
        "avg_f1": float(np.nanmean(f1_scores)),
        "avg_mcc": float(np.nanmean(mcc_scores)),

        # Dataset-level info
        "mean_true_sum_per_row": float(scores.sum(axis=1).mean()),
        "mean_pred_sum_per_row": float(cps.sum(axis=1).mean()),
    }

    return diagnostics

def get_sc_sequence_metrics(model, rank, trained_version="", scores_path="", bed_path = ""):
    benchmark_name = "sequence_class_pearson"

    # cps = predicted scores, scores = ground truth (binary or continuous)
    cps, scores = get_scores(
        model, bed_path, rank=rank, benchmark_name=benchmark_name,
        trained_version=trained_version, scores=scores_path, sc = True
    )

    # --- 1. Row-wise similarity metrics (sample-level) ---
    pearson_corrs = []
    spearman_corrs = []
    for y_true, y_pred in zip(scores, cps):
        if np.std(y_true) > 0 and np.std(y_pred) > 0:
            y_true = y_true[:40]
            y_pred = y_pred[:40]
            pearson_corrs.append(np.corrcoef(y_true, y_pred)[0, 1])
            spearman_corrs.append(spearmanr(y_true, y_pred).correlation)

    avg_pearson = np.nanmean(pearson_corrs)
    avg_spearman = np.nanmean(spearman_corrs)

    # --- 2. Column-wise prediction metrics (task-level) ---
    n_cols = 40
    ap_scores = []
    auroc_scores = []
    f1_scores = []
    mcc_scores = []

    for j in range(n_cols):
        y_true_col = expit(scores[:, j])
        y_pred_col = expit(cps[:, j])

        if y_true_col.sum() == 0 or y_true_col.sum() == len(y_true_col):
            continue  # skip columns with no positive/negative labels

        y_true_bin = (y_true_col > 0.5).astype(int)

        # AP (same as AUPRC)
        ap_scores.append(average_precision_score(y_true_bin, y_pred_col))

        # AUROC
        try:
            auroc_scores.append(roc_auc_score(y_true_bin, y_pred_col))
        except ValueError:
            pass

        # F1 and MCC (need hard predictions)
        y_pred_bin = (y_pred_col > 0.5).astype(int)
        f1_scores.append(f1_score(y_true_bin, y_pred_bin, zero_division=np.nan))
        if np.std(y_pred_bin) > 0 and np.std(y_true_bin) > 0:
            mcc_scores.append(matthews_corrcoef(y_true_bin, y_pred_bin))

    diagnostics = {
        # Row-wise similarity
        "avg_pearson": float(avg_pearson),
        "avg_spearman": float(avg_spearman),

        # Column-wise accuracy
        "avg_ap": float(np.nanmean(ap_scores)),
        "avg_auprc": float(np.nanmean(ap_scores)),  # identical to AP
        "avg_auroc": float(np.nanmean(auroc_scores)),
        "avg_f1": float(np.nanmean(f1_scores)),
        "avg_mcc": float(np.nanmean(mcc_scores)),

        # Dataset-level info
        "mean_true_sum_per_row": float(scores.sum(axis=1).mean()),
        "mean_pred_sum_per_row": float(cps.sum(axis=1).mean()),
    }

    return diagnostics



def get_gtex_eqtls_promoter(model, rank, trained_version = "", model_name = "borzoi"):
    benchmark_name = "gtex_eqtls_near_promoter"
    vcf_name ="./data/tableS1D_gtex_eqtls.vcf"
    cp_diffs, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, model_name = model_name, metric = MODEL_PARAMS[model_name]["pai_metric"], mask = MODEL_PARAMS[model_name]["pai_mask"])
    
    gtex_scores = cp_diffs.mean(axis=1)

    df = pd.read_csv("./data/tableS1D_gtex_eqtls.tsv", header = 0, sep = "\t")

    gtex_outs = get_over_under_null(gtex_scores, vcf, df)
    return gtex_outs

def get_mpra_eqtls_promoter(model, rank, trained_version = "", model_name = "borzoi"):
    benchmark_name = "mpra_eqtls_near_promoter"
    vcf_name ="./data/tableS1E_mpra_eqtls.vcf"
    cp_diffs, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, model_name = model_name, metric = MODEL_PARAMS[model_name]["pai_metric"], mask = MODEL_PARAMS[model_name]["pai_mask"])
    gtex_scores = cp_diffs.mean(axis=1)
    
    df = pd.read_csv("./data/tableS1E_mpra_eqtls.tsv", header = 0, sep = "\t")
   
    gtex_outs = get_over_under_null(gtex_scores, vcf, df)
    return gtex_outs

def get_gtex_outliers_promoter(model, rank, trained_version = "", model_name = "borzoi"):
    benchmark_name = "gtex_outliers_near_promoter"
    vcf_name ="./data/tableS1A_gtex_outliers.vcf"
    
    cp_diffs, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, model_name = model_name, metric = MODEL_PARAMS[model_name]["pai_metric"], mask =  MODEL_PARAMS[model_name]["pai_mask"])
    gtex_scores = cp_diffs.mean(axis=1)

    df = pd.read_csv("./data/tableS1A_gtex_outliers.tsv", header = 0, sep = "\t")
   
    gtex_outs = get_over_under_null(gtex_scores, vcf, df)
    return gtex_outs

def get_cagi5_sat_promoter(model, rank, trained_version = "", model_name = "borzoi"):
    benchmark_name = "cagi5_sat_near_promoter"
    vcf_name ="./data/tableS1B_cagi5_saturation.vcf"
    cp_diffs, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, model_name = model_name, metric = MODEL_PARAMS[model_name]["pai_metric"], mask = MODEL_PARAMS[model_name]["pai_mask"])
    gtex_scores = cp_diffs.mean(axis=1)
    df = pd.read_csv("./data/tableS1B_cagi5_saturation.tsv", header = 0, sep = "\t")
   
    gtex_outs = get_over_under_null(gtex_scores, vcf, df)
    return gtex_outs

def get_mpra_sat_promoter(model, rank, trained_version = "", model_name = "borzoi"):
    benchmark_name = "mpra_sat_near_promoter"
    vcf_name ="./data/tableS1C_mpra_saturation.vcf"
    cp_diffs, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, model_name = model_name, metric = MODEL_PARAMS[model_name]["pai_metric"], mask = MODEL_PARAMS[model_name]["pai_mask"])
    gtex_scores = cp_diffs.mean(axis=1)

    df = pd.read_csv("./data/tableS1C_mpra_saturation.tsv", header = 0, sep = "\t")
   
    gtex_outs = get_over_under_null(gtex_scores, vcf, df)
    return gtex_outs

def get_ukbb_proteome_promoter(model, rank, trained_version = "", model_name = "borzoi"):
    benchmark_name = "ukbb_proteome_near_promoter"
    vcf_name ="./data/tableS1F_ukbb_proteome.vcf"
    cp_diffs, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, model_name = model_name, metric = MODEL_PARAMS[model_name]["pai_metric"], mask = MODEL_PARAMS[model_name]["pai_mask"])
    gtex_scores = cp_diffs.mean(axis=1)

    df = pd.read_csv("./data/tableS1F_ukbb_proteome.tsv", header = 0, sep = "\t")
   
    gtex_outs = get_over_under_null(gtex_scores, vcf, df)
    return gtex_outs

def get_gel_rna_promoter(model, rank, trained_version = "", model_name = "borzoi"):
    benchmark_name = "gel_rna_near_promoter"
    vcf_name ="./data/tableS1G_gel_rna.vcf"
    cp_diffs, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, model_name = model_name, metric = MODEL_PARAMS[model_name]["pai_metric"], mask = MODEL_PARAMS[model_name]["pai_mask"])
    gtex_scores = cp_diffs.mean(axis=1)

    df = pd.read_csv("./data/tableS1G_gel_rna.tsv", header = 0, sep = "\t")
   
    gtex_outs = get_over_under_null(gtex_scores, vcf, df)
    return gtex_outs

def get_over_under_null(scores, vcf, df):
    df_pred = pd.DataFrame(vcf, columns=["CHROM", "POS", "NAME", "REF", "ALT"])
    df_pred["POS"] = df_pred["POS"].astype(int)
    df_pred["score"] = scores

    df_ou = df[df['consequence'].isin(['over', 'under'])].copy()
    df_combine_ou = df_ou.merge(df_pred, left_on = ["chrom", "pos", "ref", "alt"], right_on=["CHROM", "POS", "REF", "ALT"], how = "inner")
    binary_labels_ou = (df_combine_ou['consequence'] == 'over')
    roc_promoter_ou = roc_auc_score(binary_labels_ou, df_combine_ou["score"])

    df_un = df[df['consequence'].isin(['under', 'none'])].copy()
    df_combine_un = df_un.merge(df_pred, left_on = ["chrom", "pos", "ref", "alt"], right_on=["CHROM", "POS", "REF", "ALT"], how = "inner")
    binary_labels_un = (df_combine_un['consequence'] == 'under')
    roc_promoter_un = roc_auc_score(binary_labels_un, -df_combine_un["score"])

    df_on = df[df['consequence'].isin(['over', 'none'])].copy()
    df_combine_on = df_on.merge(df_pred, left_on = ["chrom", "pos", "ref", "alt"], right_on=["CHROM", "POS", "REF", "ALT"], how = "inner")
    binary_labels_on = (df_combine_on['consequence'] == 'over')
    roc_promoter_on = roc_auc_score(binary_labels_on, df_combine_on["score"])
    return roc_promoter_ou, roc_promoter_un, roc_promoter_on 


def get_eu_lcl_caqtls(model, rank, trained_version = "", model_name = "borzoi", axis=None, color = None):
    benchmark_name = "caqtls_eu_GM12878"
    vcf_name ="./data/caqtls.eu.lcls.benchmarking.all.vcf"
    cp_diffs, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, model_name = model_name)
    lcl_diffs = cp_diffs[:, MODEL_PARAMS[model_name]["gm12878"]]
    lcl_scores = lcl_diffs.mean(axis=1)

    df = pd.read_csv("./data/caqtls.eu.lcls.benchmarking.tsv", header = 0, sep = "\t")
    df = df[df["var.isused"]]
    df["log10p"] = np.log10(df["obs.pval"])*-1
   
    df_pred = pd.DataFrame(vcf, columns=["CHROM", "POS", "NAME", "REF", "ALT"])
    df_pred["POS"] = df_pred["POS"].astype(int)
    df_pred["GM12878_DNase_cp_mean"] = lcl_scores


    dataf1 = df[df["log10p"]>6]
    dataf2 = df[df["log10p"]<3]
    dataf1.loc[:, "obs.label"] = 1
    dataf2.loc[:, "obs.label"] = 0

    dataf = pd.concat([dataf1, dataf2])
    df_combine = dataf.merge(df_pred, left_on = ["var.chr", "var.pos_hg38"], right_on=["CHROM", "POS"], how = "inner")
    ap_unsigned = average_precision_score(df_combine["obs.label"], abs(df_combine["GM12878_DNase_cp_mean"]))

    df_combine = df_combine[df_combine["log10p"]>6]
    pearson_signed = scipy.stats.pearsonr(df_combine["GM12878_DNase_cp_mean"],df_combine["obs.beta"])
   
    return pearson_signed , ap_unsigned

def get_yoruba_lcl_dsqtls(model, rank, trained_version = "", model_name = "borzoi", axis = None, color = None):
    benchmark_name = "dsqtls_yoruba_GM12878"
    vcf_name ="./data/dsqtls.yoruba.lcls.benchmarking.all.vcf"
    cp_diffs, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, model_name = model_name)
    lcl_diffs = cp_diffs[:, MODEL_PARAMS[model_name]["gm12878"]]
    lcl_scores = lcl_diffs.mean(axis=1)
    
    df = pd.read_csv("./data/dsqtls.yoruba.lcls.benchmarking.all.tsv", index_col=False,  header = 0, sep = "\t")
    df = df[df["var.isused"]]
    # df = df[df["obs.label"] ==1]
  
    df_pred = pd.DataFrame(vcf, columns=["CHROM", "POS", "NAME", "REF", "ALT"])
    df_pred["POS"] = df_pred["POS"].astype(int)
    df_pred["GM12878_DNase_cp_mean"] = lcl_scores

    df_combine = df.merge(df_pred, left_on = ["var.chr", "var.pos_hg38"], right_on=["CHROM", "POS"], how = "inner")
    ap_unsigned = average_precision_score(df_combine["obs.label"], abs(df_combine["GM12878_DNase_cp_mean"]))

    df_combine = df_combine[df_combine["obs.label"] ==1]
    pearson_signed = scipy.stats.pearsonr(-df_combine["GM12878_DNase_cp_mean"],df_combine["obs.estimate"])
    
    return pearson_signed, ap_unsigned

def get_afr_lcl_caqtls(model, rank, trained_version = "", model_name = "borzoi", axis = None, color = None):
    benchmark_name = "caqtls_african_GM12878"
    vcf_name ="./data/caqtls.african.lcls.benchmarking.all.vcf"

    cp_diffs, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, model_name = model_name)
    lcl_diffs = cp_diffs[:, MODEL_PARAMS[model_name]["gm12878"]]
    lcl_scores = lcl_diffs.mean(axis=1)

    df = pd.read_csv("./data/caqtls.african.lcls.benchmarking.tsv", header = 0, sep = "\t")
    df = df[df["var.isused"]]
    df = df.dropna(subset=["obs.label"])
    # df["log10p"] = np.log10(df["obs.pval"])*-1

    df_pred = pd.DataFrame(vcf, columns=["CHROM", "POS", "NAME", "REF", "ALT"])
    df_pred["POS"] = df_pred["POS"].astype(int)
    df_pred["GM12878_DNase_cp_mean"] = lcl_scores

    df_combine = df.merge(df_pred, left_on = ["var.chr", "var.pos_hg38"], right_on=["CHROM", "POS"], how = "inner")
    ap_unsigned = average_precision_score(df_combine["obs.label"], abs(df_combine["GM12878_DNase_cp_mean"]))

    df_combine = df_combine[df_combine["obs.label"] ==1]
    pearson_signed = scipy.stats.pearsonr(df_combine["GM12878_DNase_cp_mean"],df_combine["obs.beta"])
  
    return pearson_signed , ap_unsigned

def get_microglia_caqtls(model, rank, trained_version = "", model_name = "borzoi", axis = None, color = None):
    benchmark_name = "caqtls_eu_Microglia"
    vcf_name ="./data/caqtls.microglia.benchmarking.vcf"
    
    cp_diffs, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, model_name = model_name, mask = MODEL_PARAMS[model_name]["cbpnet_mask"], metric=MODEL_PARAMS[model_name]["cbpnet_metric"])
    lcl_diffs = cp_diffs[:, MODEL_PARAMS[model_name]["microglia"]]
    lcl_scores = lcl_diffs.mean(axis=1)
    
    df = pd.read_csv("./data/caqtls.microglia.benchmarking.all.tsv", header = 0, sep = "\t")
    df = df[df["var.isused"]]
    df = df.dropna().drop_duplicates()

    df_pred = pd.DataFrame(vcf, columns=["CHROM", "POS", "NAME", "REF", "ALT"])
    df_pred["POS"] = df_pred["POS"].astype(int)
    df_pred["Macrophage_ATAC_cp_mean"] = lcl_scores

    df_combine = df.merge(df_pred, left_on = ["var.chr", "var.pos_hg38", "var.allele1", "var.allele2"], right_on=["CHROM", "POS", "REF", "ALT"], how = "inner")
    df_combine = df_combine.dropna().drop_duplicates()
    df_combine.to_csv(f"scores/caqtls_eu_microglia_{model_name}_rank{rank}.tsv.gz", sep="\t", index=False, compression="gzip")
    pearson_signed = scipy.stats.pearsonr(-df_combine["Macrophage_ATAC_cp_mean"],df_combine["obs.Beta"])
 
    return pearson_signed

def get_smc_caqtls(model, rank, trained_version = "", model_name = "borzoi", axis = None, color = None):
    benchmark_name = "caqtls_eu_SMC"
    vcf_name ="./data/caqtls.smc.benchmarking.all.vcf"
    
    cp_diffs, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, model_name = model_name)
    lcl_diffs = cp_diffs[:, MODEL_PARAMS[model_name]["smc"]]
    lcl_scores = lcl_diffs.mean(axis=1)

    df = pd.read_csv("./data/caqtls.smc.benchmarking.tsv", header = 0, sep = "\t")
    df = df.dropna()
    df = df[df["var.isused"]]

    df_pred = pd.DataFrame(vcf, columns=["CHROM", "POS", "NAME", "REF", "ALT"])
    df_pred["POS"] = df_pred["POS"].astype(int)
    df_pred["SMC_ATAC_cp_mean"] = lcl_scores

    df_combine = df.merge(df_pred, left_on = ["var.chr", "var.pos_hg38"], right_on=["CHROM", "POS"], how = "inner")
    df_combine.to_csv(f"scores/caqtls_eu_smc_{model_name}_rank{rank}.tsv.gz", sep="\t", index=False, compression="gzip")
    pearson_signed = scipy.stats.pearsonr(df_combine["SMC_ATAC_cp_mean"],df_combine["obs.Effect_size"])

    return pearson_signed

def get_spi1_bqtls(model, rank, trained_version = "", model_name = "borzoi", axis = None, color = None):
    benchmark_name = "bqtls_spi1_LCL"
    vcf_name ="./data/bqtls.pu1.lcls.benchmarking.all.vcf"
    
    cp_diffs, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, model_name = model_name)
    lcl_diffs = cp_diffs[:, MODEL_PARAMS[model_name]["spi1"]]
    lcl_scores = lcl_diffs.mean(axis=1)

    df = pd.read_csv("./data/bqtls.pu1.lcls.benchmarking.tsv", header = 0, sep = "\t")
    df = df[df["var.isused"]]

    df_pred = pd.DataFrame(vcf, columns=["CHROM", "POS", "NAME", "REF", "ALT"])
    df_pred["POS"] = df_pred["POS"].astype(int)
    df_pred["GM12878_spi1_cp_mean"] = lcl_scores

    df_combine = df.merge(df_pred, left_on = ["var.chr", "var.pos_hg38"], right_on=["CHROM", "POS"], how = "inner")
    df_combine = df_combine[df_combine["obs.pval"] < 1e-9]
    df_combine.to_csv(f"scores/bsqtls_eu_spi1_{model_name}_rank{rank}.tsv.gz", sep="\t", index=False, compression="gzip")
    pearson_signed = scipy.stats.pearsonr(df_combine["GM12878_spi1_cp_mean"],df_combine["obs.chiplogratio"])
    
    return pearson_signed


def get_variants(model, vcf, rank, benchmark_name="", trained_version = "", model_name = "borzoi", metric = "log2_diff", mask = "center"):
    dataset = VariantDataset(file_path=vcf, window_size = MODEL_PARAMS[model_name]["input_len"])
    dataloader = VariantDataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    eps = 1e-8
    if mask == "gene":
        genes_df = read_gtf_to_df("./resources/gencode.v49.basic.annotation.gtf")
    if mask == "exon":
        genes_df = read_gtf_exons("./resources/gencode.v49.basic.annotation.gtf")


    all_diffs = []
    all_vcf = []

    progress_bar = tqdm(dataloader, desc=f"Running {rank} {trained_version} {benchmark_name} benchmark")

    for batch in progress_bar:
        if batch is None: 
            continue
        ref, alt, vcf = batch
        ref, alt = ref.to(device), alt.to(device)
        ref_outputs = model(ref)  # both are tuples: (refproj, altproj)
        alt_outputs = model(alt)
    
        ref_outputs, alt_outputs =  ref_outputs.detach().cpu(), alt_outputs.detach().cpu()
        if mask =="center":
            ref_window = get_center(ref_outputs, 501, bin_length = MODEL_PARAMS[model_name]["bin_length"])
            alt_window = get_center(alt_outputs, 501, bin_length = MODEL_PARAMS[model_name]["bin_length"])
        if mask =="center_8":
            bins = MODEL_PARAMS[model_name]["bin_length"]
            ref_window = get_center(ref_outputs, bins*8, bin_length = MODEL_PARAMS[model_name]["bin_length"])
            alt_window = get_center(alt_outputs, bins*8, bin_length = MODEL_PARAMS[model_name]["bin_length"])
        elif mask == "gene":
            ref_window = get_gene(ref_outputs, vcf, genes_df, bin_length = MODEL_PARAMS[model_name]["bin_length"])
            alt_window = get_gene(alt_outputs, vcf, genes_df, bin_length = MODEL_PARAMS[model_name]["bin_length"])
        elif mask == "all":
            ref_window  = ref_outputs
            alt_window  = alt_outputs
        elif mask == "exon":
            ref_window = get_exons(ref_outputs, vcf, genes_df, bin_length = MODEL_PARAMS[model_name]["bin_length"])
            alt_window = get_exons(alt_outputs, vcf, genes_df, bin_length = MODEL_PARAMS[model_name]["bin_length"])
            
            



        if metric == "log2_diff":
            ref_sum = ref_window.sum(dim=-1)  # [B, C]
            alt_sum = alt_window.sum(dim=-1)  # [B, C]
            diffs = torch.log2((alt_sum + 1) / (ref_sum + 1))
        elif metric == "SAR":
            div_window = torch.log2((alt_window + 1)) - torch.log2((ref_window + 1))
            diffs = div_window.sum(dim=-1)
        elif metric == "SAD":
            div_window = (alt_window  - ref_window)
            diffs = div_window.sum(dim=-1)

    
        all_diffs.append(diffs)
        all_vcf.append(vcf)

        # Accumulate by appending to list
    
    all_diffs = torch.cat([t.detach().cpu() for t in all_diffs], dim=0).numpy()
    all_vcf = np.concatenate(all_vcf, axis=0)
    model = model.to("cpu")
    torch.cuda.empty_cache()

    return all_diffs, all_vcf

def get_scores(model, bed, rank, benchmark_name="", trained_version = "", model_name = "borzoi", scores = None, sc = False):
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

def save_output(rank_l = 256, trained_version = "", quant = False, model_name = "borzoi",  bpn_axes = None, color = None, full = False):
    if quant:
        q = "quant"
    else:
        q = "no_quant"

    model_name_rank = f"{model_name}_lora_lr{rank_l}_{q}" 


    model = initialize_models(k_l = rank_l, quant = quant, model_name = model_name, full = full)

    # ## PromoterAI
    gtex_eqtls = get_gtex_eqtls_promoter(model = model, rank = rank_l, trained_version = trained_version, model_name = model_name)
    mpra_eqtls = get_mpra_eqtls_promoter(model = model, rank = rank_l, trained_version = trained_version, model_name = model_name)
    gtex_outliers = get_gtex_outliers_promoter(model = model, rank = rank_l, trained_version = trained_version, model_name = model_name)
    cagi5_sat = get_cagi5_sat_promoter(model = model, rank = rank_l, trained_version = trained_version, model_name = model_name)
    mpra_sat = get_mpra_sat_promoter(model = model, rank = rank_l, trained_version = trained_version, model_name = model_name)
    ukbb_proteome = get_ukbb_proteome_promoter(model = model, rank = rank_l, trained_version = trained_version, model_name = model_name)
    gel_rna = get_gel_rna_promoter(model = model, rank = rank_l, trained_version = trained_version, model_name = model_name)

    pai_path = f"{model_name}_fixed_benchmark_promoterai.tsv"
    pai_row_dict = {
       "model": model_name_rank,
       "GTEX_eqtl_OvU_promoter": round(gtex_eqtls[0], 4),
       "GTEX_eqtl_outliers_OvU_promoter": round(gtex_outliers[0], 4),
       "CAGI5_saturation_OvU_promoter": round(cagi5_sat[0], 4),
       "MPRA_saturation_OvU_promoter": round(mpra_sat[0], 4),
       "MPRA_eqtl_OvU_promoter": round(mpra_eqtls[0], 4),
       "UKBB_proteome_OvU_promoter": round(ukbb_proteome[0], 4),
       "Gel_RNA_OvU_promoter": round(gel_rna[0], 4),

       "GTEX_eqtl_UvN_promoter": round(gtex_eqtls[1], 4),
       "GTEX_eqtl_outliers_UvN_promoter": round(gtex_outliers[1], 4),
       "CAGI5_saturation_UvN_promoter": round(cagi5_sat[1], 4),
       "MPRA_saturation_UvN_promoter": round(mpra_sat[1], 4),
       "MPRA_eqtl_UvN_promoter": round(mpra_eqtls[1], 4),
       "UKBB_proteome_UvN_promoter": round(ukbb_proteome[1], 4),
       "Gel_RNA_UvN_promoter": round(gel_rna[1], 4),

       "GTEX_eqtl_OvN_promoter": round(gtex_eqtls[2], 4),
       "GTEX_eqtl_outliers_OvN_promoter":round(gtex_outliers[2], 4),
       "CAGI5_saturation_OvN_promoter": round(cagi5_sat[2], 4),
       "MPRA_saturation_OvN_promoter": round(mpra_sat[2], 4),
       "MPRA_eqtl_OvN_promoter": round(mpra_eqtls[2], 4),
       "UKBB_proteome_OvN_promoter": round(ukbb_proteome[2], 4),
       "Gel_RNA_OvN_promoter": round(gel_rna[2], 4),
   }
    file_exists = os.path.isfile(pai_path)
    with open(pai_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=pai_row_dict.keys(), delimiter="\t")
            if not file_exists:
                writer.writeheader()
            writer.writerow(pai_row_dict)

            print(f"Results saved to {pai_path}")
    ## ChrombpNet
    
    yoruba_pearson, yoruba_ap = get_yoruba_lcl_dsqtls(model = model, rank = rank_l,  trained_version = trained_version, model_name = model_name)
    eu_pearson, eu_ap = get_eu_lcl_caqtls(model = model, rank = rank_l,  trained_version = trained_version, model_name = model_name)
    afr_pearson, afr_ap = get_afr_lcl_caqtls(model = model, rank = rank_l,  trained_version = trained_version, model_name = model_name)
    microglia_pearson = get_microglia_caqtls(model = model, rank = rank_l, trained_version = trained_version, model_name = model_name)
    smc_pearson = get_smc_caqtls(model = model, rank = rank_l,  trained_version = trained_version, model_name = model_name)
    spi1_pearson = get_spi1_bqtls(model = model, rank = rank_l, trained_version = trained_version, model_name = model_name)

    bpn_path = f"micro_benchmark_chrombpnet_full.tsv"
    bpn_row_dict = {
        "model": model_name_rank,
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

    # file_exists = os.path.isfile(bpn_path)
    # with open(bpn_path, "a", newline="") as f:
    #        writer = csv.DictWriter(f, delimiter="\t")
    #        if not file_exists:
    #            writer.writeheader()
    #        writer.writerow(microglia_pearson)

    #        print(f"Results saved to {bpn_path}")
    ## Scores

    # diagnostic_sei_sc = get_sc_sequence_metrics(model = sc_seq_mod, rank = rank, trained_version = trained_version, scores_path = "../..//filtered_sequence_segments.raw_sequence_class_scores.npy", bed_path = "../../filtered_sequence_segments_score.bed")
    # diagnostic_sei_preds = get_cp_sequence_metrics(model = cp_seq_mod, rank = rank, trained_version = trained_version, scores_path = "../../sei_predictions/chromatin-profiles-hdf5/filtered_sequence_segments_score_predictions.h5", bed_path = "../../filtered_sequence_segments_score.bed")
    # diagnostic_cistrome = get_cp_sequence_metrics(model = cp_seq_mod, rank = rank, trained_version = trained_version, scores_path = "./filtered_sequence_segments_cistrome.h5", bed_path = "../../filtered_sequence_segments_score.bed")

    # scores_path = "benchmark_scores.tsv"    
    # scores_row_dict = {
    #     "model": model_name_rank,
    #     "CP_Teacher_Pearson": round(diagnostic_sei_preds["avg_pearson"], 4),
    #     "CP_Teacher_Spearman": round(diagnostic_sei_preds["avg_spearman"], 4),
    #     "CP_Teacher_AP": round(diagnostic_sei_preds["avg_ap"], 4),
    #     "CP_Teacher_MCC": round(diagnostic_sei_preds["avg_mcc"], 4),
    #     "CP_Teacher_F1": round(diagnostic_sei_preds["avg_f1"], 4),
    #     "CP_Teacher_AUROC": round(diagnostic_sei_preds["avg_auroc"], 4),

    #     "CP_Cistrome_Pearson": round(diagnostic_cistrome["avg_pearson"], 4),
    #     "CP_Cistrome_Spearman": round(diagnostic_cistrome["avg_spearman"], 4),
    #     "CP_Cistrome_AP": round(diagnostic_cistrome["avg_ap"], 4),
    #     "CP_Cistrome_MCC": round(diagnostic_cistrome["avg_mcc"], 4),
    #     "CP_Cistrome_F1": round(diagnostic_cistrome["avg_f1"], 4),
    #     "CP_Cistrome_AUROC": round(diagnostic_cistrome["avg_auroc"], 4),

    #     "SC_Teacher_Pearson": round(diagnostic_sei_sc["avg_pearson"], 4),
    #     "SC_Teacher_Spearman": round(diagnostic_sei_sc["avg_spearman"], 4),
    #     "SC_Teacher_AP": round(diagnostic_sei_sc["avg_ap"], 4),
    #     "SC_Teacher_MCC": round(diagnostic_sei_sc["avg_mcc"], 4),
    #     "SC_Teacher_F1": round(diagnostic_sei_sc["avg_f1"], 4),
    #     "SC_Teacher_AUROC": round(diagnostic_sei_sc["avg_auroc"], 4),
    # }

    # file_exists = os.path.isfile(scores_path)
    # with open(scores_path, "a", newline="") as f:
    #         writer = csv.DictWriter(f, fieldnames=scores_row_dict.keys(), delimiter="\t")
    #         if not file_exists:
    #             writer.writeheader()
    #         writer.writerow(scores_row_dict)

    #         print(f"Results saved to {scores_path}")

    # diagnostic_sei_sc_cCREs = get_sc_sequence_metrics(model = sc_seq_mod, rank = rank, trained_version = trained_version, scores_path = "./GRCh38_cCREs_4kb.h5.raw_sequence_class_scores.npy", bed_path = "./GRCh38_cCREs_4kb.bed")
    # diagnostic_sei_preds_cCREs = get_cp_sequence_metrics(model = cp_seq_mod, rank = rank, trained_version = trained_version, scores_path = "./GRCh38_cCREs_4kb.h5", bed_path = "./GRCh38_cCREs_4kb.bed")
    # diagnostic_cistrome_cCREs = get_cp_sequence_metrics(model = cp_seq_mod, rank = rank, trained_version = trained_version, scores_path = "./GRCh38_cCREs_4kb_cistrome.h5", bed_path = "./GRCh38_cCREs_4kb.bed")
    


    # # dictionary of results
    # scores_cres_path = "benchmark_scores_cCRE_regions.tsv"
    # scores_cres_row_dict = {
    #     "model": model_name,
    #     "CP_cCREs_Teacher_Pearson": round(diagnostic_sei_preds_cCREs["avg_pearson"], 4),
    #     "CP_cCREs_Teacher_Spearman": round(diagnostic_sei_preds_cCREs["avg_spearman"], 4),
    #     "CP_cCREs_Teacher_AP": round(diagnostic_sei_preds_cCREs["avg_ap"], 4),
    #     "CP_cCREs_Teacher_MCC": round(diagnostic_sei_preds_cCREs["avg_mcc"], 4),
    #     "CP_cCREs_Teacher_F1": round(diagnostic_sei_preds_cCREs["avg_f1"], 4),
    #     "CP_cCREs_Teacher_AUROC": round(diagnostic_sei_preds_cCREs["avg_auroc"], 4),

    #     "CP_cCREs_Cistrome_Pearson": round(diagnostic_cistrome_cCREs["avg_pearson"], 4),
    #     "CP_cCREs_Cistrome_Spearman": round(diagnostic_cistrome_cCREs["avg_spearman"], 4),
    #     "CP_cCREs_Cistrome_AP": round(diagnostic_cistrome_cCREs["avg_ap"], 4),
    #     "CP_cCREs_Cistrome_MCC": round(diagnostic_cistrome_cCREs["avg_mcc"], 4),
    #     "CP_cCREs_Cistrome_F1": round(diagnostic_cistrome_cCREs["avg_f1"], 4),
    #     "CP_cCREs_Cistrome_AUROC": round(diagnostic_cistrome_cCREs["avg_auroc"], 4),

    #     # "SC_cCREs_Teacher_Pearson": round(diagnostic_sei_sc_cCREs["avg_pearson"], 4),
    #     # "SC_cCREs_Teacher_Spearman": round(diagnostic_sei_sc_cCREs["avg_spearman"], 4),
    #     # "SC_cCREs_Teacher_AP": round(diagnostic_sei_sc_cCREs["avg_ap"], 4),
    #     # "SC_cCREs_Teacher_MCC": round(diagnostic_sei_sc_cCREs["avg_mcc"], 4),
    #     # "SC_cCREs_Teacher_F1": round(diagnostic_sei_sc_cCREs["avg_f1"], 4),
    #     # "SC_cCREs_Teacher_AUROC": round(diagnostic_sei_sc_cCREs["avg_auroc"], 4),
    # }

    # file_exists = os.path.isfile(scores_cres_path)
    # with open(scores_cres_path, "a", newline="") as f:
    #         writer = csv.DictWriter(f, fieldnames=scores_cres_row_dict.keys(), delimiter="\t")
    #         if not file_exists:
    #             writer.writeheader()
    #         writer.writerow(scores_cres_row_dict)

    #         print(f"Results saved to {scores_cres_path}")

    torch.cuda.empty_cache()

def main():


    # save_output(rank_l = 1,  quant = False, model_name = "borzoi")
    # save_output(rank_l = 2,  quant = False, model_name = "borzoi")
    # save_output(rank_l = 4,  quant = False, model_name = "borzoi")
    # save_output(rank_l = 8,  quant = False, model_name = "borzoi")
    # save_output(rank_l = 16,  quant = False, model_name = "borzoi")
    # save_output(rank_l = 32,  quant = False, model_name = "borzoi")
    # save_output(rank_l = 64,  quant = False, model_name = "borzoi")
    # save_output(rank_l = 128,  quant = False, model_name = "borzoi")
    # save_output(rank_l = 256,  quant = False, model_name = "borzoi")
    # save_output(rank_l = 512,  quant = False, model_name = "borzoi")
    # save_output(rank_l = "full",  quant = False, model_name = "borzoi")


#####

    # save_output(rank_l = 1,  quant = False, model_name = "enformer")
    # save_output(rank_l = 2,  quant = False, model_name = "enformer")
    # save_output(rank_l = 4,  quant = False, model_name = "enformer")
    # save_output(rank_l = 8,  quant = False, model_name = "enformer")
    # save_output(rank_l = 16,  quant = False, model_name = "enformer")
    # save_output(rank_l = 32,  quant = False, model_name = "enformer")
    # save_output(rank_l = 64,  quant = False, model_name = "enformer")
    # save_output(rank_l = 128,  quant = False, model_name = "enformer")
    # save_output(rank_l = 256,  quant = False, model_name = "enformer")
    # save_output(rank_l = 512,  quant = False, model_name = "enformer")
    # save_output(rank_l = "full",  quant = False, model_name = "enformer")

    

    # save_output(rank_l = "full",  quant = False, model_name = "borzoi", full = True)
    save_output(rank_l = "full",  quant = False, model_name = "enformer", full = True)

if __name__ == '__main__':
    main()
