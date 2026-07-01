#!/usr/bin/env python 
import os, sys, csv

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from sei_lora.dataloaders import VariantDataset, SeqDataLoader, SeqDataset, VariantDataLoader
from tqdm import tqdm
import scipy

from sklearn.metrics import average_precision_score, matthews_corrcoef, f1_score, roc_auc_score

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import math

from ag_arch import AlphaGenome
from alphagenome_pytorch import AlphaGenome as AlphaGenomePT
from borzoi_lora_arch_mha import BorzoiModel, EnformerModel
import grelu.resources


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_PARAMS = {
    "ag": {
        "input_len": 1048576,
        "bin_length": 1,
        "model_class": AlphaGenome,
        "kwargs": {},
        "gm12878": [12, 69],
        "microglia": [41, 131, 392, 508, 517],
        "smc": [82],
        "spi1": [907],
        "pai_mask": "exon",
        "pai_metric": "log2_diff_ag",
    }
}


# ---------------------------------------------------------------------------
# Scorer config helpers
# ---------------------------------------------------------------------------

def _normalize_scorers(pai_mask, pai_metric, pai_track, pai_res):
    """
    Accept either scalar strings or equal-length tuples/lists for
    (pai_mask, pai_metric, pai_track, pai_res) and return a list of
    scorer dicts:
        [{"mask": ..., "metric": ..., "output_head": ..., "output_resolution": ...}, ...]
    """
    # Normalise everything to lists
    def _to_list(x):
        return list(x) if isinstance(x, (tuple, list)) else [x]

    masks   = _to_list(pai_mask)
    metrics = _to_list(pai_metric)
    tracks  = _to_list(pai_track)
    ress    = _to_list(pai_res)

    # Allow a scalar resolution to be broadcast across all scorers
    if len(ress) == 1 and len(masks) > 1:
        ress = ress * len(masks)

    lengths = {len(masks), len(metrics), len(tracks), len(ress)}
    if len(lengths) != 1:
        raise ValueError(
            f"pai_mask, pai_metric, pai_track, pai_res must all have the same length. "
            f"Got lengths: mask={len(masks)}, metric={len(metrics)}, "
            f"track={len(tracks)}, res={len(ress)}"
        )

    return [
        {"mask": m, "metric": mt, "output_head": tr, "output_resolution": r}
        for m, mt, tr, r in zip(masks, metrics, tracks, ress)
    ]


# ---------------------------------------------------------------------------
# GTF helpers
# ---------------------------------------------------------------------------

def read_gtf_to_df(gtf_file):
    gtf_cols = ["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
    df = pd.read_csv(gtf_file, sep="\t", comment="#", names=gtf_cols)
    df = df[(df["feature"] == "transcript") & (df["attribute"].str.contains('gene_type "protein_coding"'))]
    df["chrom"] = df["chrom"].astype(str).str.replace("chr", "", regex=False)
    return df[["chrom", "start", "end"]]


def read_gtf_exons(gtf_file):
    gtf_cols = ["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
    df = pd.read_csv(gtf_file, sep="\t", comment="#", names=gtf_cols)
    df = df[(df["feature"] == "exon") & (df["attribute"].str.contains('gene_type "protein_coding"'))].copy()
    df["transcript_id"] = df["attribute"].apply(
        lambda x: x.split('transcript_id "')[1].split('"')[0] if 'transcript_id "' in x else None
    )
    df["gene_name"] = df["attribute"].apply(
        lambda x: x.split('gene_name "')[1].split('"')[0] if 'gene_name "' in x else None
    )
    df["chrom"] = df["chrom"].astype(str).str.replace("chr", "", regex=False)
    return df[["chrom", "start", "end", "gene_name", "transcript_id"]]


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------

def get_center(tensor: torch.Tensor, center_bins: int):
    seq_len = tensor.shape[-1]
    if center_bins > seq_len:
        center_bins = seq_len  # clamp to available length
    start = (seq_len - center_bins) // 2
    end = start + center_bins
    return tensor[..., start:end]


def get_gene(ref_outputs, vcf, genes_df, bin_length):
    chr_ = str(vcf[0][0]).replace("chr", "")
    pos  = int(vcf[0][1])
    B, C, L = ref_outputs.shape
    half_bp = (L * bin_length) // 2
    model_start = pos - half_bp
    model_end   = pos + half_bp
    chr_genes = genes_df[genes_df["chrom"] == chr_]
    if chr_genes.empty:
        raise ValueError(f"No genes found on chromosome {chr_}")
    overlap = chr_genes[(chr_genes["end"] >= model_start) & (chr_genes["start"] <= model_end)].copy()
    if overlap.empty:
        return get_center(ref_outputs, max(1, math.ceil(1_000 / bin_length)))
    overlap["dist"] = np.minimum(
        (overlap["end"] - pos).abs(), (overlap["start"] - pos).abs()
    )
    gene_row  = overlap.loc[overlap["dist"].idxmin()]
    gene_start, gene_end = int(gene_row["start"]), int(gene_row["end"])
    overlap_start = max(gene_start, model_start)
    overlap_end   = min(gene_end,   model_end)
    bin_start = max(0, (overlap_start - model_start) // bin_length)
    bin_end   = min(L, (overlap_end   - model_start) // bin_length)
    return ref_outputs[:, :, bin_start:bin_end]


def get_exons(ref_outputs, vcf, exons_df, bin_length):
    chr_ = str(vcf[0][0]).replace("chr", "")
    pos  = int(vcf[0][1])
    B, C, L = ref_outputs.shape
    half_bp = (L * bin_length) // 2
    model_start = pos - half_bp
    model_end   = pos + half_bp
    chr_exons = exons_df[exons_df["chrom"] == chr_]
    if chr_exons.empty:
        raise ValueError(f"No exons found on chromosome {chr_}")
    overlap_transcripts = chr_exons[
        (chr_exons["end"] >= model_start) & (chr_exons["start"] <= model_end)
    ]["transcript_id"].unique()
    if len(overlap_transcripts) == 0:
        chr_exons = chr_exons.copy()
        chr_exons["dist"] = ((chr_exons["start"] + chr_exons["end"]) // 2 - pos).abs()
        closest_tx = chr_exons.loc[chr_exons["dist"].idxmin()]["transcript_id"]
    else:
        tx_dist = []
        for tx in overlap_transcripts:
            tx_exons = chr_exons[chr_exons["transcript_id"] == tx]
            dist = np.min(np.minimum((tx_exons["start"] - pos).abs(), (tx_exons["end"] - pos).abs()))
            tx_dist.append((tx, dist))
        closest_tx = min(tx_dist, key=lambda x: x[1])[0]
    tx_exons = chr_exons[chr_exons["transcript_id"] == closest_tx]
    bin_indices = []
    for _, exon in tx_exons.iterrows():
        exon_start = max(exon["start"], model_start)
        exon_end   = min(exon["end"],   model_end)
        start_bin  = max(0, (exon_start - model_start) // bin_length)
        end_bin    = min(L, (exon_end   - model_start) // bin_length)
        if end_bin > start_bin:
            bin_indices.extend(range(start_bin, end_bin))
    bin_indices = sorted(set(bin_indices))
    if len(bin_indices) == 0:
        center_bins = max(1, math.ceil(1_000 / bin_length))
        center_bins = min(L, center_bins)
        start = (L - center_bins) // 2
        return ref_outputs[:, :, start:start + center_bins]
    return ref_outputs[:, :, bin_indices]


def _apply_mask(tensor, mask, vcf_meta, bin_length, genes_df, exons_df):
    """Apply a single mask to a (B, C, L) tensor. Returns a windowed tensor."""
    if mask == "center":
        return get_center(tensor, max(1, math.ceil(501 / bin_length)))
    elif mask == "center_8":
        return get_center(tensor, 8)
    elif mask == "gene":
        return get_gene(tensor, vcf_meta, genes_df, bin_length=bin_length)
    elif mask == "all":
        return tensor
    elif mask == "exon":
        return get_exons(tensor, vcf_meta, exons_df, bin_length=bin_length)
    else:
        raise ValueError(f"Unknown mask: {mask!r}")


def _apply_metric(ref_window, alt_window, metric):
    """Compute a scalar diff tensor from ref/alt windows."""
    if metric == "log2_diff":
        ref_sum = ref_window.sum(dim=-1)
        alt_sum = alt_window.sum(dim=-1)
        return torch.log2((alt_sum + 1) / (ref_sum + 1))
    elif metric == "log2_diff_ag_dnase":
        ref_sum = ref_window.sum(dim=-1)
        alt_sum = alt_window.sum(dim=-1)
        return torch.log2(alt_sum + 1) - torch.log2(ref_sum + 1)
    elif metric == "log2_diff_ag_rna":
        ref_sum = ref_window.mean(dim=-1)
        alt_sum = alt_window.mean(dim=-1)
        return torch.log2(alt_sum + 0.001) - torch.log2(ref_sum + 0.001)
    elif metric == "SAR":
        return (torch.log2(alt_window + 1) - torch.log2(ref_window + 1)).sum(dim=-1)
    elif metric == "SAD":
        return (alt_window - ref_window).sum(dim=-1)
    else:
        raise ValueError(f"Unknown metric: {metric!r}")


# ---------------------------------------------------------------------------
# AlphaGenome head extraction
# ---------------------------------------------------------------------------

def _extract_head(out: dict, head: str, resolution: int) -> torch.Tensor:
    if head in ("splice_site_usage", "splice_junctions"):
        raise ValueError(f"Head '{head}' is not supported for variant scoring.")
    if head == "splice_sites":
        return out[head]["logits"]
    if head == "contact_maps":
        return out[head]
    return out[head][resolution]


# ---------------------------------------------------------------------------
# Core variant scoring  (single pass, multiple scorers)
# ---------------------------------------------------------------------------

def get_ag_variants(
    model,
    vcf,
    rank,
    benchmark_name="",
    trained_version="",
    model_name="ag",
    scorers=None,
    organism_index=0,
    input_len=None,
):
    """
    Run all scorers against a VCF in a single forward pass per variant.

    Args:
        scorers: list of dicts, each with keys:
            mask, metric, output_head, output_resolution

    Returns:
        List of (diffs_array [N, tracks], vcf_array [N, 5]) tuples, one per scorer.
    """
    if scorers is None:
        raise ValueError("scorers must be provided")

    if input_len is None:
        input_len = MODEL_PARAMS[model_name]["input_len"]

    # ── Derive the minimal set of heads/resolutions to compute ───────────────
    needed_heads = set()
    needed_resolutions = set()
    for s in scorers:
        head, res = s["output_head"], s["output_resolution"]
        if head in ("splice_site_usage", "splice_junctions"):
            raise ValueError(f"Head '{head}' is not supported for variant scoring.")
        needed_heads.add(head)
        if head not in ("splice_sites", "contact_maps"):
            needed_resolutions.add(res)

    heads_arg       = tuple(needed_heads)
    resolutions_arg = tuple(needed_resolutions) if needed_resolutions else None

    batch_size = 1
    # print( needed_resolutions)
    # if 1 not in needed_resolutions:
    #     batch_size = batch_size * 2  # can use larger batch if not computing 128bp head
    # print(batch_size)
    if input_len == 4096:
        batch_size = batch_size * 256  # can use larger batch if using smaller input window

    # ── Pre-load GTF tables only if a scorer needs them ──────────────────────
    needs_gene = any("gene" in s["mask"] for s in scorers)
    needs_exon = any("exon" in s["mask"] for s in scorers)
    genes_df = read_gtf_to_df("../resources/gencode.v49.basic.annotation.gtf") if needs_gene else None
    exons_df = read_gtf_exons("../resources/gencode.v49.basic.annotation.gtf") if needs_exon else None

    # ── Dataset / dataloader ─────────────────────────────────────────────────
    dataset    = VariantDataset(file_path=vcf, window_size=input_len)
    dataloader = VariantDataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    org_idx = (
        torch.tensor([organism_index], dtype=torch.long).to(device)
        if organism_index is not None else None
    )

    # ── Accumulators ─────────────────────────────────────────────────────────
    all_diffs_per_scorer = [[] for _ in scorers]
    all_vcf = []

    progress_bar = tqdm(
        dataloader,
        desc=f"Running {rank} {trained_version} {benchmark_name} benchmark",
    )

    with torch.no_grad():
        for batch in progress_bar:
            if batch is None:
                continue

            ref_seq, alt_seq, vcf_meta = batch

            # Single forward pass per allele, computing only needed heads
            ref_seq      = ref_seq.to(device).permute(0, 2, 1)   # (B,L,4)→(B,4,L)
            ref_out_full = model(ref_seq, org_idx, resolutions=resolutions_arg, heads=heads_arg)
            del ref_seq

            alt_seq      = alt_seq.to(device).permute(0, 2, 1)
            alt_out_full = model(alt_seq, org_idx, resolutions=resolutions_arg, heads=heads_arg)
            del alt_seq
            torch.cuda.empty_cache()

            # Score every scorer from the cached model outputs
            for scorer_idx, s in enumerate(scorers):
                head   = s["output_head"]
                res    = s["output_resolution"]
                mask   = s["mask"]
                metric = s["metric"]

                # Extract head → (B, tracks, positions)
                ref_out = _extract_head(ref_out_full, head, res).cpu().permute(0, 2, 1)
                alt_out = _extract_head(alt_out_full, head, res).cpu().permute(0, 2, 1)

                ref_window = _apply_mask(ref_out, mask, vcf_meta, res, genes_df, exons_df)
                alt_window = _apply_mask(alt_out, mask, vcf_meta, res, genes_df, exons_df)

                diffs = _apply_metric(ref_window, alt_window, metric)
                all_diffs_per_scorer[scorer_idx].append(diffs)

            del ref_out_full, alt_out_full
            torch.cuda.empty_cache()
            all_vcf.append(vcf_meta)

    all_vcf = np.concatenate(all_vcf, axis=0)
    results = [
        (
            torch.cat([t.detach().cpu() for t in diffs_list], dim=0).numpy(),
            all_vcf,
        )
        for diffs_list in all_diffs_per_scorer
    ]

    model.cpu()
    torch.cuda.empty_cache()
    return results   # List[(diffs [N, tracks], vcf [N, 5])]


def get_variants(
    model,
    vcf,
    rank,
    benchmark_name="",
    trained_version="",
    model_name="borzoi",
    scorers=None,
    input_len=None,
):
    """
    Unified variant scoring entry point.

    For AG models: single forward pass covering all scorers.
    For other models: scorers must all share the same mask/metric
                      (track selection happens downstream via column indexing).

    Returns:
        List of (diffs_array, vcf_array) tuples, one per scorer.
    """
    if scorers is None:
        raise ValueError("scorers must be provided")

    if input_len is None:
        input_len = MODEL_PARAMS[model_name]["input_len"]

    if model_name == "ag":
        return get_ag_variants(
            model=model, vcf=vcf, rank=rank,
            benchmark_name=benchmark_name, trained_version=trained_version,
            model_name=model_name, scorers=scorers,
            input_len=input_len,
        )

    # ── Non-AG path (Borzoi, Enformer …) ────────────────────────────────────
    # These models return a single tensor; scorers share a forward pass but
    # differ only in mask/metric (track slicing is done by the caller).
    dataset    = VariantDataset(file_path=vcf, window_size=input_len)
    dataloader = VariantDataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=15)
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    needs_gene = any("gene" in s["mask"] for s in scorers)
    needs_exon = any("exon" in s["mask"] for s in scorers)
    genes_df   = read_gtf_to_df("../resources/gencode.v49.basic.annotation.gtf") if needs_gene else None
    exons_df   = read_gtf_exons("../resources/gencode.v49.basic.annotation.gtf") if needs_exon else None

    all_diffs_per_scorer = [[] for _ in scorers]
    all_vcf = []

    progress_bar = tqdm(dataloader, desc=f"Running {rank} {trained_version} {benchmark_name} benchmark")

    for batch in progress_bar:
        if batch is None:
            continue
        ref, alt, vcf_meta = batch
        ref, alt = ref.to(device), alt.to(device)

        ref_outputs = model(ref).detach().cpu()
        alt_outputs = model(alt).detach().cpu()

        for scorer_idx, s in enumerate(scorers):
            ref_window = _apply_mask(ref_outputs, s["mask"], vcf_meta, s["output_resolution"], genes_df, exons_df)
            alt_window = _apply_mask(alt_outputs, s["mask"], vcf_meta, s["output_resolution"], genes_df, exons_df)
            diffs = _apply_metric(ref_window, alt_window, s["metric"])
            all_diffs_per_scorer[scorer_idx].append(diffs)

        all_vcf.append(vcf_meta)

    all_vcf = np.concatenate(all_vcf, axis=0)
    results = [
        (torch.cat([t.detach().cpu() for t in dl], dim=0).numpy(), all_vcf)
        for dl in all_diffs_per_scorer
    ]

    model.cpu()
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Benchmark helpers  (each returns a list of ROC tuples, one per scorer)
# ---------------------------------------------------------------------------

def get_over_under_null(scores, vcf, df):
    df_pred = pd.DataFrame(vcf, columns=["CHROM", "POS", "NAME", "REF", "ALT"])
    df_pred["POS"]   = df_pred["POS"].astype(int)
    df_pred["score"] = scores

    def _roc(subset_df, label_col, score_col, negate=False):
        merged = subset_df.merge(
            df_pred, left_on=["chrom", "pos", "ref", "alt"],
            right_on=["CHROM", "POS", "REF", "ALT"], how="inner"
        ).drop_duplicates()
        labels = (merged[label_col] == subset_df[label_col].iloc[0])  # placeholder
        s = -merged[score_col] if negate else merged[score_col]
        return roc_auc_score(labels, s)

    df_ou = df[df["consequence"].isin(["over", "under"])].copy()
    merged_ou = df_ou.merge(df_pred, left_on=["chrom","pos","ref","alt"], right_on=["CHROM","POS","REF","ALT"], how="inner").drop_duplicates()
    roc_ou = roc_auc_score(merged_ou["consequence"] == "over", merged_ou["score"])

    df_un = df[df["consequence"].isin(["under", "none"])].copy()
    merged_un = df_un.merge(df_pred, left_on=["chrom","pos","ref","alt"], right_on=["CHROM","POS","REF","ALT"], how="inner").drop_duplicates()
    roc_un = roc_auc_score(merged_un["consequence"] == "under", -merged_un["score"])

    df_on = df[df["consequence"].isin(["over", "none"])].copy()
    merged_on = df_on.merge(df_pred, left_on=["chrom","pos","ref","alt"], right_on=["CHROM","POS","REF","ALT"], how="inner").drop_duplicates()
    roc_on = roc_auc_score(merged_on["consequence"] == "over", merged_on["score"])

    return roc_ou, roc_un, roc_on


def _run_promoter_benchmark(
    benchmark_name, vcf_name, tsv_name,
    model, rank, trained_version, model_name, scorers,
    input_len=None,
):
    """
    Run a single promoter benchmark for all scorers in one model pass.

    Returns:
        List of (roc_ou, roc_un, roc_on) tuples, one per scorer.
    """
    results = get_variants(
        model=model, vcf=vcf_name, rank=rank,
        benchmark_name=benchmark_name, trained_version=trained_version,
        model_name=model_name, scorers=scorers,
        input_len=input_len,
    )
    df = pd.read_csv(tsv_name, header=0, sep="\t")

    rocs_per_scorer = []
    for diffs, vcf in results:
        scores = diffs.mean(axis=1)
        rocs_per_scorer.append(get_over_under_null(scores, vcf, df))
    return rocs_per_scorer   # List[(ou, un, on)]


# One thin wrapper per benchmark dataset (no logic duplication)
_PROMOTER_BENCHMARKS = {
    "gtex_eqtls":    ("gtex_eqtls_near_promoter",     "../data/tableS1D_gtex_eqtls.vcf",         "../data/tableS1D_gtex_eqtls.tsv"),
    "mpra_eqtls":    ("mpra_eqtls_near_promoter",     "../data/tableS1E_mpra_eqtls.vcf",         "../data/tableS1E_mpra_eqtls.tsv"),
    "gtex_outliers": ("gtex_outliers_near_promoter",  "../data/tableS1A_gtex_outliers.vcf",      "../data/tableS1A_gtex_outliers.tsv"),
    "cagi5_sat":     ("cagi5_sat_near_promoter",      "../data/tableS1B_cagi5_saturation.vcf",   "../data/tableS1B_cagi5_saturation.tsv"),
    "mpra_sat":      ("mpra_sat_near_promoter",       "../data/tableS1C_mpra_saturation.vcf",    "../data/tableS1C_mpra_saturation.tsv"),
    "ukbb_proteome": ("ukbb_proteome_near_promoter",  "../data/tableS1F_ukbb_proteome.vcf",      "../data/tableS1F_ukbb_proteome.tsv"),
    "gel_rna":       ("gel_rna_near_promoter",         "../data/tableS1G_gel_rna.vcf",            "../data/tableS1G_gel_rna.tsv"),
}

def run_all_promoter_benchmarks(model, rank, trained_version, model_name, scorers, input_len=None):
    """
    Run every promoter benchmark, returning a dict:
        { benchmark_key: List[(roc_ou, roc_un, roc_on)] }   # outer list = per scorer
    """
    out = {}
    for key, (bname, vcf, tsv) in _PROMOTER_BENCHMARKS.items():
        out[key] = _run_promoter_benchmark(
            benchmark_name=bname, vcf_name=vcf, tsv_name=tsv,
            model=model, rank=rank, trained_version=trained_version,
            model_name=model_name, scorers=scorers,
            input_len=input_len,
        )
        print(f"Completed {bname} benchmark: {out[key]}")
    return out


# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

def initialize_models(k_l, k_c, trained_version, quant, model_name="borzoi", full=False):
    dev = "cpu" if quant else ("cuda" if torch.cuda.is_available() else "cpu")

    if not full:
        if model_name != "ag":
            model = MODEL_PARAMS[model_name]["model_class"](
                k_l=k_l, k_c=k_c, device=dev, **MODEL_PARAMS[model_name]["kwargs"]
            )
            state_dict = torch.load(f'{model_name}_lora_weights/{model_name}_lora_lr{k_l}_cr{k_c}.pth')
            new_state_dict = {}
            for key, value in state_dict.items():
                if ".lora." in key:
                    continue
                new_key = (key
                    .replace("model.", "").replace("orig_layer.", "")
                    .replace(".conv.weight", ".conv.layer.weight")
                    .replace(".conv.bias", ".conv.layer.bias")
                    .replace(".linear.weight", ".linear.layer.weight")
                    .replace(".linear.bias", ".linear.layer.bias")
                    .replace(".pointwise.weight", ".pointwise.layer.weight")
                    .replace(".pointwise.bias", ".pointwise.layer.bias")
                    .replace(".to_pos_k.weight", ".to_pos_k.layer.weight")
                    .replace(".to_v.weight", ".to_v.layer.weight")
                    .replace(".to_q.weight", ".to_q.layer.weight")
                    .replace(".to_k.weight", ".to_k.layer.weight")
                    .replace(".to_rel_k.weight", ".to_rel_k.layer.weight")
                    .replace(".to_out.weight", ".to_out.layer.weight")
                    .replace(".to_out.bias", ".to_out.layer.bias")
                    .replace(".0.0.weight", ".0.0.layer.weight")
                    .replace(".0.0.bias", ".0.0.layer.bias")
                )
                if ".channel_transform" in key:
                    new_key = (new_key
                        .replace(".conv.layer.weight", ".conv.layer.layer.weight")
                        .replace(".conv.layer.bias", ".conv.layer.layer.bias")
                        .replace(".linear.layer.weight", ".linear.layer.layer.weight")
                        .replace(".linear.layer.bias", ".linear.layer.layer.bias")
                    )
                new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict, strict=True)

        else:  # ag LoRA
            model = MODEL_PARAMS[model_name]["model_class"](k=k_l, **MODEL_PARAMS[model_name]["kwargs"])
            state_dict = torch.load(f'{model_name}_lora_weights/{model_name}_lora_lr{k_l}.pth', weights_only=True)
            new_state_dict = {}
            for key, value in state_dict.items():
                if ".lora." in key:
                    continue
                new_key = (key
                    .replace("proj.weight", "proj.layer.weight")
                    .replace("linear_pair.weight", "linear_pair.layer.weight")
                    .replace("linear_pair.bias", "linear_pair.layer.bias")
                    .replace("linear_pos_features.weight", "linear_pos_features.layer.weight")
                    .replace("linear_pos_features.bias", "linear_pos_features.layer.bias")
                    .replace("linear_y_q.weight", "linear_y_q.layer.weight")
                    .replace("linear_y_k.weight", "linear_y_k.layer.weight")
                    .replace("linear_q.weight", "linear_q.layer.weight")
                    .replace("linear_k.weight", "linear_k.layer.weight")
                    .replace("linear_v.weight", "linear_v.layer.weight")
                    .replace("linear_v.bias", "linear_v.layer.bias")
                    .replace("linear1.weight", "linear1.layer.weight")
                    .replace("linear1.bias", "linear1.layer.bias")
                    .replace("linear2.weight", "linear2.layer.weight")
                    .replace("linear2.bias", "linear2.layer.bias")
                )
                new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict, strict=True)

    else:
        if model_name == "borzoi":
            model = torch.load("./full_models/borzoi_human_rep0.pt")
        elif model_name == "enformer":
            model = torch.load("./full_models/enformer_human.pt")
        elif model_name == "ag":
            model = AlphaGenomePT.from_pretrained("model_all_folds.safetensors")
            print("Loaded full AG model")
        else:
            model = None

    return model


# ---------------------------------------------------------------------------
# save_output  – one row per scorer, one model load per call
# ---------------------------------------------------------------------------

def save_output(
    rank_l=256, rank_c=256, trained_version=None, quant=False,
    model_name="borzoi", full=False, input_len=None,
    pai_mask=None, pai_metric=None, pai_track=None, pai_res=None,
):
    q = "quant" if quant else "no_quant"
    model_name_rank = f"{model_name}_lora_lr{rank_l}_cr{rank_c}_{trained_version}_{q}"

    # Normalise scorer config
    if pai_mask is None:
        pai_mask = MODEL_PARAMS[model_name]["pai_mask"]
    if pai_metric is None:
        pai_metric = MODEL_PARAMS[model_name]["pai_metric"]

    scorers = _normalize_scorers(pai_mask, pai_metric, pai_track, pai_res)

    resolved_input_len = input_len or MODEL_PARAMS[model_name]["input_len"]

    model = initialize_models(
        k_l=rank_l, k_c=rank_c, trained_version=trained_version,
        quant=quant, model_name=model_name, full=full,
    )

    # Run all benchmarks (each VCF processed once, all scorers in parallel)
    benchmark_results = run_all_promoter_benchmarks(
        model=model, rank=rank_l, trained_version=trained_version,
        model_name=model_name, scorers=scorers,
        input_len=resolved_input_len,
    )

    print("GTEX outliers:", [
        (round(r[0], 4), round(r[1], 4), round(r[2], 4))
        for r in benchmark_results["gtex_outliers"]
    ])

    # Write one TSV row per scorer
    pai_path = f"{model_name}_pai.tsv"
    file_exists = os.path.isfile(pai_path)

    with open(pai_path, "a", newline="") as f:
        for i, s in enumerate(scorers):
            row = {
                "model":   model_name_rank,
                "input_len": resolved_input_len,
                "mask":    s["mask"],
                "metric":  s["metric"],
                "track":   s["output_head"],
                "resolution": s["output_resolution"],
            }
            # Unpack each benchmark's (ou, un, on) for this scorer
            for bkey, col_suffixes in [
                ("gtex_eqtls",    ("GTEX_eqtl_OvU",       "GTEX_eqtl_UvN",       "GTEX_eqtl_OvN")),
                ("gtex_outliers", ("GTEX_outliers_OvU",    "GTEX_outliers_UvN",    "GTEX_outliers_OvN")),
                ("cagi5_sat",     ("CAGI5_sat_OvU",        "CAGI5_sat_UvN",        "CAGI5_sat_OvN")),
                ("mpra_sat",      ("MPRA_sat_OvU",         "MPRA_sat_UvN",         "MPRA_sat_OvN")),
                ("mpra_eqtls",    ("MPRA_eqtl_OvU",        "MPRA_eqtl_UvN",        "MPRA_eqtl_OvN")),
                ("ukbb_proteome", ("UKBB_proteome_OvU",    "UKBB_proteome_UvN",    "UKBB_proteome_OvN")),
                ("gel_rna",       ("Gel_RNA_OvU",          "Gel_RNA_UvN",          "Gel_RNA_OvN")),
            ]:
                ou, un, on = benchmark_results[bkey][i]
                row[f"{col_suffixes[0]}_promoter"] = round(ou, 4)
                row[f"{col_suffixes[1]}_promoter"] = round(un, 4)
                row[f"{col_suffixes[2]}_promoter"] = round(on, 4)

            writer = csv.DictWriter(f, fieldnames=row.keys(), delimiter="\t")
            if not file_exists and i == 0:
                writer.writeheader()
            writer.writerow(row)

    print(f"Results saved to {pai_path}")
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    torch.cuda.set_device(1)

    # save_output(
    #     rank_l="full", rank_c="full", quant=False, model_name="ag", full=True,
    #     input_len=1048576,
    #     pai_metric=("log2_diff_ag", "log2_diff_ag", "log2_diff_ag"),
    #     pai_mask=("center",         "center",        "exon"),
    #     pai_track=("dnase",         "cage",          "rna_seq"),
    #     pai_res=128,
    # )

    # save_output(
    #     rank_l="full", rank_c="full", quant=False, model_name="ag", full=True,
    #     input_len=4096,
    #     pai_metric=("log2_diff_ag_dnase", "log2_diff_ag_dnase", "log2_diff_ag_rna"),
    #     pai_mask=("center",         "center",        "exon"),
    #     pai_track=("dnase",         "cage",          "rna_seq"),
    #     pai_res=128,
    # )
    # save_output(
    #     rank_l=1, rank_c="full", quant=False, model_name="ag", full=False,
    #     input_len=4096,
    #     pai_metric=("log2_diff_ag_dnase", "log2_diff_ag_dnase", "log2_diff_ag_rna"),
    #     pai_mask=("center",         "center",        "exon"),
    #     pai_track=("dnase",         "cage",          "rna_seq"),
    #     pai_res=128,
    # )
    # save_output(
    #     rank_l=2, rank_c="full", quant=False, model_name="ag", full=False,
    #     input_len=4096,
    #     pai_metric=("log2_diff_ag_dnase", "log2_diff_ag_dnase", "log2_diff_ag_rna"),
    #     pai_mask=("center",         "center",        "exon"),
    #     pai_track=("dnase",         "cage",          "rna_seq"),
    #     pai_res=128,
    # )
    # save_output(
    #     rank_l=4, rank_c="full", quant=False, model_name="ag", full=False,
    #     input_len=4096,
    #     pai_metric=("log2_diff_ag_dnase", "log2_diff_ag_dnase", "log2_diff_ag_rna"),
    #     pai_mask=("center",         "center",        "exon"),
    #     pai_track=("dnase",         "cage",          "rna_seq"),
    #     pai_res=128,
    # )
    # save_output(
    #     rank_l=8, rank_c="full", quant=False, model_name="ag", full=False,
    #     input_len=4096,
    #     pai_metric=("log2_diff_ag_dnase", "log2_diff_ag_dnase", "log2_diff_ag_rna"),
    #     pai_mask=("center",         "center",        "exon"),
    #     pai_track=("dnase",         "cage",          "rna_seq"),
    #     pai_res=128,
    # )
    # save_output(
    #     rank_l=16, rank_c="full", quant=False, model_name="ag", full=False,
    #     input_len=4096,
    #     pai_metric=("log2_diff_ag_dnase", "log2_diff_ag_dnase", "log2_diff_ag_rna"),
    #     pai_mask=("center",         "center",        "exon"),
    #     pai_track=("dnase",         "cage",          "rna_seq"),
    #     pai_res=128,
    # )
    # save_output(
    #     rank_l=32, rank_c="full", quant=False, model_name="ag", full=False,
    #     input_len=4096,
    #     pai_metric=("log2_diff_ag_dnase", "log2_diff_ag_dnase", "log2_diff_ag_rna"),
    #     pai_mask=("center",         "center",        "exon"),
    #     pai_track=("dnase",         "cage",          "rna_seq"),
    #     pai_res=128,
    # )
    # save_output(
    #     rank_l=64, rank_c="full", quant=False, model_name="ag", full=False,
    #     input_len=4096,
    #     pai_metric=("log2_diff_ag_dnase", "log2_diff_ag_dnase", "log2_diff_ag_rna"),
    #     pai_mask=("center",         "center",        "exon"),
    #     pai_track=("dnase",         "cage",          "rna_seq"),
    #     pai_res=128,
    # )
    # save_output(
    #     rank_l=128, rank_c="full", quant=False, model_name="ag", full=False,
    #     input_len=4096,
    #     pai_metric=("log2_diff_ag_dnase", "log2_diff_ag_dnase", "log2_diff_ag_rna"),
    #     pai_mask=("center",         "center",        "exon"),
    #     pai_track=("dnase",         "cage",          "rna_seq"),
    #     pai_res=128,
    # )
    # save_output(
    #     rank_l=256, rank_c="full", quant=False, model_name="ag", full=False,
    #     input_len=4096,
    #     pai_metric=("log2_diff_ag_dnase", "log2_diff_ag_dnase", "log2_diff_ag_rna"),
    #     pai_mask=("center",         "center",        "exon"),
    #     pai_track=("dnase",         "cage",          "rna_seq"),
    #     pai_res=128,
    # )
    # save_output(
    #     rank_l=512, rank_c="full", quant=False, model_name="ag", full=False,
    #     input_len=4096,
    #     pai_metric=("log2_diff_ag_dnase", "log2_diff_ag_dnase", "log2_diff_ag_rna"),
    #     pai_mask=("center",         "center",        "exon"),
    #     pai_track=("dnase",         "cage",          "rna_seq"),
    #     pai_res=128,
    # )


    save_output(
        rank_l="full", rank_c="full", quant=False, model_name="ag", full=True,
        input_len=1048576,
        pai_metric=("log2_diff_ag_dnase", "log2_diff_ag_dnase", "log2_diff_ag_rna"),
        pai_mask=("center",         "center",        "exon"),
        pai_track=("dnase",         "cage",          "rna_seq"),
        pai_res=128,
    )
    # save_output(
    #     rank_l=1, rank_c="full", quant=False, model_name="ag", full=False,
    #     input_len=1048576,
    #     pai_metric=("log2_diff_ag_dnase", "log2_diff_ag_dnase", "log2_diff_ag_rna"),
    #     pai_mask=("center",         "center",        "exon"),
    #     pai_track=("dnase",         "cage",          "rna_seq"),
    #     pai_res=128,
    # )
    # save_output(
    #     rank_l=2, rank_c="full", quant=False, model_name="ag", full=False,
    #     input_len=1048576,
    #     pai_metric=("log2_diff_ag_dnase", "log2_diff_ag_dnase", "log2_diff_ag_rna"),
    #     pai_mask=("center",         "center",        "exon"),
    #     pai_track=("dnase",         "cage",          "rna_seq"),
    #     pai_res=128,
    # )
    # save_output(
    #     rank_l=4, rank_c="full", quant=False, model_name="ag", full=False,
    #     input_len=1048576,
    #     pai_metric=("log2_diff_ag_dnase", "log2_diff_ag_dnase", "log2_diff_ag_rna"),
    #     pai_mask=("center",         "center",        "exon"),
    #     pai_track=("dnase",         "cage",          "rna_seq"),
    #     pai_res=128,
    # )
    # save_output(
    #     rank_l=8, rank_c="full", quant=False, model_name="ag", full=False,
    #     input_len=1048576,
    #     pai_metric=("log2_diff_ag_dnase", "log2_diff_ag_dnase", "log2_diff_ag_rna"),
    #     pai_mask=("center",         "center",        "exon"),
    #     pai_track=("dnase",         "cage",          "rna_seq"),
    #     pai_res=128,
    # )
    # save_output(
    #     rank_l=16, rank_c="full", quant=False, model_name="ag", full=False,
    #     input_len=1048576,
    #     pai_metric=("log2_diff_ag_dnase", "log2_diff_ag_dnase", "log2_diff_ag_rna"),
    #     pai_mask=("center",         "center",        "exon"),
    #     pai_track=("dnase",         "cage",          "rna_seq"),
    #     pai_res=128,
    # )
    # save_output(
    #     rank_l=32, rank_c="full", quant=False, model_name="ag", full=False,
    #     input_len=1048576,
    #     pai_metric=("log2_diff_ag_dnase", "log2_diff_ag_dnase", "log2_diff_ag_rna"),
    #     pai_mask=("center",         "center",        "exon"),
    #     pai_track=("dnase",         "cage",          "rna_seq"),
    #     pai_res=128,
    # )
    # save_output(
    #     rank_l=64, rank_c="full", quant=False, model_name="ag", full=False,
    #     input_len=1048576,
    #     pai_metric=("log2_diff_ag_dnase", "log2_diff_ag_dnase", "log2_diff_ag_rna"),
    #     pai_mask=("center",         "center",        "exon"),
    #     pai_track=("dnase",         "cage",          "rna_seq"),
    #     pai_res=128,
    # )
    # save_output(
    #     rank_l=128, rank_c="full", quant=False, model_name="ag", full=False,
    #     input_len=1048576,
    #     pai_metric=("log2_diff_ag_dnase", "log2_diff_ag_dnase", "log2_diff_ag_rna"),
    #     pai_mask=("center",         "center",        "exon"),
    #     pai_track=("dnase",         "cage",          "rna_seq"),
    #     pai_res=128,
    # )
    # save_output(
    #     rank_l=256, rank_c="full", quant=False, model_name="ag", full=False,
    #     input_len=1048576,
    #     pai_metric=("log2_diff_ag_dnase", "log2_diff_ag_dnase", "log2_diff_ag_rna"),
    #     pai_mask=("center",         "center",        "exon"),
    #     pai_track=("dnase",         "cage",          "rna_seq"),
    #     pai_res=128,
    # )
    # save_output(
    #     rank_l=512, rank_c="full", quant=False, model_name="ag", full=False,
    #     input_len=1048576,
    #     pai_metric=("log2_diff_ag_dnase", "log2_diff_ag_dnase", "log2_diff_ag_rna"),
    #     pai_mask=("center",         "center",        "exon"),
    #     pai_track=("dnase",         "cage",          "rna_seq"),
    #     pai_res=128,
    # )

if __name__ == "__main__":
    main()