#!/usr/bin/env python3
"""
Per-track LoRA-vs-full correlation analysis for Borzoi / Enformer / Sei / AlphaGenome 
(per-sequence Pearson +across-sequence Spearman on model outputs written to disk via an async
pickle writer), with one difference: AlphaGenome's output dict is already
segmented by assay (atac/dnase/cage/...), so `extract_arrays` could just
key on `mode`. Borzoi/Enformer/Sei emit one flat (n_tracks, n_bins) tensor
with no such segmentation, so here the "key" during extraction is just the
track index, and assay grouping happens as a separate step afterward using
each model's real track/target metadata:

  * Borzoi / Enformer: gReLU's `tasks` dataframe (model.data_params["tasks"]),
    which carries `assay` / `sample` / `description` per track — this is how
    gReLU's own tutorials identify tracks (`tasks[tasks.assay == "CAGE"]`).
  * Sei: `pd.DataFrame(model.head.target_annot)` -- gives per-target
    annotations straight off the loaded head module. The raw assay/feature
    field is free text (verified against the uploaded target.names, 21,907
    tracks: ~9.3k TF/other ChIP -- including Pol2/POLR2A -- ~10k histone
    marks, ~1.6k DNase, ~964 ATAC-seq; 
    (`classify_sei_assay`) rather than looked up in a table. Falls back to
    parsing target.names directly (SEI_TARGET_NAMES_PATH) if target_annot
    isn't usable.

NOTE: Sei's output has no bins/length axis (shape (n_tracks,) -- already
pooled to one scalar per track per sequence), unlike Borzoi/Enformer's
(n_tracks, n_bins). `compute_correlations` detects this per-model and
returns `per_seq_corr=None` when there's nothing to correlate "within" a
sequence -- the caller below must handle that case rather than assuming
per_seq is always defined.

Three correlation metrics are computed per track:
  * per_seq       -- Pearson within each sequence (across bins), averaged
                      over sequences. Undefined (None) when there's no
                      bins axis (Sei).
  * across_seq    -- Spearman across sequences, on either per-sequence
                      bin-means (Borzoi/Enformer) or the raw pooled scalar
                      (Sei). Always defined.
  * pooled        -- Pearson across all (sequence, bin) pairs at once, no
                      per-sequence centering. By the standard ANOVA
                      variance-decomposition identity, this is a
                      variance-weighted blend of the within-sequence
                      (per_seq-like) and between-sequence (across_seq-like)
                      signal -- not an arbitrary average of the other two,
                      but a single number the data itself weights. For Sei
                      (no bins axis) this collapses to exactly across_seq.
"""

import os

# Must be set before `import torch` -- this configures PyTorch's CUDA
# allocator to avoid fragmentation-driven OOMs. Present in the original
# AG-only script; restored here since this script now runs four different
# model shapes (Borzoi/Enformer/Sei/AG) sequentially in one long-lived
# process, which is exactly the situation where allocator fragmentation
# from switching between wildly different model sizes shows up.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import sys
import gc
import pickle
import tempfile
import threading
import queue
from typing import Dict, Optional

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from tqdm import tqdm

from borzoi_lora_arch_mha import BorzoiModel, EnformerModel
from ag_arch import AlphaGenome
from alphagenome_pytorch import AlphaGenome as AlphaGenomePT
import seimodel as sm
import seillra as sl

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from sei_lora.dataloaders import SeqDataLoader, SeqDataset

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
RANKS      = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
N_SAMPLES  = 200
SEED       = 42
BED        = "./GRCh38_cCREs_4kb.bed"
FASTA      = "../resources/hg38_UCSC.fa"
DEVICE     = torch.device("cuda:1")

# Fallback source for Sei track metadata if `model.head.target_annot` isn't
# available/usable at runtime. Same content as target_annot is built from:
# one line per track, "<cell/tissue> | <assay or factor> | <ID or source>".
SEI_TARGET_NAMES_PATH = "./target.names"

MODEL_PARAMS = {
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
    "ag": {
        "input_len":  1048576,
        "bin_length": 1,
        "model_class": AlphaGenome,
        "kwargs": {},
    },
}

# Map raw assay strings (from gReLU's `tasks.assay`, or from Sei's
# `target_annot`) onto the coarse groups you want to average over. Anything
# not in this map falls back to its own raw label as its own group, so
# nothing silently gets dropped -- extend as you inspect real values.
ASSAY_GROUP_MAP = {
    "DNASE": "DNase", "DNase": "DNase", "dnase": "DNase",
    "ATAC":  "ATAC",  "atac": "ATAC",
    "CAGE":  "CAGE",  "cage": "CAGE",
    "RNA":   "RNA-seq", "RNA-SEQ": "RNA-seq", "rna_seq": "RNA-seq",
    "CHIP_TF": "ChIP-TF",
    "CHIP_HISTONE": "ChIP-Histone",
    "PROCAP": "PRO-cap",
}

# ----------------------------------------------------------------
# AlphaGenome-specific: its output is a dict segmented by assay/resolution
# (e.g. "atac" at res 1 and res 128), unlike Borzoi/Enformer/Sei's single
# tensor. Each key's own track dimension is preserved (see
# _ag_tensor_to_2d), so real track-to-track variance survives within a
# key. Each (mode, resolution/subkey) combination is treated as its own
# assay group -- e.g. atac_res1 and atac_res128 are NOT pooled together --
# via get_ag_key_groups below; every track inside one key shares that
# key's single group label.
# ----------------------------------------------------------------
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

AG_MODE_GROUP = {
    "atac": "ATAC",
    "dnase": "DNase",
    "procap": "PRO-cap",
    "cage": "CAGE",
    "rna_seq": "RNA-seq",
    "chip_tf": "ChIP-TF",
    "chip_histone": "ChIP-Histone",
    "contact_maps": "Contact-maps",
    "splice_sites": "Splice-sites",
    "splice_site_usage": "Splice-site-usage",
}


# ----------------------------------------------------------------
# Track metadata (assay group per track index)
# ----------------------------------------------------------------
def _map_assay_column(series: pd.Series) -> pd.Series:
    return series.map(ASSAY_GROUP_MAP).fillna(series)


def load_grelu_tasks(full_model: nn.Module) -> pd.DataFrame:
    """
    Reads gReLU's per-track `tasks` metadata straight off the loaded
    checkpoint. Your full_models/*.pt files are pickled
    grelu.lightning.LightningModel objects; data_params["tasks"] is a dict
    of per-track lists (e.g. `assay`, `sample`, `description`), not already
    a DataFrame -- same shape of thing as target_annot for Sei.
    """
    tasks = pd.DataFrame(full_model.data_params["tasks"]).reset_index(drop=True)
    tasks["track_index"] = tasks.index
    return tasks


def _find_column(df: pd.DataFrame, candidates, purpose: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Couldn't find a {purpose} column; got columns {list(df.columns)}. "
        f"Tried: {candidates}. Update the candidate list to match the real key."
    )


def refine_chip_groups(tasks: pd.DataFrame) -> pd.DataFrame:
    """Splits generic 'CHIP' into TF vs Histone using H3K/H4K naming in description, if available."""
    try:
        desc_col = _find_column(tasks, ("description", "descriptions"), "description")
    except KeyError:
        return tasks  # no description text to refine on -- leave assay labels as-is
    assay_col = _find_column(tasks, ("assay", "assays"), "assay")
    is_histone = tasks[desc_col].str.contains(
        r"H[234]K\d+", case=False, regex=True, na=False
    )
    tasks.loc[is_histone, assay_col] = "CHIP_HISTONE"
    tasks.loc[(tasks[assay_col] == "CHIP") & ~is_histone, assay_col] = "CHIP_TF"
    return tasks


# ----------------------------------------------------------------
# Sei-specific: target.names / target_annot has no clean categorical assay
# column -- the middle '|'-delimited field is free text like "DNase",
# "DNase.fdr0.01.hot", "H3K27ac", "H3K9ac, H3K14ac", "CTCF", "POLR2A", so we
# classify it with a few pattern rules instead of a lookup table.
# ----------------------------------------------------------------
_HISTONE_TOKEN_RE = re.compile(r'^H[1-4][A-Za-z0-9\.\/]*$')


def classify_sei_assay(raw: str) -> str:
    raw = "" if raw is None else str(raw).strip()
    if not raw:
        return "Unknown"
    low = raw.lower()
    if "dnase" in low:
        return "DNase"
    if "atac" in low:
        return "ATAC"
    if re.search(r'\bcage\b', low):
        return "CAGE"
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if tokens and all(_HISTONE_TOKEN_RE.match(t) for t in tokens):
        return "ChIP-Histone"
    return "ChIP-TF"


def parse_target_names(path: str) -> pd.DataFrame:
    """
    Parses a target.names file: one track per line, formatted as
    "<cell/tissue> | <assay or factor> | <ID or source>". Line order is
    assumed to match track/output index order (0-indexed).
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split("|")]
            while len(parts) < 3:
                parts.append(None)
            rows.append(parts[:3])
    return pd.DataFrame(rows, columns=["cell_type", "feature", "id_or_source"])


def _find_sei_assay_column(tgts: pd.DataFrame):
    """
    Locates the raw assay/feature string column on whatever target_annot
    turns out to look like. `model.head.target_annot` is a dict with an
    "assays" key (21k strings, one per track -- the target.names middle
    column), so pd.DataFrame(target_annot) surfaces that as a column
    literally named "assays". Also checks a few other likely names, then
    falls back to a positional middle field if target_annot was unnamed.
    """
    for c in ("assays", "assay", "feature", "target", "mark"):
        if c in tgts.columns:
            return c
    if 1 in tgts.columns:
        return 1
    raise KeyError(
        f"Couldn't identify the assay/feature column in target_annot; got columns "
        f"{list(tgts.columns)}. Expected an 'assays' column (as in target_annot's "
        "dict keys), or unnamed/positional columns where column 1 is target.names' "
        "middle field."
    )


def get_sei_track_groups(full_model: Optional[nn.Module] = None) -> pd.Series:
    """
    Returns a pandas Series indexed by track_index -> assay_group, for Sei.
    Prefers `model.head.target_annot` (matches your usage), falls back to
    parsing target.names directly if that's unavailable.
    """
    raw_assay = None

    if full_model is not None and hasattr(full_model.head, "target_annot"):
        tgts = pd.DataFrame(full_model.head.target_annot)
        assay_col = _find_sei_assay_column(tgts)
        raw_assay = tgts[assay_col]

    if raw_assay is None:
        if not os.path.exists(SEI_TARGET_NAMES_PATH):
            raise FileNotFoundError(
                "Sei track metadata unavailable: `model.head.target_annot` wasn't "
                f"usable, and no target.names file found at {SEI_TARGET_NAMES_PATH}."
            )
        tgts = parse_target_names(SEI_TARGET_NAMES_PATH)
        raw_assay = tgts["feature"]

    groups = raw_assay.reset_index(drop=True).map(classify_sei_assay)
    groups.index = np.arange(len(groups))
    return groups


def get_track_groups(model_name: str, full_model: Optional[nn.Module] = None) -> pd.Series:
    """
    Returns a pandas Series indexed by track_index -> assay_group (str).
    Requires the loaded full model for all three model types now: Borzoi/
    Enformer checkpoints are gReLU LightningModels carrying data_params
    directly, and Sei's metadata lives on model.head.target_annot.
    """
    if model_name in ("borzoi", "enformer"):
        if full_model is None:
            raise ValueError(f"{model_name} track metadata requires the loaded full model (data_params['tasks']).")
        tasks = load_grelu_tasks(full_model)
        tasks = refine_chip_groups(tasks)
        assay_col = _find_column(tasks, ("assay", "assays"), "assay")
        groups = _map_assay_column(tasks[assay_col])
        groups.index = tasks["track_index"].values
        return groups

    elif model_name == "sei":
        return get_sei_track_groups(full_model=full_model)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")


# ----------------------------------------------------------------
# Model init 
# ----------------------------------------------------------------
def initialize_model(k_l: int, k_c: int, quant: bool, model_name="borzoi", full=False):

    if quant == True:
        dev = "cpu"
    else:
        dev = 'cuda:1'

    if full != True:
        if model_name not in MODEL_PARAMS:
            model = sl.Sei_LLRA(k=k_l, projection=False, mode="sequence", quant=None)

        else:
            model = MODEL_PARAMS[model_name]["model_class"](k_l=k_l, k_c=k_c, device=dev, **MODEL_PARAMS[model_name]["kwargs"])
            # model = BorzoiModel(k_l =k_l, k_c = k_c, n_tasks = 7611, crop_len=5120, final_act_func="softplus", final_pool_func=None)
            state_dict = torch.load(f'{model_name}_lora_weights/{model_name}_lora_lr{k_l}_cr{k_c}.pth', weights_only=True)
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key
                if ".lora." in key:
                    continue

                else:
                    new_key = new_key.replace("model.", "")
                    new_key = new_key.replace("orig_layer.", "")
                    if ".channel_transform" in key:
                        new_key = new_key.replace(".conv.layer.weight", ".conv.layer.layer.weight")
                        new_key = new_key.replace(".conv.layer.bias", ".conv.layer.layer.bias")
                        new_key = new_key.replace(".linear.layer.weight", ".linear.layer.layer.weight")
                        new_key = new_key.replace(".linear.layer.bias", ".linear.layer.layer.bias")

                    new_key = new_key.replace(".conv.weight", ".conv.layer.weight")
                    new_key = new_key.replace(".conv.bias", ".conv.layer.bias")
                    new_key = new_key.replace(".linear.weight", ".linear.layer.weight")
                    new_key = new_key.replace(".linear.bias", ".linear.layer.bias")
                    new_key = new_key.replace(".pointwise.weight", ".pointwise.layer.weight")
                    new_key = new_key.replace(".pointwise.bias", ".pointwise.layer.bias")

                    new_key = new_key.replace(".to_pos_k.weight", ".to_pos_k.layer.weight")
                    new_key = new_key.replace(".to_v.weight", ".to_v.layer.weight")
                    new_key = new_key.replace(".to_q.weight", ".to_q.layer.weight")
                    new_key = new_key.replace(".to_k.weight", ".to_k.layer.weight")
                    new_key = new_key.replace(".to_rel_k.weight", ".to_rel_k.layer.weight")
                    new_key = new_key.replace(".to_out.weight", ".to_out.layer.weight")
                    new_key = new_key.replace(".to_out.bias", ".to_out.layer.bias")
                    new_key = new_key.replace(".0.0.weight", ".0.0.layer.weight")
                    new_key = new_key.replace(".0.0.bias", ".0.0.layer.bias")

                new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict, strict=True)
    else:
        if model_name == "borzoi":
            model = torch.load("./full_models/borzoi_human_rep0.pt", weights_only=False)
        elif model_name == "enformer":
            model = torch.load("./full_models/enformer_human.pt", weights_only=False)
        elif model_name == "sei":
            model = sl.Sei_LLRA(k=None, projection=False, mode="sequence", quant=None)

        else:
            model = None

    return model


def initialize_model_ag(rank: Optional[int] = None, full: bool = False):
    """
    AlphaGenome's checkpoint/state-dict layout is different enough from
    Borzoi/Enformer (different LoRA key names, safetensors full checkpoint)
    that it gets its own init function rather than overloading
    `initialize_model`.
    """
    if full:
        model = AlphaGenomePT.from_pretrained("model_all_folds.safetensors")
    else:
        model = MODEL_PARAMS["ag"]["model_class"](k=rank, **MODEL_PARAMS["ag"]["kwargs"])
        state_dict = torch.load(f'ag_lora_weights/ag_lora_lr{rank}.pth', weights_only=True)
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

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


# ----------------------------------------------------------------
# GPU / memory helpers
# ----------------------------------------------------------------
def free_gpu():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def print_gpu_mem(label=""):
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[GPU] {label:30s}  alloc={alloc:.2f}GB  reserved={reserved:.2f}GB")


class AsyncPickleWriter:
    """Background-thread pickle writer so the GPU never waits on disk I/O."""
    def __init__(self, path):
        self.path = path
        self._q = queue.Queue(maxsize=8)
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def _worker(self):
        with open(self.path, "ab") as f:
            while True:
                item = self._q.get()
                if item is None:
                    break
                pickle.dump(item, f)
                f.flush()
                self._q.task_done()

    def write(self, obj):
        self._q.put(obj)

    def close(self):
        self._q.put(None)
        self._t.join()


def iter_pickle_file(path):
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def make_loader(dataset):
    g = torch.Generator()
    g.manual_seed(SEED)
    return SeqDataLoader(
        dataset=dataset, batch_size=1, shuffle=True, num_workers=4,
        pin_memory=True, n_samples=N_SAMPLES, generator=g,
    )


def extract_array(out) -> np.ndarray:
    """
    Flatten one sequence's model output into a float16 numpy array. Shape
    depends on the model/config: (n_tracks, n_bins) if a length axis
    survives pooling, or (n_tracks,) if the model already pools to one
    scalar per track (this is the case for Sei). Unlike AlphaGenome's
    `extract_arrays`, there's no dict of assay-segmented keys to iterate --
    Borzoi/Enformer/Sei just emit one tensor, so we keep it whole and defer
    segmentation to the grouping step.
    """
    return out[0].to(torch.float16).cpu().numpy()


def run_inference_to_file(model, dataset, desc, tmp_path):
    """GPU inference with async background pickle writes (one record per sequence)."""
    writer = AsyncPickleWriter(tmp_path)
    loader = make_loader(dataset)
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            data, _ = batch
            data = data.to(DEVICE, non_blocking=True)
            out = model(data)                      # (1, n_tracks[, n_bins])
            arr = extract_array(out)
            del out, data
            writer.write(arr)
    writer.close()
    free_gpu()


def _ag_tensor_to_2d(t: torch.Tensor) -> np.ndarray:
    """
    Converts one sequence's AG output tensor into (n_tracks, n_bins).

    AG's layout is (batch=1, length, n_tracks) -- CHANNELS-LAST, unlike
    Borzoi/Enformer's PyTorch-native channels-first (batch, n_tracks,
    length). Confirmed empirically: a "_res128" key showed n_tracks=8192,
    which is exactly 1,048,576 / 128 -- i.e. the bin count at that
    resolution, not a track count. 

    """
    arr = t.cpu().to(torch.float16).numpy()
    arr = arr[0]                                  # drop batch dim (always 1 here)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr[None, :]                        # (n,) -> treat as (1 track, n bins)
    else:
        arr = arr.reshape(-1, arr.shape[-1])       # flatten everything but the last axis
        arr = arr.T                                # -> (n_tracks, bins_flattened)
    return arr


def extract_ag_arrays(out) -> Dict[str, np.ndarray]:
    """
    AlphaGenome's output is a dict keyed by assay/resolution (e.g.
    out["atac"][1], out["atac"][128], out["splice_sites"]["logits"], ...).
    Each key's tensor is kept as (n_tracks, n_bins) -- see
    _ag_tensor_to_2d -- instead of being fully flattened, so downstream
    correlation code can treat each key the same way Borzoi/Enformer/Sei's
    (n_tracks, n_bins) output is already treated, with real per-track
    values to compute variance from.
    """
    arrays = {}
    for mode, cfg in MODE_KEYS.items():
        if mode not in out:
            continue
        if cfg["type"] == "multi_res":
            for res in cfg["resolutions"]:
                if res in out[mode]:
                    arrays[f"{mode}_res{res}"] = _ag_tensor_to_2d(out[mode][res])
        elif cfg["type"] == "raw":
            arrays[mode] = _ag_tensor_to_2d(out[mode])
        elif cfg["type"] == "subkey":
            for sk in cfg["subkeys"]:
                if sk in out[mode]:
                    arrays[f"{mode}_{sk}"] = _ag_tensor_to_2d(out[mode][sk])
    return arrays


def run_inference_to_file_ag(model, dataset, desc, tmp_path):
    """Same async-writer pattern as run_inference_to_file, but for AG's dict-shaped output."""
    organism_index = torch.tensor([0], dtype=torch.long, device=DEVICE)
    writer = AsyncPickleWriter(tmp_path)
    loader = make_loader(dataset)
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            data, _ = batch
            data = data.to(DEVICE, non_blocking=True).permute(0, 2, 1)
            out = model(data, organism_index, resolutions=[128])
            arrays = extract_ag_arrays(out)
            del out, data
            writer.write(arrays)
    writer.close()
    free_gpu()


def get_ag_key_groups(output_keys) -> Dict[str, str]:
    """
    Maps each AG output key (e.g. "atac_res1", "atac_res128",
    "splice_sites_logits") to an assay-group display label. Resolution/
    subkey is kept as part of the group -- e.g. atac_res1 and atac_res128
    are treated as separate assays rather than pooled together -- only the
    base mode name gets a nicer display label via AG_MODE_GROUP.

    This is now a per-KEY mapping, not a per-track one: since
    _ag_tensor_to_2d preserves each key's real track dimension, every
    track within one key shares that key's single group label (all the
    individual ATAC-seq tracks inside "atac_res1" belong to the "ATAC_res1"
    group together).
    """
    suffixes = ("_res1", "_res128", "_logits", "_probs", "_predictions")

    def display_group(key: str) -> str:
        for suf in suffixes:
            if key.endswith(suf):
                base = key[: -len(suf)]
                return f"{AG_MODE_GROUP.get(base, base)}{suf}"
        return AG_MODE_GROUP.get(key, key)

    return {k: display_group(k) for k in output_keys}


# ----------------------------------------------------------------
# Correlation computation (per-track version of the AlphaGenome per_seq /
# across_seq pattern)
# ----------------------------------------------------------------
def _row_pearson(fa: np.ndarray, la: np.ndarray) -> np.ndarray:
    """Vectorized Pearson correlation along axis=1, one value per row (track)."""
    fa = fa - fa.mean(axis=1, keepdims=True)
    la = la - la.mean(axis=1, keepdims=True)
    num = (fa * la).sum(axis=1)
    den = np.sqrt((fa ** 2).sum(axis=1) * (la ** 2).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _col_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Pearson correlation along axis=0, one value per column (track)."""
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    num = (x * y).sum(axis=0)
    den = np.sqrt((x ** 2).sum(axis=0) * (y ** 2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def compute_correlations(full_tmp: str, lora_tmp: str, n_tracks: int):
    """
    Same structure as the AlphaGenome script's compute_correlations:
      - per_seq: for each track, Pearson correlation between full and lora
        computed within each sequence (across bins), then averaged over
        sequences. Only defined if the model output actually has a
        length/bin axis -- Sei's output is already pooled to one scalar
        per track per sequence, in which case there's nothing to correlate
        "within" a sequence and this is skipped (returns None).
      - across_seq: for each track, Spearman correlation between full and
        lora sequence-level values, computed across sequences. Always
        defined -- if there's no bins axis, "sequence-level value" is just
        the single scalar itself rather than a per-sequence mean.
    """
    per_seq_rows = []     # list of (n_tracks,) arrays, one per sequence
    full_val_rows = []
    lora_val_rows = []
    has_bins = None

    for full_arr, lora_arr in zip(iter_pickle_file(full_tmp), iter_pickle_file(lora_tmp)):
        fa = full_arr.astype(np.float32)
        la = lora_arr.astype(np.float32)

        if has_bins is None:
            has_bins = (fa.ndim == 2)
            if not has_bins:
                print(f"  [compute_correlations] output has no bins/length axis "
                      f"(shape {fa.shape}) -- skipping per_seq, using across_seq only")

        if has_bins:
            per_seq_rows.append(_row_pearson(fa, la))
            full_val_rows.append(fa.mean(axis=1))
            lora_val_rows.append(la.mean(axis=1))
        else:
            full_val_rows.append(fa)
            lora_val_rows.append(la)

    full_vals = np.stack(full_val_rows, axis=0)   # (n_seq, n_tracks)
    lora_vals = np.stack(lora_val_rows, axis=0)   # (n_seq, n_tracks)

    if has_bins:
        per_seq_mat = np.stack(per_seq_rows, axis=0)   # (n_seq, n_tracks)
        per_seq_corr = np.nanmean(per_seq_mat, axis=0)  # (n_tracks,)
    else:
        per_seq_corr = None

    full_ranks = np.apply_along_axis(rankdata, 0, full_vals)
    lora_ranks = np.apply_along_axis(rankdata, 0, lora_vals)
    across_seq_corr = _col_pearson(full_ranks, lora_ranks)  # spearman, (n_tracks,)

    return per_seq_corr, across_seq_corr


def compute_pooled_correlations(full_tmp: str, lora_tmp: str, n_tracks: int) -> np.ndarray:
    """
    Pooled per-track Pearson correlation across all (sequence, bin) pairs
    at once (no per-sequence centering) -- combines within-sequence shape
    fidelity (per_seq) and across-sequence ranking fidelity (across_seq)
    into one number, automatically weighted by how much variance each
    source actually contributes (the standard ANOVA decomposition: total
    covariance = between-sequence + within-sequence covariance). Streams
    via running sums so we never materialize the full
    (n_tracks x n_seq x n_bins) array. For Sei (no bins axis), this
    collapses to exactly the across_seq metric since there's no
    within-sequence term to blend in.

    NaN/Inf values are masked out per-element before accumulating -- a
    running sum has no way to "skip" a bad value after the fact the way
    per_seq's per-sequence nanmean can, so a single non-finite value would
    otherwise poison that track's entire running total for every remaining
    sequence in the stream.
    """
    n   = np.zeros(n_tracks, dtype=np.float64)
    sx  = np.zeros(n_tracks, dtype=np.float64)
    sy  = np.zeros(n_tracks, dtype=np.float64)
    sxx = np.zeros(n_tracks, dtype=np.float64)
    syy = np.zeros(n_tracks, dtype=np.float64)
    sxy = np.zeros(n_tracks, dtype=np.float64)

    for full_arr, lora_arr in zip(iter_pickle_file(full_tmp), iter_pickle_file(lora_tmp)):
        fa = full_arr.astype(np.float64)
        la = lora_arr.astype(np.float64)
        if fa.ndim == 1:            # Sei: no bins axis, one value per track
            fa = fa[:, None]
            la = la[:, None]

        valid = np.isfinite(fa) & np.isfinite(la)   # (n_tracks, n_bins)
        fa_c = np.where(valid, fa, 0.0)
        la_c = np.where(valid, la, 0.0)

        n   += valid.sum(axis=1)
        sx  += fa_c.sum(axis=1)
        sy  += la_c.sum(axis=1)
        sxx += (fa_c * fa_c).sum(axis=1)
        syy += (la_c * la_c).sum(axis=1)
        sxy += (fa_c * la_c).sum(axis=1)

    num = n * sxy - sx * sy
    den = np.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((den > 0) & (n > 1), num / den, np.nan)


def compute_correlations_ag(full_tmp: str, lora_tmp: str, output_keys):
    """
    Same three metrics as compute_correlations/compute_pooled_correlations,
    now computed PER TRACK within each AG output key (not one scalar per
    key), since each key's tensor retains a real track dimension after the
    extract_ag_arrays fix. Reuses the exact same _row_pearson/_col_pearson
    machinery as the Borzoi/Enformer path -- just run once per key, since
    different keys have different track/bin counts and can't be stacked
    into one uniform array together.

    Returns three dicts, each keyed by output key, each value an array of
    length n_tracks_for_that_key:
        per_seq[key]    -- Pearson within each sequence (across bins),
                            averaged over sequences.
        across_seq[key] -- Spearman across sequences of each track's
                            per-sequence mean.
        pooled[key]     -- Pearson across all (sequence, bin) pairs at
                            once, no per-sequence centering.
    """
    first_full = next(iter_pickle_file(full_tmp))
    n_tracks_by_key = {k: first_full[k].shape[0] for k in output_keys}

    per_seq_rows  = {k: [] for k in output_keys}
    full_val_rows = {k: [] for k in output_keys}
    lora_val_rows = {k: [] for k in output_keys}
    n_rows   = {k: [] for k in output_keys}
    sx_rows  = {k: [] for k in output_keys}
    sy_rows  = {k: [] for k in output_keys}
    sxx_rows = {k: [] for k in output_keys}
    syy_rows = {k: [] for k in output_keys}
    sxy_rows = {k: [] for k in output_keys}

    for full_d, lora_d in zip(iter_pickle_file(full_tmp), iter_pickle_file(lora_tmp)):
        for k in output_keys:
            fa = full_d[k].astype(np.float64)   # (n_tracks_key, n_bins_key)
            la = lora_d[k].astype(np.float64)

            per_seq_rows[k].append(_row_pearson(fa, la))

            # nanmean rather than mean -- a handful of non-finite bins
            # shouldn't turn this track's whole sequence-level summary
            # into NaN and corrupt across_seq's rank-based correlation.
            with np.errstate(invalid="ignore"):
                full_val_rows[k].append(np.nanmean(fa, axis=1))
                lora_val_rows[k].append(np.nanmean(la, axis=1))

            # Mask non-finite elements before accumulating -- a running sum
            # has no way to "skip" a bad value after the fact, so a single
            # NaN/Inf would otherwise poison this track's entire pooled
            # total for every remaining sequence in the stream.
            valid = np.isfinite(fa) & np.isfinite(la)
            fa_c = np.where(valid, fa, 0.0)
            la_c = np.where(valid, la, 0.0)
            n_rows[k].append(valid.sum(axis=1))
            sx_rows[k].append(fa_c.sum(axis=1))
            sy_rows[k].append(la_c.sum(axis=1))
            sxx_rows[k].append((fa_c * fa_c).sum(axis=1))
            syy_rows[k].append((la_c * la_c).sum(axis=1))
            sxy_rows[k].append((fa_c * la_c).sum(axis=1))

    per_seq = {}
    across_seq = {}
    pooled = {}
    for k in output_keys:
        per_seq_mat = np.stack(per_seq_rows[k], axis=0)   # (n_seq, n_tracks_key)
        with np.errstate(invalid="ignore"):
            per_seq[k] = np.nanmean(per_seq_mat, axis=0)

        full_vals = np.stack(full_val_rows[k], axis=0)
        lora_vals = np.stack(lora_val_rows[k], axis=0)
        full_ranks = np.apply_along_axis(rankdata, 0, full_vals)
        lora_ranks = np.apply_along_axis(rankdata, 0, lora_vals)
        across_seq[k] = _col_pearson(full_ranks, lora_ranks)

        n_   = np.stack(n_rows[k],   axis=0).sum(axis=0)
        sx_  = np.stack(sx_rows[k],  axis=0).sum(axis=0)
        sy_  = np.stack(sy_rows[k],  axis=0).sum(axis=0)
        sxx_ = np.stack(sxx_rows[k], axis=0).sum(axis=0)
        syy_ = np.stack(syy_rows[k], axis=0).sum(axis=0)
        sxy_ = np.stack(sxy_rows[k], axis=0).sum(axis=0)
        num = n_ * sxy_ - sx_ * sy_
        den = np.sqrt((n_ * sxx_ - sx_ ** 2) * (n_ * syy_ - sy_ ** 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            pooled[k] = np.where((den > 0) & (n_ > 1), num / den, np.nan)

    return per_seq, across_seq, pooled


def group_correlations(track_corrs: np.ndarray, track_groups: pd.Series, label: str) -> pd.DataFrame:
    """
    std is real track-to-track spread within the assay group (sample std,
    ddof=1).
    """
    df = pd.DataFrame({
        "track_index": np.arange(len(track_corrs)),
        label: track_corrs,
    })
    df["assay_group"] = df["track_index"].map(track_groups)
    summary = df.groupby("assay_group")[label].agg(
        mean="mean", std="std", n_tracks="count"
    ).sort_values("mean")
    return summary, df


# ----------------------------------------------------------------
# Main per-model driver
# ----------------------------------------------------------------
def calculate_correlations(model_name: str) -> Dict:
    input_len = MODEL_PARAMS[model_name]["input_len"] if model_name in MODEL_PARAMS else 4096
    dataset = SeqDataset(
        file_path=BED, scores_path=False, fasta_path=FASTA,
        window_size=input_len, mode="test", test_chrom=["chr8", "chr9"],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        full_tmp = os.path.join(tmpdir, "full.pkl")

        print_gpu_mem("before full model load")
        full_model = initialize_model(k_l="full", k_c="full", quant=False, model_name=model_name, full=True)
        full_model.to(DEVICE)
        print_gpu_mem("after full model load")

        # Sei's track metadata lives on the loaded model itself; grab it
        # before the model is deleted. Borzoi/Enformer pull it independently
        # from gReLU's zoo, so full_model isn't needed there.
        track_groups = get_track_groups(model_name, full_model=full_model)
        n_tracks = len(track_groups)

        run_inference_to_file(full_model, dataset, "Full model", full_tmp)
        del full_model
        free_gpu()
        print_gpu_mem("after full model deleted")

        per_seq_by_rank = {}
        across_seq_by_rank = {}
        pooled_by_rank = {}
        per_seq_group_by_rank = {}
        across_seq_group_by_rank = {}
        pooled_group_by_rank = {}

        for r in RANKS:
            lora_tmp = os.path.join(tmpdir, f"rank_{r}.pkl")

            print_gpu_mem(f"before rank {r} load")
            lora_model = initialize_model(k_l=r, k_c="full", quant=False, model_name=model_name, full=False)
            lora_model.to(DEVICE)
            print_gpu_mem(f"after rank {r} load")

            run_inference_to_file(lora_model, dataset, f"LoRA rank {r}", lora_tmp)
            del lora_model
            free_gpu()
            print_gpu_mem(f"after rank {r} deleted")

            per_seq_corr, across_seq_corr = compute_correlations(full_tmp, lora_tmp, n_tracks)
            pooled_corr = compute_pooled_correlations(full_tmp, lora_tmp, n_tracks)

            # per_seq_corr is None when the model's output has no bins/length
            # axis (e.g. Sei) -- nothing to group/average in that case.
            if per_seq_corr is not None:
                per_seq_group, per_seq_df = group_correlations(per_seq_corr, track_groups, "per_seq_pearson")
                per_seq_by_rank[r] = per_seq_df
                per_seq_group_by_rank[r] = per_seq_group
                print(f"\n[{model_name}] rank {r} — per-seq Pearson by assay group:")
                print(per_seq_group)
            else:
                per_seq_by_rank[r] = None
                per_seq_group_by_rank[r] = None

            across_seq_group, across_seq_df = group_correlations(across_seq_corr, track_groups, "across_seq_spearman")
            across_seq_by_rank[r] = across_seq_df
            across_seq_group_by_rank[r] = across_seq_group

            print(f"\n[{model_name}] rank {r} — across-seq Spearman by assay group:")
            print(across_seq_group)

            pooled_group, pooled_df = group_correlations(pooled_corr, track_groups, "pooled_pearson")
            pooled_by_rank[r] = pooled_df
            pooled_group_by_rank[r] = pooled_group

            print(f"\n[{model_name}] rank {r} — pooled Pearson by assay group:")
            print(pooled_group)

            os.remove(lora_tmp)

    return {
        "per_seq_per_track": per_seq_by_rank,
        "across_seq_per_track": across_seq_by_rank,
        "pooled_per_track": pooled_by_rank,
        "per_seq_by_group": per_seq_group_by_rank,
        "across_seq_by_group": across_seq_group_by_rank,
        "pooled_by_group": pooled_group_by_rank,
    }


def calculate_correlations_ag() -> Dict:
    """
    Same overall shape of result as calculate_correlations, but for AG:
    output is a dict of per-key (n_tracks, n_bins) arrays rather than one
    single tensor covering everything, so inference writing / correlation
    computation are done per key (compute_correlations_ag) and then
    concatenated into one long per-track vector -- with a matching
    per-track group-label vector built by repeating each key's group label
    across its own tracks -- so the existing group_correlations can be
    reused completely unchanged.
    """
    input_len = MODEL_PARAMS["ag"]["input_len"]
    dataset = SeqDataset(
        file_path=BED, scores_path=False, fasta_path=FASTA,
        window_size=input_len, mode="test", test_chrom=["chr8", "chr9"],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        full_tmp = os.path.join(tmpdir, "full.pkl")

        print_gpu_mem("before full model load")
        full_model = initialize_model_ag(full=True)
        full_model.to(DEVICE)
        print_gpu_mem("after full model load")

        run_inference_to_file_ag(full_model, dataset, "Full model", full_tmp)
        del full_model
        free_gpu()
        print_gpu_mem("after full model deleted")

        # Output keys are only known once we've seen a real record, same
        # as your original AG script.
        output_keys = list(next(iter_pickle_file(full_tmp)).keys())
        print(f"Output keys: {output_keys}")
        key_groups = get_ag_key_groups(output_keys)

        per_seq_by_rank = {}
        across_seq_by_rank = {}
        pooled_by_rank = {}
        per_seq_group_by_rank = {}
        across_seq_group_by_rank = {}
        pooled_group_by_rank = {}

        for r in RANKS:
            lora_tmp = os.path.join(tmpdir, f"rank_{r}.pkl")

            print_gpu_mem(f"before rank {r} load")
            lora_model = initialize_model_ag(rank=r, full=False)
            lora_model.to(DEVICE)
            print_gpu_mem(f"after rank {r} load")

            run_inference_to_file_ag(lora_model, dataset, f"LoRA rank {r}", lora_tmp)
            del lora_model
            free_gpu()
            print_gpu_mem(f"after rank {r} deleted")

            per_seq_by_key, across_seq_by_key, pooled_by_key = compute_correlations_ag(full_tmp, lora_tmp, output_keys)

            def concat_with_groups(per_key_dict):
                vals = np.concatenate([per_key_dict[k] for k in output_keys])
                labels = np.concatenate([
                    np.full(len(per_key_dict[k]), key_groups[k]) for k in output_keys
                ])
                groups_series = pd.Series(labels, index=np.arange(len(labels)))
                return vals, groups_series

            per_seq_vals, per_seq_groups = concat_with_groups(per_seq_by_key)
            across_vals, across_groups = concat_with_groups(across_seq_by_key)
            pooled_vals, pooled_groups = concat_with_groups(pooled_by_key)

            per_seq_group, per_seq_df = group_correlations(per_seq_vals, per_seq_groups, "per_seq_pearson")
            per_seq_by_rank[r] = per_seq_df
            per_seq_group_by_rank[r] = per_seq_group
            print(f"\n[ag] rank {r} — per-seq Pearson by assay group:")
            print(per_seq_group)

            across_seq_group, across_seq_df = group_correlations(across_vals, across_groups, "across_seq_spearman")
            across_seq_by_rank[r] = across_seq_df
            across_seq_group_by_rank[r] = across_seq_group
            print(f"\n[ag] rank {r} — across-seq Spearman by assay group:")
            print(across_seq_group)

            pooled_group, pooled_df = group_correlations(pooled_vals, pooled_groups, "pooled_pearson")
            pooled_by_rank[r] = pooled_df
            pooled_group_by_rank[r] = pooled_group
            print(f"\n[ag] rank {r} — pooled Pearson by assay group:")
            print(pooled_group)

            os.remove(lora_tmp)

    return {
        "per_seq_per_track": per_seq_by_rank,
        "across_seq_per_track": across_seq_by_rank,
        "pooled_per_track": pooled_by_rank,
        "per_seq_by_group": per_seq_group_by_rank,
        "across_seq_by_group": across_seq_group_by_rank,
        "pooled_by_group": pooled_group_by_rank,
    }


def main():
    for model_name in ("ag", None):
        print(f"\n=== {model_name} ===")
    
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print_gpu_mem(f"before {model_name}")

        if model_name == "ag":
            results = calculate_correlations_ag()
        else:
            results = calculate_correlations(model_name)
        with open(f"{model_name}_track_corrs.pkl", "wb") as f:
            pickle.dump(results, f)
        print(f"Saved {model_name}_track_corrs.pkl")


if __name__ == "__main__":
    main()