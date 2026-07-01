#!/usr/bin/env python3
"""
benchmark_inference_realistic.py
================================
REALISTIC-configuration latency benchmark for Sei, Borzoi, Enformer, and
AlphaGenome at each LoRA rank.

This is the "how fast does it run in practice" counterpart to the single-core
benchmark.  It measures models the way they're actually used: multi-threaded,
unpinned (the OS schedules across all cores), one dtype for the whole run.


Usage:
  # all models, fp32, all cores
  python benchmark_inference_realistic.py --model all

  # match a notebook that showed torch.get_num_threads() == 16
  BENCH_NUM_THREADS=16 python benchmark_inference_realistic.py --model alphagenome \
      --ag-seq-len 1048576 --ag-resolutions 128

  # bf16, more iterations for a tighter median
  python benchmark_inference_realistic.py --model borzoi --dtype bf16 --n-iter 30
"""

import os
import time
import csv
import argparse
import numpy as np
import psutil
import torch
import torch.nn as nn
from threading import Thread, Event

# ── Thread configuration (must precede heavy BLAS use) ────────────────────────
# Default: all logical cores (notebook-equivalent). The SAME count is applied to
# every model in the run so the comparison stays fair.
_N_THREADS = int(os.environ.get("BENCH_NUM_THREADS", str(os.cpu_count() or 1)))
os.environ["OMP_NUM_THREADS"]      = str(_N_THREADS)
os.environ["MKL_NUM_THREADS"]      = str(_N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(_N_THREADS)
os.environ["NUMEXPR_NUM_THREADS"]  = str(_N_THREADS)
torch.set_num_threads(_N_THREADS)
# Intra-op pool gets all the threads; leave inter-op at default for realism.
# ─────────────────────────────────────────────────────────────────────────────

# ── Model configs ─────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "sei": {
        "input_len": 4096,
        "ranks": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, "full"],
        "input_shape": lambda L: (1, 4, L),   # (B, C, S)
    },
    "borzoi": {
        "input_len": 524288,
        "ranks": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, "full"],
        "input_shape": lambda L: (1, 4, L),
    },
    "enformer": {
        "input_len": 196608,
        "ranks": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, "full"],
        "input_shape": lambda L: (1, 4, L),
    },
    "alphagenome": {
        "input_len": 1_048_576,
        "ranks": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, "full"],
        "input_shape": lambda L: (1, L, 4),   # NLC + organism index
    },
}

AG_SEQ_LEN_TAGS = {4096: "4kb", 1_048_576: "1Mb"}

KEY_REPLACEMENTS_BORZOI = [
    ("model.", ""), ("orig_layer.", ""),
    (".conv.layer.weight",    ".conv.layer.layer.weight"),
    (".conv.layer.bias",      ".conv.layer.layer.bias"),
    (".linear.layer.weight",  ".linear.layer.layer.weight"),
    (".linear.layer.bias",    ".linear.layer.layer.bias"),
    (".conv.weight",          ".conv.layer.weight"),
    (".conv.bias",            ".conv.layer.bias"),
    (".linear.weight",        ".linear.layer.weight"),
    (".linear.bias",          ".linear.layer.bias"),
    (".pointwise.weight",     ".pointwise.layer.weight"),
    (".pointwise.bias",       ".pointwise.layer.bias"),
    (".to_pos_k.weight",      ".to_pos_k.layer.weight"),
    (".to_v.weight",          ".to_v.layer.weight"),
    (".to_q.weight",          ".to_q.layer.weight"),
    (".to_k.weight",          ".to_k.layer.weight"),
    (".to_rel_k.weight",      ".to_rel_k.layer.weight"),
    (".to_out.weight",        ".to_out.layer.weight"),
    (".to_out.bias",          ".to_out.layer.bias"),
    (".0.0.weight",           ".0.0.layer.weight"),
    (".0.0.bias",             ".0.0.layer.bias"),
]

KEY_REPLACEMENTS_AG = [
    ("linear_embedding.weight",    "linear_embedding.layer.weight"),
    ("linear_embedding.bias",      "linear_embedding.layer.bias"),
    ("fc1.weight",                 "fc1.layer.weight"),
    ("fc1.bias",                   "fc1.layer.bias"),
    ("fc2.weight",                 "fc2.layer.weight"),
    ("fc2.bias",                   "fc2.layer.bias"),
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


# ── Weight loading helpers ────────────────────────────────────────────────────

def _remap(state_dict, replacements, skip_lora=True):
    out = {}
    for key, val in state_dict.items():
        if skip_lora and ".lora." in key:
            continue
        new_key = key
        for old, new in replacements:
            new_key = new_key.replace(old, new)
        out[new_key] = val
    return out


def _remap_borzoi(state_dict):
    out = {}
    for key, val in state_dict.items():
        if ".lora." in key:
            continue
        new_key = key.replace("model.", "").replace("orig_layer.", "")
        if ".channel_transform" in new_key:
            new_key = new_key.replace(".conv.layer.weight",   ".conv.layer.layer.weight")
            new_key = new_key.replace(".conv.layer.bias",     ".conv.layer.layer.bias")
            new_key = new_key.replace(".linear.layer.weight", ".linear.layer.layer.weight")
            new_key = new_key.replace(".linear.layer.bias",   ".linear.layer.layer.bias")
        for old, new in KEY_REPLACEMENTS_BORZOI[2:]:
            new_key = new_key.replace(old, new)
        out[new_key] = val
    return out


# ── Model initialisation ──────────────────────────────────────────────────────

def load_sei(rank):
    import seillra as sl
    k = None if rank == "full" else rank
    return sl.Sei_LLRA(k=k, projection=False, mode="sequence", quant=None, compile=False)


def load_borzoi(rank):
    from borzoi_lora_arch_mha import BorzoiModel
    model = BorzoiModel(
        k_l=rank, k_c="full", device="cpu",
        n_tasks=7611, crop_len=5120,
        final_act_func="softplus", final_pool_func=None,
    )
    sd = torch.load(f"borzoi_lora_weights/borzoi_lora_lr{rank}_crfull.pth", weights_only=True)
    model.load_state_dict(_remap_borzoi(sd), strict=True)
    return model


def load_enformer(rank):
    from borzoi_lora_arch_mha import EnformerModel
    model = EnformerModel(
        k_l=rank, k_c="full", device="cpu",
        n_tasks=5313, crop_len=320,
        final_act_func="softplus", final_pool_func=None,
    )
    sd = torch.load(f"enformer_lora_weights/enformer_lora_lr{rank}_crfull.pth", weights_only=True)
    model.load_state_dict(_remap_borzoi(sd), strict=True)
    return model


def load_alphagenome(rank):
    from ag_arch_quant import AlphaGenome
    from alphagenome_pytorch import AlphaGenome as AlphaGenomePT
    if rank == "full":
        return AlphaGenomePT.from_pretrained("model_all_folds.safetensors")
    model = AlphaGenome(k=rank)
    sd = torch.load(f"ag_lora_weights/ag_lora_lr{rank}.pth", weights_only=True)
    model.load_state_dict(_remap(sd, KEY_REPLACEMENTS_AG), strict=True)
    return model


LOADERS = {
    "sei": load_sei, "borzoi": load_borzoi,
    "enformer": load_enformer, "alphagenome": load_alphagenome,
}


# ── Input construction ────────────────────────────────────────────────────────

def make_input(model_name, seq_len, dtype):
    shape = MODEL_CONFIGS[model_name]["input_shape"](seq_len)
    x = torch.zeros(*shape, dtype=dtype)
    if model_name == "alphagenome":
        x[..., 0] = 1.0
        return (x, torch.tensor([0], dtype=torch.long))
    else:
        x[:, 0, :] = 1.0
        return (x,)


# ── Model preparation (single locked dtype, optional compile) ─────────────────

def prepare(model, dtype, compile_model, call_kwargs):
    model = model.eval().cpu()
    if dtype == torch.bfloat16:
        model = model.to(torch.bfloat16)
    runner = (torch.compile(model, mode="max-autotune", fullgraph=False)
              if compile_model else model)

    def call_fn(inputs):
        cast = tuple(x.to(dtype) if x.is_floating_point() else x for x in inputs)
        return runner(*cast, **call_kwargs)
    return call_fn


# ── System load monitor (informational; machine is NOT pinned here) ───────────

class SystemLoadMonitor:
    """Records average other-process CPU load (system-wide %) during an iter."""

    def __init__(self, sample_interval: float = 0.05):
        self.sample_interval = sample_interval
        self.samples = []
        self.stop_event = Event()
        self.thread = None
        self.proc = psutil.Process()
        self.ncpu = psutil.cpu_count() or 1

    def _sample(self):
        try:
            total = psutil.cpu_percent(interval=None)          # 0..100 (avg across cores)
            ours  = self.proc.cpu_percent(interval=None) / self.ncpu
            return max(0.0, total - ours)
        except Exception:
            return 0.0

    def _loop(self):
        psutil.cpu_percent(interval=None)
        self.proc.cpu_percent(interval=None)
        while not self.stop_event.is_set():
            time.sleep(self.sample_interval)
            self.samples.append(self._sample())

    def start(self):
        self.samples = []
        self.stop_event.clear()
        psutil.cpu_percent(interval=None)
        self.proc.cpu_percent(interval=None)
        self.thread = Thread(target=self._loop, daemon=True)
        self.thread.start()
        time.sleep(0.05)

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)
        return float(np.mean(self.samples)) if self.samples else 0.0


# ── Benchmarking ──────────────────────────────────────────────────────────────

def benchmark(call_fn, inputs, label, monitor, n_warmup, n_iter):
    times, loads = [], []
    with torch.no_grad():
        print(f"  [warmup] {n_warmup} iterations...")
        for w in range(n_warmup):
            t0 = time.perf_counter()
            call_fn(inputs)
            print(f"    [warmup {w+1}/{n_warmup}] {time.perf_counter()-t0:.1f} s", flush=True)

        for i in range(n_iter):
            monitor.start()
            t0 = time.perf_counter()
            call_fn(inputs)
            ms = (time.perf_counter() - t0) * 1000.0
            load = monitor.stop()
            times.append(ms)
            loads.append(load)
            print(f"  [OK] {i+1}/{n_iter}: {ms:.1f} ms (sys other-load {load:.1f}%)", flush=True)

    return times, loads


def summarize(times, seq_len):
    t = np.asarray(times, dtype=float)
    med = float(np.median(t))
    p25, p75 = (float(np.percentile(t, 25)), float(np.percentile(t, 75)))
    std = float(t.std(ddof=1)) if t.size > 1 else 0.0
    sec = med / 1000.0
    bp_per_s  = seq_len / sec if sec > 0 else 0.0
    seq_per_s = 1.0 / sec if sec > 0 else 0.0
    return dict(median_ms=med, p25_ms=p25, p75_ms=p75, std_ms=std,
                bp_per_s=bp_per_s, seq_per_s=seq_per_s)


# ── CSV output ────────────────────────────────────────────────────────────────

def append_iters(path, rows):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["model", "rank", "dtype", "n_threads", "seq_len",
                        "resolutions", "iteration", "forward_time_ms",
                        "sys_other_cpu_pct"])
        w.writerows(rows)


def append_summary(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["model", "rank", "dtype", "n_threads", "seq_len",
                        "resolutions", "median_ms", "p25_ms", "p75_ms", "std_ms",
                        "bp_per_s", "seq_per_s"])
        w.writerow(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_model(model_name, monitor, args, dtype, ag_resolutions):
    cfg = MODEL_CONFIGS[model_name]
    ranks = cfg["ranks"]
    seq_len = cfg["input_len"]

    call_kwargs = {}
    res_label = "n/a"
    tag = f"{args.dtype}_t{_N_THREADS}"
    if model_name == "alphagenome":
        seq_len = args.ag_seq_len
        call_kwargs["resolutions"] = list(ag_resolutions)
        res_label = "+".join(str(r) for r in ag_resolutions)
        len_tag = AG_SEQ_LEN_TAGS.get(seq_len, f"{seq_len}bp")
        tag = f"{len_tag}_res{'-'.join(str(r) for r in ag_resolutions)}_{tag}"

    iters_csv   = f"{model_name}_realistic_{tag}_iters.csv"
    summary_csv = f"{model_name}_realistic_{tag}_summary.csv"

    print(f"\n{'='*72}")
    print(f"  REALISTIC  {model_name.upper()}   seq_len={seq_len:,}   "
          f"dtype={args.dtype}   threads={_N_THREADS}   resolutions={res_label}")
    print(f"{'='*72}")

    summary_rows = []
    for rank in ranks:
        print(f"\n── rank={rank} ──────────────────────────────────────────")
        print("  Loading weights...")
        model = LOADERS[model_name](rank)
        call_fn = prepare(model, dtype, args.compile, call_kwargs)
        inputs = make_input(model_name, seq_len, dtype)

        label = f"{model_name} rank={rank}"
        times, loads = benchmark(call_fn, inputs, label, monitor,
                                 args.n_warmup, args.n_iter)

        append_iters(iters_csv, [
            [model_name, rank, args.dtype, _N_THREADS, seq_len, res_label,
             i + 1, f"{t:.4f}", f"{l:.2f}"]
            for i, (t, l) in enumerate(zip(times, loads))
        ])

        s = summarize(times, seq_len)
        append_summary(summary_csv, [
            model_name, rank, args.dtype, _N_THREADS, seq_len, res_label,
            f"{s['median_ms']:.4f}", f"{s['p25_ms']:.4f}", f"{s['p75_ms']:.4f}",
            f"{s['std_ms']:.4f}", f"{s['bp_per_s']:.2f}", f"{s['seq_per_s']:.6f}",
        ])
        summary_rows.append((rank, s))
        print(f"  → median={s['median_ms']:.1f} ms  IQR=[{s['p25_ms']:.1f}, "
              f"{s['p75_ms']:.1f}]  throughput={s['bp_per_s']:,.0f} bp/s")

        del model, call_fn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Pretty summary table
    print(f"\n  SUMMARY — {model_name}  (dtype={args.dtype}, threads={_N_THREADS})")
    print(f"  {'rank':>6} {'median_ms':>11} {'IQR_ms':>17} {'bp/s':>14} {'seq/s':>10}")
    for rank, s in summary_rows:
        print(f"  {str(rank):>6} {s['median_ms']:>11.1f} "
              f"[{s['p25_ms']:>6.1f},{s['p75_ms']:>7.1f}] "
              f"{s['bp_per_s']:>14,.0f} {s['seq_per_s']:>10.4f}")
    print(f"\n[DONE] {iters_csv} / {summary_csv}")


def main():
    p = argparse.ArgumentParser(description="Realistic (multi-core) inference latency benchmark")
    p.add_argument("--model", choices=["sei", "borzoi", "enformer", "alphagenome", "all"],
                   default="all")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32",
                   help="Locked for the whole run; the same dtype is applied to every model.")
    p.add_argument("--compile", action="store_true",
                   help="Wrap in torch.compile(max-autotune). Adds a long first-call "
                        "compile, especially at 1 Mb.")
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-iter",   type=int, default=15,
                   help="Timed iterations per rank (more = tighter median).")
    p.add_argument("--max-load", type=float, default=20.0,
                   help="Warn if mean system other-load exceeds this %% in any iter.")
    p.add_argument("--ag-seq-len", type=int, choices=[4096, 1048576], default=1048576)
    p.add_argument("--ag-resolutions", choices=["128", "1,128"], default="1,128")
    args = p.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    ag_resolutions = [1, 128] if args.ag_resolutions == "1,128" else [128]

    print(f"[INFO] REALISTIC mode — NOT pinned. threads={_N_THREADS} "
          f"(set BENCH_NUM_THREADS to change), dtype={args.dtype}")
    print(f"[INFO] torch.get_num_threads()={torch.get_num_threads()}  "
          f"logical cores={os.cpu_count()}")
    print(f"[INFO] Keep the machine quiet; check the sys_other_cpu_pct column "
          f"(warn threshold {args.max_load}%).")

    monitor = SystemLoadMonitor()

    models = (["sei", "borzoi", "enformer", "alphagenome"]
              if args.model == "all" else [args.model])
    for m in models:
        run_model(m, monitor, args, dtype, ag_resolutions)


if __name__ == "__main__":
    main()
