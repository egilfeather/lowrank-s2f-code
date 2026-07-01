#!/usr/bin/env python3
"""
benchmark_quantized_realistic.py
================================
Realistic (multi-core, unpinned) latency benchmark for the QUANTIZED models.

Covers:
  * AlphaGenome — the 4 static-quant variants from ag_static_quant.py
    (in4kb_res128, in4kb_res1-128, in1Mb_res128, in1Mb_res1-128), each timed
    float and int8.
  * Borzoi / Enformer — FX-static int8 loaded from *_quant.pth (rebuild
    skeleton + load_state_dict), plus float.
  * Sei — FX-static int8 loaded from seillra package, plus float

Methodology mirrors the realistic float harness: NOT pinned, one thread count
for the whole run (BENCH_NUM_THREADS, default all logical cores), median + IQR,
throughput in bp/sec, and a per-iteration system other-load audit column.
Keep these results in their own CSVs; never plot against single-core numbers.

Usage:
    BENCH_NUM_THREADS=16 python benchmark_quantized_realistic.py \
        --models alphagenome --ranks 1 8 64 512 --ag-weights ag_static_weights
"""

import os
import time
import csv
import argparse
import importlib

import numpy as np
import psutil
import torch
from threading import Thread, Event

# Threads fixed once, before heavy BLAS use (see realistic float harness notes).
_N_THREADS = int(os.environ.get("BENCH_NUM_THREADS", str(os.cpu_count() or 1)))
os.environ["OMP_NUM_THREADS"]      = str(_N_THREADS)
os.environ["MKL_NUM_THREADS"]      = str(_N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(_N_THREADS)
os.environ["NUMEXPR_NUM_THREADS"]  = str(_N_THREADS)
torch.set_num_threads(_N_THREADS)

# fbgemm int8 kernels are the target backend on x86 CPUs.
try:
    torch.backends.quantized.engine = "fbgemm"
except Exception:
    pass

import ag_static_quant as agq  # build_float_variant / build_skeleton / VARIANTS / make_onehot


# ── System load monitor (informational; unpinned run) ─────────────────────────
class SystemLoadMonitor:
    def __init__(self, sample_interval=0.05):
        self.sample_interval = sample_interval
        self.samples = []
        self.stop_event = Event()
        self.thread = None
        self.proc = psutil.Process()
        self.ncpu = psutil.cpu_count() or 1

    def _sample(self):
        try:
            total = psutil.cpu_percent(interval=None)
            ours = self.proc.cpu_percent(interval=None) / self.ncpu
            return max(0.0, total - ours)
        except Exception:
            return 0.0

    def _loop(self):
        psutil.cpu_percent(interval=None); self.proc.cpu_percent(interval=None)
        while not self.stop_event.is_set():
            time.sleep(self.sample_interval)
            self.samples.append(self._sample())

    def start(self):
        self.samples = []; self.stop_event.clear()
        psutil.cpu_percent(interval=None); self.proc.cpu_percent(interval=None)
        self.thread = Thread(target=self._loop, daemon=True)
        self.thread.start(); time.sleep(0.05)

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)
        return float(np.mean(self.samples)) if self.samples else 0.0


# ── Timing ────────────────────────────────────────────────────────────────────
def time_callable(call_fn, monitor, n_warmup, n_iter, label):
    times, loads = [], []
    with torch.no_grad():
        print(f"  [warmup] {n_warmup}...", flush=True)
        for w in range(n_warmup):
            t0 = time.perf_counter(); call_fn()
            print(f"    [warmup {w+1}/{n_warmup}] {time.perf_counter()-t0:.1f} s", flush=True)
        for i in range(n_iter):
            monitor.start()
            t0 = time.perf_counter(); call_fn()
            ms = (time.perf_counter() - t0) * 1000.0
            load = monitor.stop()
            times.append(ms); loads.append(load)
            print(f"  [OK] {i+1}/{n_iter}: {ms:.1f} ms (sys {load:.1f}%)", flush=True)
    return times, loads


def summarize(times, seq_len):
    t = np.asarray(times, float)
    med = float(np.median(t))
    sec = med / 1000.0
    return dict(
        median_ms=med,
        p25_ms=float(np.percentile(t, 25)),
        p75_ms=float(np.percentile(t, 75)),
        std_ms=float(t.std(ddof=1)) if t.size > 1 else 0.0,
        bp_per_s=(seq_len / sec) if sec > 0 else 0.0,
        seq_per_s=(1.0 / sec) if sec > 0 else 0.0,
    )


# ── CSV ────────────────────────────────────────────────────────────────────────
ITER_HEADER = ["model", "variant", "rank", "method", "n_threads", "seq_len",
               "resolutions", "iteration", "forward_time_ms", "sys_other_cpu_pct"]
SUMM_HEADER = ["model", "variant", "rank", "method", "n_threads", "seq_len",
               "resolutions", "median_ms", "p25_ms", "p75_ms", "std_ms",
               "bp_per_s", "seq_per_s"]


def _append(path, header, rows):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerows(rows)


def record(model, variant, rank, method, seq_len, res_label,
           times, loads, iters_csv, summ_csv):
    _append(iters_csv, ITER_HEADER, [
        [model, variant, rank, method, _N_THREADS, seq_len, res_label,
         i + 1, f"{t:.4f}", f"{l:.2f}"]
        for i, (t, l) in enumerate(zip(times, loads))
    ])
    s = summarize(times, seq_len)
    _append(summ_csv, SUMM_HEADER, [[
        model, variant, rank, method, _N_THREADS, seq_len, res_label,
        f"{s['median_ms']:.4f}", f"{s['p25_ms']:.4f}", f"{s['p75_ms']:.4f}",
        f"{s['std_ms']:.4f}", f"{s['bp_per_s']:.2f}", f"{s['seq_per_s']:.6f}",
    ]])
    print(f"  -> {method}: median={s['median_ms']:.1f} ms  "
          f"IQR=[{s['p25_ms']:.1f},{s['p75_ms']:.1f}]  "
          f"{s['bp_per_s']:,.0f} bp/s", flush=True)


# ── AlphaGenome: 4 variants, float + int8 ──────────────────────────────────────
def run_alphagenome(args, monitor):
    iters_csv = "alphagenome_quant_realistic_iters.csv"
    summ_csv  = "alphagenome_quant_realistic_summary.csv"
    for rank in args.ranks:
        for vname, (input_len, resolutions) in agq.VARIANTS.items():
            res_label = "+".join(str(r) for r in resolutions)
            x, org = agq.make_onehot(input_len, dtype=torch.float32)

            print(f"\n== AlphaGenome rank={rank} {vname} "
                  f"(len={input_len:,}, res={res_label}) ==", flush=True)

            # float baseline
            if not args.int8_only:
                fmodel = agq.build_float_variant(rank, resolutions)
                times, loads = time_callable(
                    lambda: fmodel(x, org), monitor, args.n_warmup, args.n_iter,
                    f"{vname} float")
                record("alphagenome", vname, rank, "float", input_len, res_label,
                       times, loads, iters_csv, summ_csv)
                del fmodel

            # int8 (load saved variant)
            path = os.path.join(args.ag_weights, f"ag_static_{vname}_lr{rank}.pth")
            if not os.path.exists(path):
                print(f"  [skip int8] missing {path}", flush=True)
                continue
            qmodel = agq.build_skeleton(rank, input_len, resolutions)
            qmodel.load_state_dict(torch.load(path, map_location="cpu"))
            qmodel.eval()
            times, loads = time_callable(
                lambda: qmodel(x, org), monitor, args.n_warmup, args.n_iter,
                f"{vname} int8")
            record("alphagenome", vname, rank, "int8", input_len, res_label,
                   times, loads, iters_csv, summ_csv)
            del qmodel


# ── Borzoi / Enformer: FX-static int8 (rebuild + load), float baseline ─────────
def _grelu_config(model_name):
    if model_name == "borzoi":
        from borzoi_lora_arch_mha import BorzoiModel
        return dict(cls=BorzoiModel, length=524288, wdir="./borzoi_lora_weights",
                    kw=dict(n_tasks=7611, crop_len=5120,
                            final_act_func="softplus", final_pool_func=None))
    else:
        from borzoi_lora_arch_mha import EnformerModel
        return dict(cls=EnformerModel, length=196608, wdir="./enformer_lora_weights",
                    kw=dict(n_tasks=5313, crop_len=320,
                            final_act_func="softplus", final_pool_func=None))


def _build_grelu_float(cfg, rank):
    m = cfg["cls"](k_l=rank, k_c="full", device="cpu", **cfg["kw"]).eval()
    return m


def _build_grelu_int8(cfg, rank, dummy):
    from torch.ao.quantization import get_default_qconfig, QConfigMapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    base = _build_grelu_float(cfg, rank)
    _ = base(dummy)
    qmap = QConfigMapping().set_global(get_default_qconfig("fbgemm"))
    prepared = prepare_fx(base, qmap, dummy)
    qmodel = convert_fx(prepared)
    qpath = os.path.join(cfg["wdir"], f"{('borzoi' if cfg['length']==524288 else 'enformer')}_lora_lr{rank}_crfull_quant.pth")
    if os.path.exists(qpath):
        qmodel.load_state_dict(torch.load(qpath, map_location="cpu"))
    else:
        print(f"  [warn] missing {qpath}; timing int8 graph with FP-converted weights")
    return qmodel.eval()


def run_grelu(model_name, args, monitor):
    cfg = _grelu_config(model_name)
    L = cfg["length"]
    dummy = torch.randn(1, 4, L)
    iters_csv = f"{model_name}_quant_realistic_iters.csv"
    summ_csv  = f"{model_name}_quant_realistic_summary.csv"
    for rank in args.ranks:
        print(f"\n== {model_name} rank={rank} (len={L:,}) ==", flush=True)
        if not args.int8_only:
            fm = _build_grelu_float(cfg, rank)
            times, loads = time_callable(lambda: fm(dummy), monitor,
                                         args.n_warmup, args.n_iter, "float")
            record(model_name, "native", rank, "float", L, "native",
                   times, loads, iters_csv, summ_csv)
            del fm
        qm = _build_grelu_int8(cfg, rank, dummy)
        times, loads = time_callable(lambda: qm(dummy), monitor,
                                     args.n_warmup, args.n_iter, "int8")
        record(model_name, "native", rank, "int8", L, "native",
               times, loads, iters_csv, summ_csv)
        del qm


# ── Sei: float reference ───────────────────────────────────────────────────────
def run_sei(args, monitor):
    import seillra as sl
    L = 4096
    dummy = torch.randn(1, 4, L)
    iters_csv = "sei_quant_realistic_iters.csv"
    summ_csv  = "sei_quant_realistic_summary.csv"
    for rank in args.ranks:
        print(f"\n== sei rank={rank} (len={L:,}) ==", flush=True)
        model = sl.Sei_LLRA(k=rank, projection=False, mode="sequence",
                                  quant="CPU").eval()
        times, loads = time_callable(lambda: model(dummy), monitor,
                                     args.n_warmup, args.n_iter, "float")
        record("sei", "native", rank, "float", L, "native",
               times, loads, iters_csv, summ_csv)
        del model


def main():
    p = argparse.ArgumentParser(description="Realistic latency benchmark for quantized models")
    p.add_argument("--models", nargs="+",
                   default=["alphagenome"],
                   choices=["alphagenome", "borzoi", "enformer", "sei"])
    p.add_argument("--ranks", nargs="+", default=["1", "2", "4", "8", "16", "32", "64", "128", "256", "512"])
    p.add_argument("--ag-weights", default="ag_static_weights")
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-iter", type=int, default=15)
    p.add_argument("--int8-only", action="store_true",
                   help="Skip the float baseline (faster, but no delta).")
    args = p.parse_args()
    args.ranks = [r if r == "full" else int(r) for r in args.ranks]

    print(f"[INFO] REALISTIC quantized benchmark — NOT pinned. threads={_N_THREADS}, "
          f"engine={getattr(torch.backends.quantized,'engine','?')}", flush=True)
    print(f"[INFO] torch.get_num_threads()={torch.get_num_threads()} "
          f"logical cores={os.cpu_count()}", flush=True)

    monitor = SystemLoadMonitor()
    for m in args.models:
        if m == "alphagenome":
            run_alphagenome(args, monitor)
        elif m in ("borzoi", "enformer"):
            run_grelu(m, args, monitor)
        elif m == "sei":
            run_sei(args, monitor)


if __name__ == "__main__":
    main()
