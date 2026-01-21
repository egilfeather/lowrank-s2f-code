import os
import time
import csv
import torch
import numpy as np
import psutil
from threading import Thread, Event

from typing import Optional, Literal
import torch.nn as nn
import seimodel as sm
import seillra as sl

from torch.ao.quantization import get_default_qconfig, QConfigMapping, quantize_fx
from borzoi_lora_arch_mha import BorzoiModel, EnformerModel

# ---------------- CPU isolation ----------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
# -----------------------------------------------

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
    

class CPUMonitor:
    """Monitor CPU usage by OTHER processes on a specific core during benchmarking."""
    
    def __init__(self, cpu_id: int, avg_threshold: float = 1.0, sample_interval: float = 0.01):
        """
        Args:
            cpu_id: Specific CPU core ID to monitor
            avg_threshold: Maximum allowed AVERAGE usage by other processes (%)
            sample_interval: How often to sample CPU usage (seconds)
        """
        self.cpu_id = cpu_id
        self.avg_threshold = avg_threshold
        self.sample_interval = sample_interval
        self.samples = []
        self.stop_event = Event()
        self.monitor_thread = None
        self.our_process = psutil.Process()
        
    def _get_other_processes_cpu_usage(self):
        """Calculate CPU usage on specific core by all processes except ours."""
        try:
            # Get per-CPU usage
            cpu_percent_all = psutil.cpu_percent(interval=None, percpu=True)
            if self.cpu_id >= len(cpu_percent_all):
                return 0.0
            
            total_usage = cpu_percent_all[self.cpu_id]
            
            # Get our process CPU usage (percentage across all CPUs)
            our_usage = self.our_process.cpu_percent(interval=None)
            
            # Get CPU affinity to calculate our usage on this specific CPU
            try:
                affinity = self.our_process.cpu_affinity()
                if len(affinity) > 0:
                    # Our usage on this specific CPU (assuming equal distribution across affinity)
                    our_usage_this_cpu = our_usage / len(affinity) if self.cpu_id in affinity else 0.0
                else:
                    our_usage_this_cpu = our_usage / psutil.cpu_count()
            except:
                our_usage_this_cpu = our_usage / psutil.cpu_count()
            
            # Usage by other processes
            other_usage = max(0.0, total_usage - our_usage_this_cpu)
            
            return other_usage
            
        except Exception as e:
            print(f"[WARN] Error calculating CPU usage: {e}")
            return 0.0
    
    def _monitor_loop(self):
        """Background thread that monitors CPU usage by other processes."""
        # Initial call to initialize cpu_percent
        psutil.cpu_percent(interval=None, percpu=True)
        self.our_process.cpu_percent(interval=None)
        
        while not self.stop_event.is_set():
            time.sleep(self.sample_interval)
            other_usage = self._get_other_processes_cpu_usage()
            self.samples.append(other_usage)
    
    def start(self):
        """Start monitoring."""
        self.samples = []
        self.stop_event.clear()
        
        # Reset CPU percent counters
        psutil.cpu_percent(interval=None, percpu=True)
        self.our_process.cpu_percent(interval=None)
        
        self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        time.sleep(0.05)  # Give it a moment to start collecting
    
    def stop(self):
        """Stop monitoring and return statistics."""
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        if not self.samples:
            return 0.0, 0.0, False
        
        avg_usage = np.mean(self.samples)
        max_usage = np.max(self.samples)
        
        # Contamination based on AVERAGE, not max
        contaminated = avg_usage > self.avg_threshold
        
        return avg_usage, max_usage, contaminated


def set_cpu_affinity(cpu_id: int):
    """Set process affinity to a specific CPU core."""
    try:
        # Linux
        os.sched_setaffinity(0, {cpu_id})
        print(f"[INFO] Process pinned to CPU {cpu_id}")
        return True
    except AttributeError:
        # macOS/Windows - try psutil
        try:
            p = psutil.Process()
            p.cpu_affinity([cpu_id])
            print(f"[INFO] Process pinned to CPU {cpu_id} (via psutil)")
            return True
        except Exception as e:
            print(f"[WARN] Could not set CPU affinity: {e}")
            return False


def initialize_model(model_name: str, dummy_input: torch.Tensor, rank: int, device: torch.device):
    """
    Initializes the model and applies FX quantization for Borzoi/Enformer.
    Loads pre-quantized weights into the FX-quantized model.
    Sei remains unquantized.
    """
    model_quantized = None
    is_quantized = False

    if model_name.lower() == "borzoi" or model_name.lower() == "enformer" :
        lora_weights_dir = f"./{model_name}_lora_weights"
        if model_name.lower() == "borzoi":
            model = BorzoiModel(
                k_l=rank,
                device=device,
                n_tasks=7611,
                crop_len=5120,
                final_act_func="softplus",
                final_pool_func=None,
            )
        else:
            model = EnformerModel(
                k_l=rank,
                device=device,
                n_tasks=5313,
                crop_len=320,
                final_act_func="softplus",
                final_pool_func=None,
            )
        model.eval()
        if rank == "full":
            base_name = f"{model_name}_lora_lr{rank}"
            quant_path = os.path.join(lora_weights_dir, f"{base_name}.pth")
            state_dict = torch.load(quant_path, weights_only=True)
            model.load_state_dict(state_dict, strict = True)
            model_quantized = model  # No quantization for full rank
        else:
            base_name = f"{model_name}_lora_lr{rank}l"
            quant_path = os.path.join(lora_weights_dir, f"{base_name}.pth")

            state_dict = torch.load(quant_path, weights_only=True)
            model.load_state_dict(state_dict, strict = True)
            model_quantized = model

    elif model_name.lower() == "sei":
        if rank == "full":
            model_quantized = SeiWrapper(k=rank, projection=False, mode="sequence", device="cuda")
            model_quantized.to(device)
        else:
            model_quantized = sl.Sei_LLRA(k=rank, projection=False, mode="sequence")
            model_quantized.to(device)
            print(model_quantized)
            is_quantized = True

    else:
        raise ValueError(f"Unknown model: {model_name}")

    model_quantized.eval()
    model_quantized.to(device)

    return model_quantized, is_quantized


def benchmark_model(
    model, 
    dummy_input, 
    model_name: str, 
    rank: int, 
    cpu_monitor: CPUMonitor,
    n_warmup: int = 2, 
    n_iter: int = 20,
    max_retries: int = 3,
    wait_on_contamination: bool = True
):
    """
    Benchmark model with CPU monitoring of OTHER processes.
    Uses AVERAGE CPU usage threshold for contamination detection.
    
    Args:
        wait_on_contamination: If True, wait and retry when CPU is contaminated.
                               If False, skip contaminated measurements.
    """
    total_times = []
    contamination_info = []

    with torch.no_grad():
        # Warmup
        print(f"[INFO] Warming up ({n_warmup} iterations)...")
        for _ in range(n_warmup):
            _ = model(dummy_input)

        # Timed runs
        iteration = 0
        attempts = 0
        while iteration < n_iter and attempts < n_iter * max_retries:
            attempts += 1
            
            # Start monitoring
            cpu_monitor.start()
            
            # Run inference
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            
            # Stop monitoring and check
            avg_usage, max_usage, contaminated = cpu_monitor.stop()
            elapsed_ms = (end - start) * 1000.0
            
            if contaminated:
                print(f"[WARN] Iteration {iteration+1} contaminated: "
                      f"OTHER processes AVERAGE={avg_usage:.2f}% (threshold={cpu_monitor.avg_threshold:.2f}%) "
                      f"max={max_usage:.1f}%")
                
                if wait_on_contamination:
                    print(f"[INFO] Waiting 2 seconds before retry...")
                    time.sleep(2.0)
                    continue  # Retry
                else:
                    print(f"[INFO] Skipping contaminated measurement")
                    continue  # Skip this measurement
            
            # Valid measurement
            total_times.append(elapsed_ms)
            contamination_info.append({
                'avg_usage': avg_usage,
                'max_usage': max_usage,
                'contaminated': False
            })
            iteration += 1
            print(f"[OK] Iteration {iteration}/{n_iter}: {elapsed_ms:.2f} ms "
                  f"(OTHER processes: avg={avg_usage:.2f}%, max={max_usage:.1f}%)")

    if len(total_times) < n_iter:
        print(f"[WARN] Only got {len(total_times)}/{n_iter} clean measurements")

    avg = float(np.mean(total_times)) if total_times else 0.0
    med = float(np.median(total_times)) if total_times else 0.0

    print(f"\n{model_name} (rank={rank}) | mean = {avg:.2f} ms | median = {med:.2f} ms")

    return total_times, avg, med, contamination_info


def append_csv(csv_path, rows):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "model", "rank", "iteration", "forward_time_ms", 
                "other_cpu_avg_usage", "other_cpu_max_usage", "contaminated"
            ])
        for row in rows:
            writer.writerow(row)


def main(
    model_name, 
    length=524288, 
    cpu_id=4, 
    avg_threshold=1.0,
    wait_on_contamination=True
):
    """
    Args:
        model_name: Name of the model to benchmark
        length: Input sequence length
        cpu_id: Specific CPU core ID to use (e.g., 4 for CPU #4)
        avg_threshold: Maximum allowed AVERAGE CPU usage by OTHER processes (%)
        wait_on_contamination: If True, wait and retry. If False, skip.
    """
    # Set CPU affinity
    set_cpu_affinity(cpu_id)
    
    # Initialize CPU monitor
    cpu_monitor = CPUMonitor(cpu_id=cpu_id, avg_threshold=avg_threshold)
    
    ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, "full" ] #, 1024, 2048]
    device = torch.device("cpu")
    dummy_input = torch.randn(1, 4, length, device=device)

    for rank in ranks:
        print(f"\n{'='*60}")
        print(f"Benchmarking {model_name} with rank={rank}")
        print(f"{'='*60}")
        
        model, is_quantized = initialize_model(
            model_name=model_name,
            dummy_input=dummy_input,
            rank=rank,
            device=device,
        )
   

        times, avg, med, contamination_info = benchmark_model(
            model,
            dummy_input,
            model_name,
            rank,
            cpu_monitor,
            wait_on_contamination=wait_on_contamination
        )

        csv_rows = [
            [
                model_name, 
                rank, 
                i + 1, 
                f"{t:.4f}",
                f"{info['avg_usage']:.2f}",
                f"{info['max_usage']:.2f}",
                info['contaminated']
            ]
            for i, (t, info) in enumerate(zip(times, contamination_info))
        ]

        csv_name = f"{model_name}_count_times_noquant.csv"
        append_csv(csv_name, csv_rows)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark models with CPU monitoring")
    parser.add_argument("--cpu-id", type=int, default=4, 
                        help="CPU core ID to use (default: 4)")
    parser.add_argument("--avg-threshold", type=float, default=1.0,
                        help="Max AVERAGE CPU usage by OTHER processes in %% (default: 1.0)")
    parser.add_argument("--mode", choices=["wait", "skip"], default="wait",
                        help="Action on contamination: wait and retry, or skip (default: wait)")
    
    args = parser.parse_args()
    
    wait_mode = (args.mode == "wait")
    
    print(f"[CONFIG] CPU ID: {args.cpu_id}")
    print(f"[CONFIG] AVERAGE CPU Threshold (OTHER processes): {args.avg_threshold}%")
    print(f"[CONFIG] Contamination mode: {args.mode}")
    print()
    
    main(model_name="borzoi", length=524288, 
         cpu_id=args.cpu_id, avg_threshold=args.avg_threshold,
         wait_on_contamination=wait_mode)
    main(model_name="enformer", length=196608,
         cpu_id=args.cpu_id, avg_threshold=args.avg_threshold,
         wait_on_contamination=wait_mode)
    main(model_name="sei", length=4096,
         cpu_id=args.cpu_id, avg_threshold=args.avg_threshold,
         wait_on_contamination=wait_mode)