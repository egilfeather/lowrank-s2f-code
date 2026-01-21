#!/usr/bin/env python3


import os
import sys
import torch
import torch.nn as nn


from borzoi_lora_arch_mha import BorzoiModel, EnformerModel
import seillra as sl
import seimodel as sm


MODEL_PARAMS = {
    "borzoi": {"input_len": 524288},
    "enformer": {"input_len": 196608},
    "sei": {"input_len": 4096},
}


class SeiWrapper(nn.Module):
    def __init__(self, k: int, ft=None, projection: bool = True, mode="sequence", device: str = "cpu"):
        super().__init__()
        self.device = device
        self.mode = mode
        self.projection = projection
        self.head = sm.get_sei_head().load_weights()
        self.trunk = sm.get_sei_trunk().load_weights()
        if self.projection:
            self.proj = sm.get_sei_projection().load_weights()
            self.proj.set_mode(mode)

    def forward(self, x):
        if self.projection:
            for_x = self.trunk(x)
            for_x = self.head(for_x)
            rev_x = torch.flip(x, dims=[1, 2])
            rev_x = self.trunk(rev_x)
            rev_x = self.head(rev_x)
            out = (for_x + rev_x) / 2
            out = self.proj(out)
        else:
            for_x = self.trunk(x)
            for_x = self.head(for_x)
            rev_x = torch.flip(x, dims=[1, 2])
            rev_x = self.trunk(rev_x)
            rev_x = self.head(rev_x)
            out = (for_x + rev_x) / 2
        return out


def initialize_model(model_name: str, dummy_input: torch.Tensor, rank, device):
    if device == "cpu":
        quant = "CPU"
    else:
        quant = None

    if model_name.lower() == "borzoi":
        lora_weights_dir = "./borzoi_lora_weights"
        model = BorzoiModel(
            k_l=rank, k_c="full", device=device,
            n_tasks=7611, crop_len=5120,
            final_act_func="softplus", final_pool_func=None,
        )
        model.eval()
        base_name = f"borzoi_lora_lr{rank}"
        quant_path = os.path.join(lora_weights_dir, f"{base_name}.pth")
        state_dict = torch.load(quant_path, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        
    elif model_name.lower() == "enformer":
        lora_weights_dir = "./enformer_lora_weights"
        model = EnformerModel(
            k_l=rank, k_c="full", device=device,
            n_tasks=5313, crop_len=320,
            final_act_func="softplus", final_pool_func=None,
        )
        model.eval()
        base_name = f"enformer_lora_lr{rank}"
        quant_path = os.path.join(lora_weights_dir, f"{base_name}.pth")
        state_dict = torch.load(quant_path, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        
    elif model_name.lower() == "sei":
        if rank == "full":
            model = SeiWrapper(k=rank, projection=False, mode="sequence", device=device)
        else:
            model = sl.Sei_LLRA(k=rank, projection=False, mode="sequence", quant=quant)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.eval()
    model.to(device)
    return model


# =============================================================================
# Option 2: Using ptflops (simple and reliable)
# =============================================================================
def count_macs_ptflops(model, input_shape):
    """
    Use ptflops for MAC counting.
    Install: pip install ptflops
    
    Returns MACs and number of parameters.
    """
    from ptflops import get_model_complexity_info
    
    # ptflops expects input shape without batch dimension for some models
    # For (B, C, L) input, pass (C, L)
    if len(input_shape) == 3:
        input_res = (input_shape[1], input_shape[2])
    else:
        input_res = input_shape[1:]
    
    macs, params = get_model_complexity_info(
        model, 
        input_res,
        as_strings=False,
        print_per_layer_stat=True,
        verbose=True
    )
    
    return macs, params




# =============================================================================
# Main function using your preferred package
# =============================================================================
def calculate_macs_per_sequence(model_name: str):
    """
    Calculate MACs for different ranks using specified package.
    
    Args:
        model_name: "borzoi", "enformer", or "sei"
    """
    ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, "full"]
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    
    seq_len = MODEL_PARAMS.get(model_name, {}).get("input_len", 4096)
    x = torch.zeros((1, 4, seq_len), dtype=torch.float32, device=device)
    
    macs_by_rank = {}
    
    for r in ranks:
        print(f"\n{'='*60}")
        print(f"Processing {model_name} rank={r}")
        print(f"{'='*60}")
        
        try:
            model = initialize_model(model_name, x, r, device)
            model.eval()
            
          
            macs, _ = count_macs_ptflops(model, x.shape)
            
            macs_by_rank[r] = macs
            print(f"\nRank {r}: MACs = {macs:,}")
            
            del model
            if 'cuda' in device:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing rank {r}: {e}")
            import traceback
            traceback.print_exc()
            macs_by_rank[r] = 0
    
    return macs_by_rank


def main():
    
    for model_name in ["borzoi", "enformer", "sei"]:
        print(f"\n{'#'*60}")
        print(f"# Processing {model_name.upper()}")
        print(f"{'#'*60}")
        
        macs = calculate_macs_per_sequence(model_name)
        
        output_file = f"{model_name}_flops.tsv"
        with open(output_file, "w") as f:
            f.write("rank\tmacs\n")
            for k, v in macs.items():
                f.write(f"{k}\t{v}\n")
        
        print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()