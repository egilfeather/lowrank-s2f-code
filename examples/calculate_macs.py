#!/usr/bin/env python3


import os
import sys
import torch
import torch.nn as nn


from borzoi_lora_arch_mha import BorzoiModel, EnformerModel
import seillra as sl
import seimodel as sm

from ag_arch import AlphaGenome
from alphagenome_pytorch import AlphaGenome as AlphaGenomePT


MODEL_PARAMS = {
    "borzoi": {"input_len": 524288},
    "enformer": {"input_len": 196608},
    "sei": {"input_len": 4096},
    "ag": {"input_len": 1048576},
}
device = "cpu"

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


class AGFullWrapper(nn.Module):
    """
    Wraps AlphaGenomePT so ptflops can call forward(x) with a single tensor.
    organism_index=0 corresponds to human in the multi-organism model;
    adjust if you need a different organism.
    """
    def __init__(self, ag: nn.Module, organism_index: int = 0):
        super().__init__()
        self.ag = ag
        self.organism_index = torch.tensor([organism_index], dtype=torch.long).to(device)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        return self.ag(x, organism_index=self.organism_index)


def initialize_model(model_name: str, dummy_input: torch.Tensor, rank, device):

    if model_name.lower() == "borzoi":
        lora_weights_dir = "./borzoi_lora_weights"
        model = BorzoiModel(
            k_l=rank, k_c="full", device=device,
            n_tasks=7611, crop_len=5120,
            final_act_func="softplus", final_pool_func=None,
        )
        model.eval()
        base_name = f"borzoi_lora_lr{rank}_crfull"
        quant_path = os.path.join(lora_weights_dir, f"{base_name}.pth")
        state_dict = torch.load(quant_path, weights_only=True)
        # Apply key remapping
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if ".lora." in key:
                continue
            new_key = new_key.replace("model.", "").replace("orig_layer.", "")
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

    elif model_name.lower() == "enformer":
        lora_weights_dir = "./enformer_lora_weights"
        model = EnformerModel(
            k_l=rank, k_c="full", device=device,
            n_tasks=5313, crop_len=320,
            final_act_func="softplus", final_pool_func=None,
        )
        model.eval()
        base_name = f"enformer_lora_lr{rank}_crfull"
        quant_path = os.path.join(lora_weights_dir, f"{base_name}.pth")
        state_dict = torch.load(quant_path, weights_only=True)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if ".lora." in key:
                continue
            new_key = new_key.replace("model.", "").replace("orig_layer.", "")
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

    elif model_name.lower() == "sei":
        if rank == "full":
            model = SeiWrapper(k=rank, projection=False, mode="sequence", device=device)
        else:
            model = sl.Sei_LLRA(k=rank, projection=False, mode="sequence", device=device)

    elif model_name.lower() == "ag":
        if rank == "full":
            _ag = AlphaGenomePT.from_pretrained("model_all_folds.safetensors")
            print("Loaded full AG model")
            # Wrap so ptflops can call forward(x) without organism_index.
            # organism_index=0 = human; adjust if needed.
            model = AGFullWrapper(_ag, organism_index=0)
        else:
            _ag = AlphaGenome(k=rank)
            state_dict = torch.load(
                f'{model_name}_lora_weights/{model_name}_lora_lr{rank}.pth',
                weights_only=True,
            )
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key
                if ".lora." in key:
                    continue
                new_key = new_key.replace("proj.weight", "proj.layer.weight")
                new_key = new_key.replace("linear_pair.weight", "linear_pair.layer.weight")
                new_key = new_key.replace("linear_pair.bias", "linear_pair.layer.bias")
                new_key = new_key.replace("linear_pos_features.weight", "linear_pos_features.layer.weight")
                new_key = new_key.replace("linear_pos_features.bias", "linear_pos_features.layer.bias")
                new_key = new_key.replace("linear_y_q.weight", "linear_y_q.layer.weight")
                new_key = new_key.replace("linear_y_k.weight", "linear_y_k.layer.weight")
                new_key = new_key.replace("linear_q.weight", "linear_q.layer.weight")
                new_key = new_key.replace("linear_k.weight", "linear_k.layer.weight")
                new_key = new_key.replace("linear_v.weight", "linear_v.layer.weight")
                new_key = new_key.replace("linear_v.bias", "linear_v.layer.bias")
                new_key = new_key.replace("linear1.weight", "linear1.layer.weight")
                new_key = new_key.replace("linear1.bias", "linear1.layer.bias")
                new_key = new_key.replace("linear2.weight", "linear2.layer.weight")
                new_key = new_key.replace("linear2.bias", "linear2.layer.bias")
                new_state_dict[new_key] = value
            _ag.load_state_dict(new_state_dict, strict=True)
            # Also wrap LoRA AG models so forward signature is uniform
            model = AGFullWrapper(_ag, organism_index=0)

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

    Returns MACs and number of parameters, or (None, None) on failure.
    """
    from ptflops import get_model_complexity_info

    # ptflops expects input shape without batch dimension
    if len(input_shape) == 3:
        input_res = (input_shape[1], input_shape[2])
    else:
        input_res = input_shape[1:]

    macs, params = get_model_complexity_info(
        model,
        input_res,
        as_strings=False,
        print_per_layer_stat=True,
        verbose=True,
    )

    return macs, params


# =============================================================================
# Main function
# =============================================================================
def calculate_macs_per_sequence(model_name: str):
    """
    Calculate MACs for different ranks using ptflops.

    Args:
        model_name: "borzoi", "enformer", "sei", or "ag"
    """
    ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, "full"]
    device = "cpu" #"cuda:1" if torch.cuda.is_available() else "cpu"

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

            if macs is None:
                print(f"\nRank {r}: MACs computation returned None — custom module hooks may be missing.")
                macs_by_rank[r] = 0
            else:
                macs_by_rank[r] = macs
                print(f"\nRank {r}: MACs = {macs:,}")

            del model
            if "cuda" in device:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing rank {r}: {e}")
            import traceback
            traceback.print_exc()
            macs_by_rank[r] = 0

    return macs_by_rank


def main():

    # for model_name in ["borzoi", "enformer", "sei"]:
    #     print(f"\n{'#'*60}")
    #     print(f"# Processing {model_name.upper()}")
    #     print(f"{'#'*60}")
    #
    #     macs = calculate_macs_per_sequence(model_name)
    #
    #     output_file = f"{model_name}_flops.tsv"
    #     with open(output_file, "w") as f:
    #         f.write("rank\tmacs\n")
    #         for k, v in macs.items():
    #             f.write(f"{k}\t{v}\n")
    #
    #     print(f"\nSaved to {output_file}")

    for model_name in ["ag"]:
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