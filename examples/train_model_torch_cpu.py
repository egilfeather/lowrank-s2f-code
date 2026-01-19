#!/usr/bin/env python 
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from sei_lora.dataloaders import VariantDataset, SeqDataLoader, SeqDataset, EmbeddingDataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sei_lora.utils.loss_functions import FocalLoss, WeightedBCELoss

import seimodel as sm
import seillra as sl
from torch.utils.data import DataLoader, Dataset


## to lauch tensorboard: tensorboard --logdir=logs/



class SeiLoraHeadWrapper(nn.Module):
    def __init__(self, k: int, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.head = sl.get_sei_head_llra(k)

    def forward(self, x):
        """
        Forward pass: computes output for both original and reversed input
        and averages the results. This is fed into the projector.
        """
        for_x, rev_x = x
        for_y = self.head(for_x)
        rev_y = self.head(rev_x)

        out = (for_y + rev_y) / 2

        return out

class seiSmallPhead(torch.nn.Module):
    def __init__(self):
        super(seiSmallPhead, self).__init__()
        self.l1 = torch.nn.Linear(21907, 64)
        # self.l2 = torch.nn.Linear(64, 64)
        self.l2= torch.nn.Linear(64, 1)

    def forward(self, x):
        x1, x2 = x
        x1 = torch.nn.functional.gelu(self.l1(x1))
        x2 = torch.nn.functional.gelu(self.l1(x2))
        x = x2 - x1
        # x = torch.nn.functional.gelu(self.l2(x))
        x = self.l2(x)
        return x
    
def get_loss(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()   
    errs_l2 = torch.nn.functional.mse_loss(y_pred, y_true, reduction="none")
    # errs_l1 = torch.abs(y_pred - y_true)
    # errs = torch.minimum(errs_l2, errs_l1)
    errs = errs_l2
    # max_err_idx = torch.argmax(errs)  # - drop largest error
    # errs[max_err_idx] = 0
    return errs.mean()


def main(rank=256):
    if not os.path.exists(f"checkpoint_{rank}"):
        os.makedirs(f"checkpoint_{rank}")
    dev =  "cpu" #'cuda:1'
    model = sl.Sei_LLRA(k=rank, projection = False, mode = "variant", device = dev)
    model.eval()
    vcf = "../data/finetune_gtex_filtered.tsv"
    train_dataset = SeqDataset(file_path=vcf,fasta_path="../../sei-framework-main/resources/hg38_UCSC.fa",
            mode = "train", val_chrom =["chr10", "chr21", "chr22"], test_chrom = ["chr8", "chr9"], variant_train=True)
    
    train_dataloader = EmbeddingDataLoader(dataset=train_dataset, model = model, batch_size=16, shuffle=True, num_workers=7, n_samples= 3200, device = dev)
    val_dataset = SeqDataset(file_path=vcf,fasta_path="../../sei-framework-main/resources/hg38_UCSC.fa",
            mode = "val", val_chrom =["chr10", "chr21", "chr22"], test_chrom = ["chr8", "chr9"], variant_train=True)
    
    val_dataloader = EmbeddingDataLoader(dataset=val_dataset, model = model, batch_size=16, shuffle=False, num_workers=7, n_samples= 1600, device = dev)
    

    print("Data Loaded")
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


    ft_model = seiSmallPhead()
    ft_model = ft_model.to(dev)
    ft_model.train()
    # model.train()

    optimizer = torch.optim.AdamW((p for p in ft_model.parameters() if p.requires_grad), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=4, verbose=True, cooldown = 4)

    writer = SummaryWriter(log_dir=f"logs/torch_model_{rank}_pai_probe_v1")

    best_val_loss = float("inf")
    patience = 8
    patience_counter = 0

    num_epochs = 100

    # === Training Loop ===
    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch+1}")
        ft_model.train()
        # model.train()
        running_loss = 0.0
        # true_sums = 0.0
        # pred_sums = 0.0
        # num_samples = 0.0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        trainable_params = sum(p.numel() for p in ft_model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in ft_model.parameters() if not p.requires_grad)

        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        for batch in progress_bar:
            x, y, v = batch
            # print(y.shape)
            # print(y[:5, :5])
            x, y= x.to(dev), y.to(dev)
            z = v[:, -1].astype(np.float32)  
            y_true = torch.tensor(z, device=dev, dtype=torch.float32)
            optimizer.zero_grad()
            diff = ft_model((x, y))
            #print(diff)
            
            # print(outputs.shape)
            # print(outputs[:5, :5])
            loss = get_loss(y_true, diff)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"train_loss": loss.item()})
            # true_sums  += v.sum(axis = 1).sum().item()
            # pred_sums += outputs.sum(axis=1).sum().item()
            # num_samples += v.size(0)

        avg_train_loss = running_loss / len(train_dataloader)
        # avg_true_sums = true_sums / num_samples
        # avg_pred_sums = pred_sums / num_samples
        writer.add_scalar("train_loss", avg_train_loss, epoch)
        # writer.add_scalar("train_true_sums", avg_true_sums, epoch)
        # writer.add_scalar("train_preds_sums", avg_pred_sums, epoch)

        # === Validation ===
        ft_model.eval()
        val_loss = 0.0
        # val_true_sums = 0.0
        # val_pred_sums = 0.0
        # val_num_samples = 0.0
        with torch.no_grad():
            for x, y, v in val_dataloader:
                x, y= x.to(dev), y.to(dev)
                z = v[:, -1].astype(np.float32)  
                y_true = torch.tensor(z, device=dev, dtype=torch.float32)
                optimizer.zero_grad()
                diff = ft_model((x, y))
                # alt = ft_model(y)
                # diff = alt -ref
                loss = get_loss(y_true, diff)
                val_loss += loss.item()
                


        avg_val_loss = val_loss / len(val_dataloader)
        # avg_val_true_sums = val_true_sums/ val_num_samples
        # avg_val_pred_sums = val_pred_sums/ val_num_samples
        writer.add_scalar("val_loss", avg_val_loss, epoch)
        # writer.add_scalar("val_true_sums", avg_val_true_sums, epoch)
        # writer.add_scalar("val_preds_sums", avg_val_pred_sums, epoch)

        print(f"Epoch {epoch+1}, train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        # === Early Stopping and Checkpoint ===
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(ft_model.state_dict(), f"checkpoint_{rank}/best_model_pai_probe_v1.pt")
            print(f"Saved best checkpoint at epoch {epoch+1}")
        else:
            if epoch > 5:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

    writer.close()
    torch.cuda.empty_cache()
    print("Finished training.")

if __name__ == '__main__':
    main(rank=1)
    #main(rank=2)
    # main(rank=4)
    #main(rank=8)
    # main(rank=16)
    #main(rank=32)
    # main(rank=64)
    # main(rank=128)
    main(rank=256)
    # main(rank=512)
    #main(rank=1024)
    #main(rank=2048)