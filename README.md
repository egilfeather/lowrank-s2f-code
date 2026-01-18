# Compressing DNA sequence-to-function models with low-rank linear layers

Code repository for reproducing figures in Compressing DNA sequence-to-function models with low-rank linear layers.

## Requirements

- Dependencies listed in `environment.yml`

```bash
conda env create -f environment.yml
```

## Data

- Data should be placed in `data/`

Data availability: `examples/download_data.sh`
- Promoter benchmark variants 
- Promoter finetuning variants
- ENCODE cCREs

Manually download cell type specific QTLs (Figure 4) from synapse: https://www.synapse.org/Synapse:syn59449898/files/

Process data:
- `examples/preocess_data.ipynb` - generate cCRE .bed file, generate .vcf files from .tsv files
- `liftover_utils.py` - convert hg19 variants to hg38 using [liftover](https://liftover.broadinstitute.org)

## Repository Structure

```
├── README.md
├── requirements.txt
├── sample_usage.ipynb
├── data/
│   ├── ...
├── examples/
│   ├── download_data.sh
│   ├── figure1.py
│   ├── figure2.py
│   ├── figure3.py
│   └── figure4.py
│   └── ...
├── figs/
│   └── [generated figures saved here]
├── resources/
│   └── helpers.py
└── sei_lora/
    ├── dataloaders
    ├── model
    └── module
```
## Model Creation
See [seimodel](https://github.com/kostkalab/seimodel) package for Full Sei model blocks.

See [sei_llra](https://github.com/kostkalab/seilora) package for Low-rank Sei model architecture, weight creation, and quantization.

Low-rank Borzoi and Enformer Model Architecture: `examples/borzoi_lora_arch_mha.py`

Notebook creating Borzoi and Enformer Model weights: `examples/get_grelu_model_low_rank.ipynb`

Quantization of Borzoi and Enformer: `examples/grelu_quantize.py`

## Figures

### Figure 1: Low-rank Sei, Borzoi, and Enformer

- `examples/figure1.ipynb` - Plots

#### 1A: Models Schematic

#### 1B: Model rank x parameters

Scripts:
- `examples/grelu_quantize.py` - calculates Enformer-LLRA and Borzoi-LLRA model sizes
- `examples/figure1.ipynb` - calculates Sei-LLRA model sizes

#### 1C: Model rank x MACs

Scripts:
- `examples/calculate_macs.py` — calculate MACs for each model

#### 1D: Model rank x parameters

Scripts:
- `examples/lora_compare_full_cCREs.py` - calculates correlation between full and low rank models

---

### Figure 2: Promoter eQTL effect-size prediction

- `examples/figure2.ipynb` - Plots

#### 2A: Full model auROCs

AuROC predictions for promoter variant effect from 7 datasets from [Jaganathan et al., 2025](https://www.science.org/doi/10.1126/science.ads7373).

Scripts:
- `examples/get_benchmarking_metrics_seilora.py` — prediction for the full Sei model
- `examples/get_benchmarking_metrics_lora_grelu.py` - prediction for the full Borzoi and Enformer models

Data:
- `examples/benchmark_pai_sota.tsv` - tsv file with predictions for ChromBPNet and PromoterAI

#### 2B-D: Low-rank model auROCs

Scripts:
- `examples/get_benchmarking_metrics_seilora.py` — prediction for the low-rank Sei model
- `examples/get_benchmarking_metrics_lora_grelu.py` - prediction for the low-rank Borzoi and Enformer models

Output:
- `examples/borzoi_fixed_benchmark_promoterai.tsv` - tsv file with predictions for Borzoi-LLRA
- `examples/enformer_fixed_benchmark_promoterai.tsv` - tsv file with predictions for Enformer-LLRA
- `examples/benchmark_pai_seilora.tsv` - tsv file with predictions for Sei-LLRA

---

### Figure 3: Model Quantization and Finetuning

- `examples/figure3.ipynb` - Plots

#### 3A,B: Quantized models inference time on CPU

Calculates inference time for all 3 model, full, LLRA quantized, and LLRA unquantized.

Scripts:
- `examples/timing_grelu_full_track_cpu.py` — time inference for the quantized models
- `examples/timing_grelu_full_track_cpu_noquant.py` — time inference for the un-quantized models


#### 3C: Inference time vs mean Over vs Under expression promoter variant auROC

Data already calculated in previus figures. Inference timing from 3A,B. Promoter variant auROC (from unquantized models) from 2B-D.


#### 3D: Correlation between Sei-LLRA quantized and unquantized asay predictions for cCREs

Scripts:
- `examples/lora_compare_full_cCREs_quant.py` — calculate correlation between Sei-LLRA quantized and unquantized predictions 


#### 3E,F: Finetuning and Linear Probing of Sei-LLRA

Linear Probing of the quantized Sei-LLRA model on z-scores for GTEx eQTL outliers (see variant set used for finetuning in [Jaganathan et al., 2025](https://www.science.org/doi/10.1126/science.ads7373).):

Scripts:
- `examples/train_model_torch_cpu.py` — Linear probing for Sei-LLRA 

Finetuning of the quantized Sei-LLRA model on z-scores for GTEx eQTL outliers:

Scripts:
- `examples/train_model_torch_cpu_finetune_head.py` — Finetuning of low-rank layers for Sei-LLRA 

Variant prediction from finetuning and linear probing:

Scripts:
- `examples/get_benchmarking_metrics_seilora_finetune.py` — Evalutaiton of linear probing and finetuning on QTL variants datasets


---

### Figure 4: Quantized Sei-LLRA prediction of cell-type specific variant effects

- `examples/figure4.ipynb` - Plots

#### 4A,B: Sei-LLRA predictions of signed and un-signed variant effect sizes


Scripts:
- `examples/get_benchmarking_metrics_seilora.py` — Predction of cell-type QTL variants datasets on quantized Sei-LLRA models and full Sei
- `examples/get_benchmarking_metrics_lora_grelu.py` — Predction of cell-type QTL variants datasets on full Enformer

Data:
- `data/caqtls.african.lcls.benchmarking.all.tsv` - tsv file with predictions for ChromBPNet and Enformer
- `data/caqtls.eu.lcls.benchmarking.all.tsv` - tsv file with predictions for ChromBPNet and Enformer
- `data/dsqtls.yoruba.lcls.benchmarking.all.tsv` - tsv file with predictions for ChromBPNet and Enformer
- `data/caqtls.microglia.benchmarking.all.tsv".tsv` - tsv file with predictions for ChromBPNet
- `data/caqtls.smc.benchmarking.all.tsv` - tsv file with predictions for ChromBPNet
- `data/bqtls.pu1.lcls.benchmarking.all.tsv` - tsv file with predictions for ChromBPNet

- Alphagenome and Borzoi scores from [Avsec et al., 2025](https://doi.org/10.1101/2025.06.25.661532)

#### 4C: Change in Pearson corerlation between full and low-rank Sei

- Uses the same predictions as 4A,B

#### 4D,E: Change in Pearson corerlation between full and low-rank Sei

- Uses the same predictions as 4A,B
- Uses CPU inference time from 3A,B

---

## Utilities

`sei_lora/` contains utilities used throughout.

### `sei_lora.dataloaders`

Data loading utilities for sequence-based genomic models.

#### Datasets

- **`SeqDataset`**: Base dataset class for loading genomic sequences from BED files with optional score labels. Supports train/val/test splits by chromosome, automatic one-hot encoding, and variable window sizes.

- **`VariantDataset`**: Specialized dataset for variant effect prediction. Loads variants from VCF-like files and generates both reference and alternate allele sequences centered on each variant.

#### DataLoaders

- **`SeqDataLoader`**: Custom DataLoader with support for limiting the number of samples per epoch via `n_samples` parameter.

- **`VariantDataLoader`**: DataLoader for variant prediction that handles paired (reference, alternate) sequences and filters invalid samples.

- **`EmbeddingDataLoader`**: DataLoader that automatically generates embeddings on-the-fly by passing sequences through a provided model.

- **`EmbeddingScoreDataLoader`**: Similar to `EmbeddingDataLoader`, but also returns model prediction scores alongside embeddings.

---

### `sei_lora.score`

Utilities for processing model outputs and computing variant effect scores.

#### Sequence Class Scores

- **`sc_hnorm_varianteffect`**: Computes histone-normalized variant effect scores by projecting reference and alternate chromatin profiles onto sequence class features. Returns reference projections, alternate projections, and their difference across 40 sequence classes.

- **`get_sequence_class_scores_and_max`**: Converts sequence class difference scores into a DataFrame with human-readable class names and identifies the sequence class with the maximum absolute effect (signed) for each variant.

#### Cell Type and Assay-Specific Scores

- **`get_celltype_assay_specific`**: Extracts variant effect scores for specific cell types and assays. Supports flexible matching with optional strict mode, and outputs matched chromatin profile names to a file.

- **`get_index`**: Helper function that returns indices of chromatin profiles matching specified cell types and assays. Supports special category terms:
  - `"TF"`: Matches transcription factor binding assays (excludes ATAC-seq, DNase, FAIRE, and histone marks)
  - `"Histones"`: Matches histone modification assays (H1-H5 prefixes)




## Contact

Elizabeth Gilfeather - ejg66 @ pitt.edu

Dennis Kostka - kostka @ pitt.edu

## License

