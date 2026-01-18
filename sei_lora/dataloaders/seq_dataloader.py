
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader, Subset
import h5py
import os

from Bio import SeqIO
from Bio.Seq import Seq
from pybedtools import BedTool
from itertools import islice
import math

LOOKUP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, scores_path = "", fasta_path = "", mode = "variant_prediction", val_chrom = None, test_chrom = None, window_size = 4096, variant_train = False):
        """
        Args:
            bed_path (str): Path to the BED file with positions.
            scores_path (str): Path to the file with 61-dimensional scores.
            fasta_path (str): Path to the FASTA file for sequence retrieval.
        """
        self.mode = mode
        self.variant_train = variant_train
        self.fasta_path = fasta_path

        self.window_size = window_size
        self.genome = self._load_fasta(fasta_path)  # Preload FASTA for efficiency
        if self.mode == "variant_prediction":
            self.vcf_positions = pd.read_csv(file_path, comment="#", sep="\t", header = None)
            self.vcf_positions.columns = ["CHROM", "POS", "STRAND", "REF", "ALT" ]
            self.vcf_positions["CHROM"] = self.vcf_positions["CHROM"].apply(
                lambda x: f"chr{x}" if not str(x).startswith("chr") else x
            )
        elif self.variant_train == True:
            self.vcf_positions = pd.read_csv(file_path, comment="#", sep="\t", header = 0)
            self.vcf_positions.columns = ["CHROM", "POS", "REF", "ALT", "_", "_", "_", "STRAND", "_", "_", "_", "_", "p_under", "p_over", "z" ]
            self.vcf_positions["CHROM"] = self.vcf_positions["CHROM"].apply(
                lambda x: f"chr{x}" if not str(x).startswith("chr") else x
            )
            
            self.vcf_positions = self.vcf_positions[["CHROM", "POS", "STRAND", "REF", "ALT", "p_under", "p_over", "z"]]
            
            if val_chrom is None:
                val_chrom = []
            elif isinstance(val_chrom, str):
                val_chrom = [val_chrom]
            if test_chrom is None:
                test_chrom = []
            elif isinstance(test_chrom, str):
                test_chrom = [test_chrom]
       
            if self.mode == "train":
                mask = ~self.vcf_positions['CHROM'].isin(val_chrom + test_chrom)
            elif self.mode == "val":
                mask = self.vcf_positions['CHROM'].isin(val_chrom)
            elif self.mode == "test":
                mask = self.vcf_positions['CHROM'].isin(test_chrom)
            else:
                mask = np.ones(len(self.vcf_positions), dtype=bool)

            # Apply mask to bed_positions and scores
            self.vcf_positions = self.vcf_positions[mask].reset_index(drop=True)
            mask = (self.vcf_positions["p_over"] < 0.05) | (self.vcf_positions["p_under"] < 0.05)
            self.vcf_positions = self.vcf_positions[mask].reset_index(drop=True)


        else:
            if isinstance(file_path, pd.DataFrame):
                # file_path is already a DataFrame
                bed_df = file_path.copy()
                # Ensure it has the correct columns
                expected_cols = ["chrom", "start", "end", "name", "score", "strand"]
                if list(bed_df.columns[:6]) != expected_cols:
                    bed_df.columns = expected_cols

            else:
            
                if file_path.split(".")[-1] == "bed":
                    bed_df = BedTool(file_path).to_dataframe(names=["chrom", "start", "end", "name", "score", "strand"])
                else:
                    raise ValueError(f"Unsupported file format: {file_path.split('.')[-1]}. Expected a .bed file.")

            if scores_path:
                ext = os.path.splitext(scores_path)[1]
                if ext == ".npy":
                    scores = np.load(scores_path)
                elif ext in [".h5", ".hdf5"]:
                    with h5py.File(scores_path, "r") as hf:
                        for name in hf:
                            if isinstance(hf[name], h5py.Dataset) and len(hf[name].shape) == 2:
                                scores = hf[name][:]
                                break
                        else:
                            raise ValueError("No 2D dataset found in HDF5 file.")
                else:
                    raise ValueError(f"Unsupported file format for scores_path: {ext}")
            else:
                scores = None
            if val_chrom is None:
                val_chrom = []
            elif isinstance(val_chrom, str):
                val_chrom = [val_chrom]
            if test_chrom is None:
                test_chrom = []
            elif isinstance(test_chrom, str):
                test_chrom = [test_chrom]
       
            if self.mode == "train":
                mask = ~bed_df['chrom'].isin(val_chrom + test_chrom)
            elif self.mode == "val":
                mask = bed_df['chrom'].isin(val_chrom)
            elif self.mode == "test":
                mask = bed_df['chrom'].isin(test_chrom)
            else:
                mask = np.ones(len(bed_df), dtype=bool)

            # Apply mask to bed_positions and scores
            self.bed_positions = bed_df[mask].reset_index(drop=True)
            self.scores = scores[mask.values] if scores is not None else None




    def __len__(self):
        if self.mode == "variant_prediction":
            return len(self.vcf_positions)
        elif self.mode == "variant_train":
            return len(self.vcf_positions)
        else:
            if self.variant_train == True:
                return len(self.vcf_positions)
            else:
                return len(self.bed_positions)

    def __getitem__(self, index):
        # Retrieve BED position and corresponding score
        if self.mode == "variant_prediction":
            # print("+++++++++++++++++++++++++++++++++++++++++++++")
            vcf_row = self.vcf_positions.iloc[index]
            ##TODO: Assuming that pos is the center of ref even when there is variable length
            chrom, pos, strand, ref, alt = vcf_row
            if strand == "-":
                ref = str(Seq(ref).reverse_complement())
                alt = str(Seq(alt).reverse_complement())

            center = pos + (len(ref) // 2)
            start = center - (self.window_size // 2) - 1
            end = start + self.window_size

            sequence = self._get_sequence(chrom, start, end)
            # Find where REF should be
            ref_start = (self.window_size // 2) - (len(ref) // 2)
            ref_end = ref_start + len(ref)
            ref_seq_segment = sequence[ref_start:ref_end]

            # Skip if reference doesn't match
            if ref_seq_segment != ref:
                if ref_seq_segment != alt:
                    print(f"Failed to insert allele at {chrom}:{pos}:{ref}:{alt}:{strand}")
                    print(f"Center bp: {sequence[2047]}")
                    print(f"Center 3bp: {sequence[2046:2049]}")
                    return None, None, None
                else:
                    temp = sequence
                    sequence = temp[:ref_start] + ref + temp[ref_end:]
                    # temp = (ref, alt)
                    # ref = temp[1]
                    # alt = temp[0]
            alt_sequence = sequence[:ref_start] + alt + sequence[ref_end:]
            # print(len(sequence))
            if len(alt_sequence) < self.window_size:
                alt_sequence += "N" * (self.window_size - len(alt_sequence))
            elif len(alt_sequence) > self.window_size:
                alt_sequence = alt_sequence[:self.window_size]

            row = vcf_row.to_numpy().tolist()
            return torch.tensor(self.returnonehot(sequence, index=index), dtype=torch.float32), torch.tensor(self.returnonehot(alt_sequence, index=index), dtype=torch.float32), row
        elif self.variant_train == True:
            # print("+++++++++++++++++++++++++++++++++++++++++++++")
            vcf_row = self.vcf_positions.iloc[index]
            ##TODO: Assuming that pos is the center of ref even when there is variable length
            chrom, pos, strand, ref, alt, _, _, _ = vcf_row
            if strand == "-":
                ref = str(Seq(ref).reverse_complement())
                alt = str(Seq(alt).reverse_complement())

            center = pos + (len(ref) // 2)
            start = center - (self.window_size // 2) - 1
            end = start + self.window_size

            sequence = self._get_sequence(chrom, start, end)
            # Find where REF should be
            ref_start = (self.window_size // 2) - (len(ref) // 2)
            ref_end = ref_start + len(ref)
            ref_seq_segment = sequence[ref_start:ref_end]

            # Skip if reference doesn't match
            if ref_seq_segment != ref:
                if ref_seq_segment != alt:
                    print(f"Failed to insert allele at {chrom}:{pos}:{ref}:{alt}:{strand}")
                    print(f"Center bp: {sequence[2047]}")
                    print(f"Center 3bp: {sequence[2046:2049]}")
                    return None, None, None
                else:
                    temp = sequence
                    sequence = temp[:ref_start] + ref + temp[ref_end:]
                    # temp = (ref, alt)
                    # ref = temp[1]
                    # alt = temp[0]
            alt_sequence = sequence[:ref_start] + alt + sequence[ref_end:]
            # print(len(sequence))
            if len(alt_sequence) < self.window_size:
                alt_sequence += "N" * (self.window_size - len(alt_sequence))
            elif len(alt_sequence) > self.window_size:
                alt_sequence = alt_sequence[:self.window_size]

            row = vcf_row.to_numpy().tolist()
            return torch.tensor(self.returnonehot(sequence, index=index), dtype=torch.float32), torch.tensor(self.returnonehot(alt_sequence, index=index), dtype=torch.float32), row
        else:
            bed_row = self.bed_positions.iloc[index]
            chrom, start, end, strand, score, _= bed_row
            if end-start != self.window_size:
                center = (start + end)//2
                start = center - (self.window_size // 2)
                end = start + self.window_size

            # Convert BED position to sequence
            sequence = self._get_sequence(chrom, start, end, strand)
        
            one_hot_sequence = self.returnonehot(sequence, index=index)

            # Retrieve scores
            if self.scores is not None:
                scores = self.scores[index].astype(np.float32)
                return torch.tensor(one_hot_sequence, dtype=torch.float32), torch.tensor(scores, dtype=torch.float32)
            else:
                return torch.tensor(one_hot_sequence, dtype=torch.float32), torch.zeros(1, 61)

    def _load_fasta(self, fasta_path):
        """
        Load a FASTA file into a dictionary for fast sequence retrieval.
        """
        genome = {}
        for record in SeqIO.parse(fasta_path, "fasta"):
            genome[record.id] = str(record.seq)
        return genome

    def _get_sequence(self, chrom, start, end, strand = "+"):
        """
        Retrieve a sequence from the FASTA dictionary based on chromosome and positions.
        """
        seq = []
        # Left pad if start < 0
        if start < 0:
            seq.append("N" * (-start))
            start = 0
        # Middle
        if start < len(self.genome[chrom]):
            seq.append(self.genome[chrom][start:min(end, len(self.genome[chrom]))])
        # Right pad if end past chrom length
        if end > len(self.genome[chrom]):
            seq.append("N" * (end - len(self.genome[chrom])))
        return "".join(seq).upper()
    
    def _insert_allele(self, chrom, start, end, ref, alt, strand = "+"):
        """
        Replace the REF allele at the center of the sequence with ALT.
        
        Args:
            sequence (str): the reference sequence window
            ref (str): the reference allele (from VCF)
            alt (str): the alternate allele (from VCF)

        Returns:
            str: the modified sequence with the allele inserted
        """
        for char in ref:
            if char not in LOOKUP:
                return None
        for char in alt:
            if char not in LOOKUP:
                return None
        sequence = self._get_sequence(chrom, start, end, strand)
        sequence = sequence.upper()
        center_idx = self.window_size//2
        ref_len = len(ref)
        alt_len = len(alt)
        diff = ref_len - alt_len

        # Get the slice around the center that should match REF
        start_idx = center_idx -1 - ref_len//2
        end_idx = start_idx + (ref_len +1)//2

        ref_seq_segment = sequence[start_idx:end_idx]

        if ref_seq_segment != ref:
            return None
        # else:
        #     print(f"Matched! Expected '{ref}', found '{ref_seq_segment}' at center {start_idx}:{end_idx}")
        # Build the new sequence with ALT inserted
        new_sequence = sequence[:start_idx] + alt + sequence[end_idx:]
        if diff >=0:
            new_sequence = self._get_sequence(chrom, start-(diff//2), start, strand) + new_sequence + self._get_sequence(chrom, end, end+((diff+1)//2), strand)
        else:
            new_sequence = new_sequence[(-diff//2):-(-diff+1)//2]

        return new_sequence.upper()  

    def returnonehot(self, string, index = None):
        """
        One-hot encode a DNA sequence.
        
        Args:
            string (str): DNA sequence.
        
        Returns:
            np.ndarray: One-hot encoded matrix of shape (4, len(sequence)).
        """
        string = string.upper()
        lookup = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        tmp = np.array(list(string))
        icol = np.where(tmp != 'N')[0]
        out = np.zeros((4, len(tmp)), dtype=np.float32)
        irow = np.array([lookup[i] for i in tmp[icol]])

        if len(icol) > 0:
            out[irow, icol] = 1


        return out



class VariantDataset(SeqDataset):
    def __init__(self, file_path, fasta_path = "../resources/hg38_UCSC.fa", window_size = 4096):
        super().__init__(file_path = file_path, scores_path = "", fasta_path = fasta_path, mode = "variant_prediction", val_chrom = None, test_chrom = None, window_size = window_size)
    

class SeqDataLoader(DataLoader):
    def __init__(self, dataset, *, batch_size=1, n_samples=None, **kwargs):
        """
        Custom DataLoader that limits the number of batches per epoch.

        Args:
            n_batches (int): Maximum number of batches to yield per epoch.
        """
        self.user_n_samples = n_samples  # Save the user's request
        self.batch_size_here = batch_size
        print(self.batch_size_here)
        self.__pl_cls_kwargs__ = {
            "dataset": dataset,
            "batch_size": batch_size,
            "n_samples": n_samples,
            **kwargs
        }
        if getattr(dataset, "mode", None) == "variant_prediction":
            kwargs["collate_fn"] = safe_collate

        super().__init__(dataset=dataset, batch_size=batch_size, **kwargs)

        dataset_size = len(self.dataset)
        print("Batch size after DataLoader:", self.batch_size_here)
        print("Batch size after DataLoader:", self.batch_size)
        print(dataset_size)
        # print(batch_size)
        max_batches = math.ceil(int(dataset_size) / int(self.batch_size_here))
        print(max_batches)
        if self.user_n_samples is not None:
            user_n_batches = math.ceil(self.user_n_samples / self.batch_size_here)
            self._effective_batches = min(user_n_batches, max_batches)
        else:
            self._effective_batches = max_batches
        print(self._effective_batches )

    def __iter__(self):
        base_iter = super().__iter__()
        return islice(base_iter, self._effective_batches)
        
    def __len__(self):
            return self._effective_batches 

def safe_collate(batch):
    # batch is a list of (x, y, v)
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        print("Not Batch")
        return None  # whole batch is invalid
  
    xs, ys, vs = zip(*batch)
    return torch.stack(xs),torch.stack(ys), np.stack(vs)

class VariantDataLoader(SeqDataLoader):
    def __init__(self, dataset, *, batch_size=1, n_samples = None, **kwargs):
        """
        Custom DataLoader that limits the number of batches per epoch.

        Args:
            n_batches (int): Maximum number of batches to yield per epoch.
        """

        kwargs.pop("collate_fn", None)
        super().__init__(dataset = dataset, batch_size=batch_size, n_samples = n_samples, collate_fn = safe_collate, **kwargs)





class EmbeddingDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, model, *, batch_size=1, n_samples=None, device="cuda", **kwargs):
        """
        A DataLoader that automatically converts one-hot sequences into embeddings
        using a pretrained model, on the fly.
        """
        self.model = model.eval().to(device)
        self.device = device
        self.user_n_samples = n_samples
        self.batch_size_here = batch_size
        self._base_kwargs = kwargs

        # use embedding-aware collate fn
        kwargs["collate_fn"] = self.make_embedding_collate_fn()

        super().__init__(dataset=dataset, batch_size=batch_size, **kwargs)

        dataset_size = len(dataset)
        max_batches = math.ceil(dataset_size / batch_size)
        if n_samples is not None:
            user_batches = math.ceil(n_samples / batch_size)
            self._effective_batches = min(user_batches, max_batches)
        else:
            self._effective_batches = max_batches

    def make_embedding_collate_fn(self):
        model = self.model
        device = self.device

        @torch.no_grad()
        def collate_fn(batch):
            """
            Takes batch of (onehot, scores) or (ref_onehot, alt_onehot, row)
            and runs them through model to get embeddings.
            """
            # Filter out invalid samples (None)
            batch = [b for b in batch if b[0] is not None]
            if not batch:
                return None

            # Determine variant or standard
            if len(batch[0]) == 2:
                # standard dataset (onehot, score)
                xs, ys = zip(*batch)
                xs = torch.stack(xs).to(device)  # [B, 4, L]
                embeddings = model(xs)  # → [B, D] or [B, D, L]
                embeddings = embeddings.detach().cpu()
                return embeddings, torch.stack(ys)
            else:
                # variant mode (ref_onehot, alt_onehot, meta)
                ref_xs, alt_xs, metas = zip(*batch)
                ref_xs = torch.stack(ref_xs).to(device)
                alt_xs = torch.stack(alt_xs).to(device)
                ref_emb, alt_emb = model((ref_xs, alt_xs))
                ref_emb, alt_emb = ref_emb.detach().cpu(), alt_emb.detach().cpu()
                return ref_emb, alt_emb, np.stack(metas)

        return collate_fn

    def __iter__(self):
        base_iter = super().__iter__()
        return islice(base_iter, self._effective_batches)

    def __len__(self):
        return self._effective_batches


class EmbeddingScoreDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, model, *, batch_size=1, n_samples=None, device="cuda", **kwargs):
        """
        A DataLoader that automatically converts one-hot sequences into embeddings
        using a pretrained model, on the fly.
        """
        self.model = model.eval().to(device)
        self.device = device
        self.user_n_samples = n_samples
        self.batch_size_here = batch_size
        self._base_kwargs = kwargs

        # use embedding-aware collate fn
        kwargs["collate_fn"] = self.make_embedding_collate_fn()

        super().__init__(dataset=dataset, batch_size=batch_size, **kwargs)

        dataset_size = len(dataset)
        max_batches = math.ceil(dataset_size / batch_size)
        if n_samples is not None:
            user_batches = math.ceil(n_samples / batch_size)
            self._effective_batches = min(user_batches, max_batches)
        else:
            self._effective_batches = max_batches

    def make_embedding_collate_fn(self):
        model = self.model
        device = self.device

        @torch.no_grad()
        def collate_fn(batch):
            """
            Takes batch of (onehot, scores) or (ref_onehot, alt_onehot, row)
            and runs them through model to get embeddings.
            """
            # Filter out invalid samples (None)
            batch = [b for b in batch if b[0] is not None]
            if not batch:
                return None

            # Determine variant or standard
            if len(batch[0]) == 2:
                # standard dataset (onehot, score)
                xs, ys = zip(*batch)
                xs = torch.stack(xs).to(device)  # [B, 4, L]
                embeddings, score = model(xs)  # → [B, D] or [B, D, L]
                embeddings , score = embeddings.detach().cpu(), score.detatch().cpu()
                return embeddings, score, torch.stack(ys)
            else:
                # variant mode (ref_onehot, alt_onehot, meta)
                ref_xs, alt_xs, metas = zip(*batch)
                ref_xs = torch.stack(ref_xs).to(device)
                alt_xs = torch.stack(alt_xs).to(device)
                ref_emb, alt_emb, ref_score = model((ref_xs, alt_xs))
                ref_emb, alt_emb, ref_score = ref_emb.detach().cpu(), alt_emb.detach().cpu(), ref_score.detach().cpu()
                return ref_emb, alt_emb, ref_score, np.stack(metas)

        return collate_fn

    def __iter__(self):
        base_iter = super().__iter__()
        return islice(base_iter, self._effective_batches)

    def __len__(self):
        return self._effective_batches
