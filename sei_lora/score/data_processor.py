
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def sc_hnorm_varianteffect(chromatin_profile_ref, chromatin_profile_alt, clustervfeat, histone_inds):
    chromatin_profile_ref_adjust = chromatin_profile_ref.copy()
    chromatin_profile_ref_adjust[:, histone_inds] = chromatin_profile_ref_adjust[:, histone_inds] * (
        (np.sum(chromatin_profile_ref[:, histone_inds], axis=1)*0.5 +
         np.sum(chromatin_profile_alt[:, histone_inds], axis=1)*0.5) /
        np.sum(chromatin_profile_ref[:, histone_inds], axis=1))[:, None]

    chromatin_profile_alt_adjust = chromatin_profile_alt.copy()
    chromatin_profile_alt_adjust[:, histone_inds] = chromatin_profile_alt_adjust[:, histone_inds] * (
        (np.sum(chromatin_profile_ref[:, histone_inds], axis=1)*0.5 +
         np.sum(chromatin_profile_alt[:, histone_inds], axis=1)*0.5) /
        np.sum(chromatin_profile_alt[:, histone_inds], axis=1))[:, None]

    refproj = np.dot(chromatin_profile_ref_adjust, clustervfeat.T) / np.linalg.norm(clustervfeat, axis=1)
    altproj = np.dot(chromatin_profile_alt_adjust, clustervfeat.T) / np.linalg.norm(clustervfeat, axis=1)
    diffproj = altproj[:, :40] - refproj[:, :40]
    return refproj[:, :40], altproj[:, :40], diffproj



def get_sequence_class_scores_and_max(sc_diff):
    seqclass_path = os.path.join(BASE_DIR, "../../resources/seqclass.names")
    with open(seqclass_path, "r") as f:
        sc_names = []
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                sc_names.append("-".join(parts[1:]))
            else:
                sc_names.append(parts[0])

    df = pd.DataFrame(sc_diff[:, :40], columns=sc_names[:40])

    # Step 1: absolute values
    abs_vals = df.abs()

    # Step 2: index of max abs value per row
    idx_max = abs_vals.values.argmax(axis=1)

    # Step 3: extract signed values at those indices
    row_idx = np.arange(len(df))
    signed_max_vals = df.values[row_idx, idx_max]

    # Step 4: assign to new column
    df["signed_max"] = signed_max_vals
    return df

def get_celltype_asssy_specific(chromatin_diff, celltypes = None, assays = None, strict = False):
    indices, matched_lines = get_index(celltypes, assays, subset_strict = strict)

    if len(indices) == 0:
        print(f"{'_'.join(celltypes)}_{'_'.join(assays)} had 0 matches.")
        return  # Skip if no match

    subset_vals = chromatin_diff[:, indices]
    signed_mean = subset_vals.mean(axis=1)

    # Create a clean name for the .obs column
    name = f"{'_'.join(celltypes)}_{'_'.join(assays)}"
    # Optional: write matched lines to a .txt file per subset
    with open(f"{name}_chromatin_profiles.txt", "w") as f:
        for line in matched_lines:
            f.write(line + "\n")
    return signed_mean

def get_index(celltypes, assays, subset_strict = False):
    """
    Return indices and matching chromatin_names for given celltypes and assays.
    - 'TF': assay must NOT be accessibility assay (ATAC/DNase/FAIRE) and NOT start with H1-H5
    - 'Histones': assay must start with H1–H5
    - Other terms are matched literally, or strictly if self.subset_strict is True
    """
    def get_targets(filename):
        with open(filename) as f:
            return [line.strip() for line in f]

    target_path = os.path.join(BASE_DIR, "../../resources/target.names")    
    chromatin_names = get_targets(target_path)
    exclude_assays = {'ATAC-seq', 'DNase', 'FAIRE'}
    histone_prefixes = ('H1', 'H2', 'H3', 'H4', 'H5')

    indices = []
    matched_lines = []

    # Define category matchers
    def is_tf(assay):
        return assay not in exclude_assays and not assay.startswith(histone_prefixes)

    def is_histone(assay):
        return assay.startswith(histone_prefixes)

    category_matchers = {
        "TF": is_tf,
        "Histones": is_histone
    }

    for i, name in enumerate(chromatin_names):
        parts = name.strip().split('|')
        if len(parts) < 2:
            continue

        cell, assay = parts[0].strip(), parts[1].strip()

        # Match celltypes
        if subset_strict:
            if cell not in celltypes:
                continue
        else:
            if not any(ct in cell for ct in celltypes):
                continue

        # Match assays
        match = False
        for term in assays:
            if term in category_matchers:
                if category_matchers[term](assay):
                    match = True
                    break
            else:
                if subset_strict:
                    if assay == term:
                        match = True
                        break
                else:
                    if term in assay:
                        match = True
                        break

        if match:
            indices.append(i)
            matched_lines.append(name)

    return indices, matched_lines





