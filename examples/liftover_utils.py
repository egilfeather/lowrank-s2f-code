#!/usr/bin/env python 
import os
import pandas as pd


def sanitize_filename(filename, tag_to_remove="hg19"):
    """
    Remove a tag (like 'hg19') from the filename before the extension using only string methods.
    """
    parts = filename.split('.')
    cleaned_parts = [part for part in parts if part != tag_to_remove]
    return '.'.join(cleaned_parts)


def replace_vcf_and_tsv_coords_from_bed(bed_file):
    """
    Given only a BED file, automatically locate the matching VCF and TSV files in the same directory,
    apply coordinate updates, and save cleaned versions without 'hg19' in their names.
    """
    bed_dir = os.path.dirname(bed_file)
    bed_base = os.path.basename(bed_file)
    base_prefix = bed_base.rsplit('.', 1)[0].replace('.hg38', '')  # remove .hg38 if present

    # Infer file paths
    vcf_file = os.path.join(bed_dir, f"{base_prefix}.hg19.vcf")
    tsv_file = os.path.join(bed_dir, f"{base_prefix}.hg19.tsv")

    # Construct output filenames (same dir, no .hg19)
    output_file_vcf = os.path.join(bed_dir, sanitize_filename(os.path.basename(vcf_file)))
    output_file_tsv = os.path.join(bed_dir, sanitize_filename(os.path.basename(tsv_file)))

    # Load BED
    bed_cols = ['chr_bed', 'start', 'end', 'snp_id', 'extra']
    bed = pd.read_csv(bed_file, sep='\t', header=None, names=bed_cols)

    # Extract original coords from snp_id
    parsed = bed['snp_id'].str.extract(r'(?P<old_chr>[^:]+):(?P<old_pos>\d+)-')
    parsed['old_pos'] = parsed['old_pos'].astype(int)
    bed = pd.concat([bed, parsed], axis=1)

    # Process VCF
    vcf = pd.read_csv(vcf_file, sep='\t', comment='#', header=None)
    merged_vcf = pd.merge(vcf, bed, how='inner', left_on=[0, 1], right_on=['old_chr', 'old_pos'])
    merged_vcf[0] = merged_vcf['chr_bed']
    merged_vcf[1] = merged_vcf['end']
    out_vcf = merged_vcf.drop(columns=bed_cols + ['old_chr', 'old_pos'])
    out_vcf.to_csv(output_file_vcf, sep='\t', index=False, header=False)

    # Process TSV
    tsv = pd.read_csv(tsv_file, sep='\t', comment='#', header=0)
    merged_tsv = pd.merge(tsv, bed, how='inner', left_on=["var.chr", "var.pos_hg19"], right_on=['old_chr', 'old_pos'])
    merged_tsv["var.chr"] = merged_tsv['chr_bed']
    merged_tsv["var.pos_hg38"] = merged_tsv['end']
    out_tsv = merged_tsv.drop(columns=bed_cols + ['old_chr', 'old_pos', 'var.pos_hg19'])
    out_tsv.to_csv(output_file_tsv, sep='\t', index=False, header=True)

    return out_vcf


def main():
    # Example usage:
    # replace_vcf_and_tsv_coords_from_bed(
    #     "../full_variants/bqtls.pu1.lcls.benchmarking.all.bed"
    # )
    replace_vcf_and_tsv_coords_from_bed(
        "../full_variants/dsqtls.yoruba.lcls.benchmarking.all.hg38.bed"
    )

if __name__ == '__main__':
    main()