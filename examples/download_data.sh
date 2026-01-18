#!/usr/bin/env bash
set -e

mkdir -p ../data

wget -P ../data \
https://raw.githubusercontent.com/Illumina/PromoterAI/master/data/annotation/finetune_gtex.tsv

wget -P ../data \
https://raw.githubusercontent.com/Illumina/PromoterAI/master/data/benchmark/CAGI5_saturation.tsv \
https://raw.githubusercontent.com/Illumina/PromoterAI/master/data/benchmark/GTEx_outlier.tsv \
https://raw.githubusercontent.com/Illumina/PromoterAI/master/data/benchmark/GTEx_eQTL.tsv \
https://raw.githubusercontent.com/Illumina/PromoterAI/master/data/benchmark/MPRA_eQTL.tsv \
https://raw.githubusercontent.com/Illumina/PromoterAI/master/data/benchmark/MPRA_saturation.tsv \
https://raw.githubusercontent.com/Illumina/PromoterAI/master/data/benchmark/UKBB_proteome.tsv \
https://raw.githubusercontent.com/Illumina/PromoterAI/master/data/benchmark/GEL_RNA.tsv

wget -P ../examples \
https://screen.encodeproject.org/index/cversions/GRCH38-cCRE.bed

wget https://sei-files.s3.amazonaws.com/resources.tar.gz
tar -xzvf resources.tar.gz

wget -P ../resources \
https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.basic.annotation.gtf.gz



# install cell-type specific QTLs from synapse: https://www.synapse.org/Synapse:syn59449898/files/