# DNA Shift-And -- Data Provenance

## Source

Human genome reference sequence (GRCh38/hg38) chromosome 1 from UCSC
Genome Browser.

## Download

```bash
wget "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr1.fa.gz"  # ~67 MB, ~249 Mbp
gunzip chr1.fa.gz
```

## Conversion

`convert.py` parses FASTA files, strips headers and newlines, maps each
nucleotide to 4 Parabix-style basis bitstreams {bA, bC, bG, bT}, and
handles N (unknown) bases by setting none of the basis bits.

## Datasets

- **`small/hg38_chr1_2m.npz`** (979 KB) -- first 2M bp of chr1.
- **`medium/hg38_chr1_20m.npz`** (9.6 MB) -- first 20M bp of chr1.
- **`large/hg38_chr1_200m.npz`** (96 MB) -- first 200M bp of chr1.
