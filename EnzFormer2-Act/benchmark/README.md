# Activity benchmark inputs

The `benchmark/` tree contains point-mutation inputs, consolidated source notes, manifests,
and shared evaluators for EnzFormer2-Act. The six processed datasets included
for local validation are CYP2C9, NDM1, PTEN, PjDHFR, RNase III, and VIM2.

## Layout

```text
benchmark/
├── enzymes/                       per-enzyme data and input metadata
├── shared/point_mutation/         runners, scoring, and metrics
└── shared/manifests/              processed benchmark input catalog
```

## Scope manifests

The original manuscript analysis organization was:

- backbone screening: ten datasets, including eCTL;
- hyperparameter optimization: eCTL only;
- main Act comparison: nine datasets—TmIGPS, SsIGPS, TtIGPS, NDM1, VIM2,
  PjDHFR, RNase III, CYP2C9, and PTEN—with seeds 13, 17, and 23.

The nine-dataset comparison excludes eCTL. A manifest may describe external
inputs without making them locally executable; the runner reports missing files
before loading a model.

## Manifest contract

The Act benchmark runner accepts manifests with `queries` and `runs`. A query
supplies a wild-type FASTA and truth CSV. A run supplies `query`, `method`, and
either `source: predictions_csv` with `predictions_csv`, or
`source: checkpoint_scan` with a local Act checkpoint. Paths are resolved from
the manifest directory, with repository-relative paths supported for manifests
stored in the repository. For raw ESM-C predictions, run
`python -m benchmark.shared.point_mutation.raw_esmc_baseline --help` and provide
the query FASTA, truth CSV and output path explicitly. Additional PLM screening
utilities and their manifests are not included.

Validate a benchmark-run manifest without loading a model:

```bash
enzformer2 act benchmark --help
enzformer2 act benchmark --manifest MANIFEST_PATH \
  --output-root OUTPUT_DIRECTORY --validate-only
```

## Score and metrics

The Act score is:

```text
log p(mutant | masked context) - log p(WT | masked context)
```

Higher values rank first. The shared evaluator implements Hit@10,
Top-1%-hits@10, MaxFit@10, MeanFit@10, Top1%Rank, NDCG@10, and Spearman. It
handles invalid/self/stop/duplicate mutation rows explicitly, uses a ceiling
for fractional top-percent cutoffs, and applies deterministic mutation-ID tie
handling. The runner collects per-run metrics in `aggregate.csv`. The separate
four-model table generators, which averaged seeds within protein before
equal-protein macro-averaging, are not included.

## Data provenance

Each included enzyme has a processed WT FASTA, truth CSV and generated metadata.
Source paths and hashes in `data/*_meta.json` are historical provenance records.
Raw/intermediate source files and the rebuild/source-hash validation utility are
not bundled. The evaluator reads the processed FASTA and truth CSV directly.
The combined manifest is an input catalog; running it requires adding prediction
files or checkpoint scans to `runs`.

| Dataset | Source identifier | Processing and numbering used for the included inputs |
| --- | --- | --- |
| NDM1 | Nature Communications 2024, s41467-024-52614-w, Supplementary Data 3 and 7; UniProt C7C422 | SD3 scores and assay WT. The included truth is **CTX32**; the early acquisition note's AMP 256 suggestion was not the final condition. |
| VIM2 | MaveDB urn:mavedb:00000073-b-1; DOI 10.7554/eLife.56707 | Translation frame selected by HGVS WT agreement. The extra Gly after the first Met is retained; canonical UniProt numbering requires a remap. |
| PjDHFR | PLOS Genetics 2024, journal.pgen.1011252, S2; Landrylab/Rouleau_et_al_2024; UniProt A0EPZ9 | MTX IC75 scores, using the UniProt protein reference. Yeast-codon-optimized supplement DNA must not replace the numbering reference by blind translation. |
| RNaseIII | MBE 2023, msad047, Supplementary Data S1; UniProt P0A7Y0 | The workbook's `Fitness (weighted mean)` sheet. Fitness, transformation and functional scores are distinct phenotypes. |
| CYP2C9 | MaveDB urn:mavedb:00000095-a-1; UniProt P11712 | Activity scores and UniProt WT, with construct-DNA cross-checks recorded in metadata. |
| PTEN | MaveDB urn:mavedb:00000054-a-1; DOI 10.1016/j.ajhg.2018.03.018; UniProt P60484 | Scores restricted to truthy `High_conf` rows and UniProt WT, with construct-DNA cross-checks. |

The NDM1 assay WT was extracted from the GenBank-format Supplementary Data 7
construct: CDS 1854..2672 on the positive strand, standard genetic code,
translation truncated at the first stop, cross-checked against C7C422.
Rebuilding the processed inputs requires obtaining the original sources and
reimplementing the recorded processing choices.

Raw sequencing reads are not included. Recorded accessions are NDM1 SRA
`PRJNA974578`, PjDHFR SRA `PRJNA1013557`, RNaseIII SRA `PRJNA887445`, and CYP2C9
GEO `GSE165412`.

The stored MaveDB source records for VIM2, CYP2C9 and PTEN declared CC0 1.0.
This records their supplied metadata and does not assign terms to other
supplements or third-party models. The original providers' terms apply.
