# EnzFormer2-Temp

EnzFormer2-Temp predicts mutation-induced ΔΔG using
[ESM-C 600M](https://huggingface.co/biohub/esmc-600m-2024-12). The supplied ensemble
uses seeds 42, 43 and 44. See the [Act guide](../EnzFormer2-Act/README.md) for
activity ranking and the [combined guide](../README.md) for both models.

## Usage

### 1. Install

From the top-level EnzFormer2 directory:

```bash
cd EnzFormer2-Temp
conda env create --file enviroment.yml
conda activate enzformer2_temp
```

### 2. Score a sequence

Supply a single-record WT FASTA:

```bash
enzformer2-temp score \
  --checkpoint-dir checkpoints/enzformer2_temp \
  --fasta /path/to/query.fasta \
  --sequence-id query \
  --batch-size 16 \
  --top-k 20 \
  --output-dir artifacts/query_scan
```

Reduce `--batch-size` if GPU memory is insufficient. Use `--sequence` instead
of `--fasta` for a raw amino-acid sequence. To score with one member, point
`--checkpoint-dir` to `checkpoints/enzformer2_temp/seed42`.

A length-L sequence yields 19 × L substitutions; `--top-k` limits only the
exported top subset. Use a new output directory or `--allow-overwrite` to
replace existing outputs.

| Output | Contents |
| --- | --- |
| `predictions.csv`, `predictions.parquet` | All substitutions, mean and per-seed ΔΔG, and ranks |
| `top_mutations.csv` | Requested top subset |
| `candidates_with_logodds.parquet` | Candidate inputs and prior features |
| `prediction_report.json` | Checkpoints, seeds, settings and output paths |

`predicted_ddg` is in kcal/mol; negative values indicate stabilization.
`score_stabilizing = -predicted_ddg`, with higher scores ranked first.

## Training and evaluation

These commands require prepared Parquet inputs, supplied separately in the
Zenodo archive. Dataset mappings are in [configs/data/](configs/data/).

```bash
python scripts/precompute_esmc_masked_logodds.py \
  --input-dir data/processed/megascale_thermompnn \
  --out-dir data/processed/megascale_thermompnn_esmc_logodds

enzformer2-temp train \
  --config configs/paper/enzformer2_temp.yaml \
  --validate-config

enzformer2-temp train --config configs/paper/enzformer2_temp.yaml

# Replace DATA with the directory containing prepared validation/test parquets.
enzformer2-temp evaluate \
  --checkpoint-dir checkpoints/enzformer2_temp \
  --data-dir DATA \
  --out-dir artifacts/evaluation
```


## Data and models

The pretrained ensemble is included in `checkpoints/enzformer2_temp/`.
Training data, paper predictions and source tables are supplied in the
accompanying Zenodo archive.

## License

Code is distributed under the [MIT license](LICENSE); third-party models and
data retain their providers' terms.
