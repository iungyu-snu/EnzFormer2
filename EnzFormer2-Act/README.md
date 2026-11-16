# EnzFormer2-Act

EnzFormer2-Act fine-tunes an enzyme-family model and ranks single amino-acid
substitutions. Built on [ESM-C 600M](https://huggingface.co/biohub/esmc-600m-2024-12).
See the [Temp guide](../EnzFormer2-Temp/README.md) for stability prediction and the
[combined guide](../README.md) for both models.

## Usage

### 1. Install

From the top-level EnzFormer2 directory:

```bash
cd EnzFormer2-Act
conda env create --file environment.yml
conda activate enzformer2_act
python -m pip check
```

### 2. Collect a family corpus

Set `OPENROUTER_API_KEY` in `EnzFormer2-Act/.env`:

```dotenv
OPENROUTER_API_KEY=your_openrouter_key
```

The CLI reads this file automatically; a working-directory `.env` takes
precedence. Keep API keys out of Git.

Create a working directory and copy a single-record WT FASTA into it.
Run the remaining Act commands from this directory:

```bash
export ACT_WORKDIR=/path/to/my-enzyme-run
mkdir -p "$ACT_WORKDIR/inputs"
cp /path/to/query.fasta "$ACT_WORKDIR/inputs/query.fasta"
cd "$ACT_WORKDIR"

enzformer2 profile \
  --enzyme-type "NDM-1 metallo-beta-lactamase" \
  --query-fasta inputs/query.fasta \
  --llm-provider openrouter \
  --output-root retrieval \
  --run-id profile_run
```

Replace the enzyme description with your target. The default LLM is
`openai/gpt-5.4-pro`; override it with `--llm-model MODEL_ID`.
Downloads are unlimited by default; `--download-limit 50` limits each accession
source for a small trial.

Key outputs under `retrieval/profile_run/`:

- `llm/enzyme_profile.json`: saved family profile.
- `single_fasta/`: collected sequences and `manifest.tsv`.
- `logs/step_status.json`: stage status and failures.

To reuse saved inputs, replace the provider flag with
`--frozen-profile /path/to/enzyme_profile.json`; add
`--frozen-corpus-manifest /path/to/manifest.tsv` to reuse a local corpus.
Use a new run ID for another collection.

### 3. Prepare and train

This small example checks execution with 16 sequences and one epoch:

```bash
enzformer2 act prepare \
  --corpus-dir retrieval/profile_run/single_fasta \
  --query-fasta inputs/query.fasta \
  --output-dir training/qwoff \
  --max-sequences 16 \
  --epochs 1 \
  --batch-size 1 \
  --precision bf16 \
  --validation-fraction 0.25 \
  --seed 13

enzformer2 act train \
  --config training/qwoff/train.yaml \
  --validate-config

enzformer2 act train \
  --config training/qwoff/train.yaml \
  --device cuda
```

### 4. Rank substitutions

Export the top ten recommendations:

```bash
enzformer2 act recommend \
  --checkpoint training/qwoff/best.pt \
  --fasta inputs/query.fasta \
  --seq-id NDM1 \
  --top-k 10 \
  --chunk-size 1 \
  --device cuda \
  --output predictions/top10.csv
```

Or export all single substitutions:

```bash
enzformer2 act score \
  --checkpoint training/qwoff/best.pt \
  --fasta inputs/query.fasta \
  --seq-id NDM1 \
  --chunk-size 1 \
  --device cuda \
  --output predictions/all_mutants.csv
```

Replace `NDM1` with your query ID. Outputs include `seq_id`, 1-based `pos`,
`wt_aa`, `mut_aa`, `score` and `rank`; higher scores rank first.
A length-L sequence yields 19 × L substitutions. `--top-k` limits the exported
subset; all positions are scanned.

### 5. Evaluate against measurements

Supply a truth CSV with `pos,wt_aa,mut_aa,fitness` and numbering matching the WT:

```bash
enzformer2 act evaluate \
  --checkpoint training/qwoff/best.pt \
  --wt-fasta inputs/query.fasta \
  --truth-csv /path/to/target_truth.csv \
  --output-dir evaluation/target \
  --seq-id NDM1 \
  --model-name EnzFormer2-Act \
  --chunk-size 1 \
  --device cuda
```

This runs a fresh scan and writes `predictions.csv`, `merged_predictions.csv`
and `metrics.json`. Check matched and dropped row counts in the report.
For multiple targets, see `enzformer2 act benchmark --help`.

## Data, models and reproducibility

- [Benchmark guide](benchmark/README.md): included activity inputs and evaluation.
- [Act configuration](configs/paper/act/qwoff.yaml): editable training template.
- The accompanying Zenodo deposit supplies paper-run corpora, trained Act
  adapters, predictions and source tables. ESM-C weights come from its provider.

## License

Code is distributed under the [MIT license](LICENSE); third-party models and
data retain their providers' terms.
