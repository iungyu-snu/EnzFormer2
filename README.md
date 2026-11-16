# EnzFormer2

EnzFormer2 provides two complementary models for enzyme engineering in one
repository: **EnzFormer2-Act** for activity-focused mutation ranking and
**EnzFormer2-Temp** for predicted thermostability changes. Both use the pretrained
[ESM-C 600M](https://huggingface.co/biohub/esmc-600m-2024-12) protein language model,
with separate adaptation and scoring methods.

| Model | Workflow | Score | Start here |
| --- | --- | --- | --- |
| EnzFormer2-Act | Build an enzyme-family sequence corpus, fine-tune LoRA, and rank single substitutions | Masked log odds; higher ranks first | [Act guide](EnzFormer2-Act/README.md) |
| EnzFormer2-Temp | Score single substitutions with the released three-seed ensemble | Predicted ΔΔG in kcal/mol; stabilizing score = −ΔΔG | [Temp guide](EnzFormer2-Temp/README.md) |

Act scores prioritize substitutions without using target-enzyme activity labels
for fine-tuning. Temp predicts changes in stability rather than absolute melting
temperature. The scores have different meanings and are reported separately.

## Layout

```text
EnzFormer2-Act/     Activity package, configuration, benchmarks and documentation
EnzFormer2-Temp/    Thermostability package, configuration, released checkpoints and documentation
```

Each model directory is its own installation and configuration root.

### EnzFormer2-Act

```bash
cd EnzFormer2-Act
conda env create --file environment.yml
conda activate enzformer2_act
python -m pip check
enzformer2 --help
```

Follow the [Act workflow](EnzFormer2-Act/README.md#usage) for
`profile → act prepare → act train → act score / act evaluate`.
The guide supports a separate user-selected working directory. A frozen profile
and corpus can be reused; live profile generation requires an API key.

### EnzFormer2-Temp

```bash
cd EnzFormer2-Temp
conda env create --file enviroment.yml
conda activate enzformer2_temp
enzformer2-temp score \
  --checkpoint-dir checkpoints/enzformer2_temp \
  --fasta /path/to/query.fasta \
  --sequence-id query \
  --batch-size 16 \
  --top-k 20 \
  --output-dir artifacts/query_scan
```

Replace the FASTA path with your single-record WT sequence. The supplied ensemble
contains seeds 42, 43 and 44 and averages decoded ΔΔG predictions. See the
[Temp guide](EnzFormer2-Temp/README.md#usage) for outputs and runtime options.

## Data and reproducibility

- **Act:** the existing public benchmark inputs are included. Exact paper-run
  corpora, trained Act adapters and full experiment outputs are separate assets.
  See [Act data and reproducibility](EnzFormer2-Act/README.md#data-models-and-reproducibility).
- **Temp:** complete final ensemble bundles are included in this repository and
  model source distributions; wheels contain code only. Training datasets and
  experimental results are external. See
  [Temp data and models](EnzFormer2-Temp/README.md#data-and-models)
  and [training and evaluation](EnzFormer2-Temp/README.md#training-and-evaluation).

## References

Both models use the pretrained ESM-C 600M backbone. Please cite the upstream
ESM-C work when using EnzFormer2:

- Candido, S., Hayes, T., Derry, A., et al. (2026).
  [Language Modeling Materializes a World Model of Protein Biology](https://doi.org/10.64898/2026.06.03.729735).
  *bioRxiv* (preprint).
- ESM Team (2024).
  [ESM Cambrian: Revealing the mysteries of proteins with unsupervised learning](https://www.evolutionaryscale.ai/blog/esm-cambrian).
  EvolutionaryScale.
- ESM-C 600M pretrained weights:
  [`biohub/esmc-600m-2024-12`](https://huggingface.co/biohub/esmc-600m-2024-12).

## License

The authors' software is distributed under the [MIT license](LICENSE), including
EnzFormer2-Act and EnzFormer2-Temp. Third-party materials retain their original
terms. Model and dataset availability are described in each model's README.
