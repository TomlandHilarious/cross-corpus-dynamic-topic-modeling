# Cross-Corpus Dynamic Topic Modeling

Code accompanying the paper **“Dynamic Topic Modeling for Cross-Corpus Temporal Analysis”**, accepted at CIKM 2026.

This repository implements shared-backbone Dynamic Embedded Topic Models (D-ETM) for comparing temporally evolving topics across multiple corpora. The code supports independent baselines, joint multi-corpus training, source-specific residual adaptation, full fine-tuning baselines, seed robustness runs, K-sensitivity runs, and paper figure generation.

## Repository structure

- `DETM_weights/`: model code, training launchers, evaluation scripts, and paper-analysis scripts.
- `data_processing_scripts/`: preprocessing utilities for constructing temporal corpus inputs.
- `detm_merged_5year/`: shared-backbone SB-Joint checkpoint directory.
- `detm_source_adapted_5year/`: SB-RA source-adapted checkpoint directory.
- `detm_full_finetune_baseline/`: SB-FT full fine-tuning checkpoint directories.
- `detm_individual_merged_5year_v3/`: Ind-MV independently trained checkpoints.
- `detm_individual_specific_vocab_5year_v3/`: Ind-CS independently trained checkpoints.
- `multiseed_runs/`: seed robustness and K-sensitivity run outputs.
- `environment.yml`: minimal conda environment for running the code.

## Environment setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate shared_backbone_detm
```

The shell launchers use the active `python` by default. To run with a specific interpreter, set `PYTHON` explicitly:

```bash
PYTHON=/path/to/env/bin/python bash DETM_weights/run_merged_5year.sh
```

## Training and comparison conditions

The paper evaluates five training conditions. Some scripts and result files use the shorthand labels `A`-`E`:

| Code label | Paper condition | Description |
|---|---|---|
| `A` | Ind-CS | Independently train one D-ETM per corpus using that corpus's own active vocabulary. |
| `B` | Ind-MV | Independently train one D-ETM per corpus using the shared merged vocabulary. |
| `C` | SB-Joint | Train a single D-ETM on the merged multi-corpus collection to learn a shared dynamic topic backbone. |
| `D` | SB-RA | Initialize from SB-Joint, freeze the shared backbone, and learn source-specific residual topic offsets. |
| `E` | SB-FT | Initialize from SB-Joint and fully fine-tune all model parameters separately on each corpus. |

## Default model configuration

The Stage 1 conditions, Ind-CS, Ind-MV, and SB-Joint, use the same core D-ETM configuration unless otherwise noted:

- `num_topics=20`
- non-overlapping five-year temporal bins
- learning rate `5e-5`
- `epochs=80`
- `delta=0.01`
- `kl_alpha_scale=1e-6`
- `kl_weight_max=0.9`
- 50-epoch KL warmup
- `emb_size=300`
- `rho_size=300`
- `theta_hidden_size=800`
- `num_layers=3`
- `batch_size=500`
- `min_df=100`
- default seed `2019`

The Stage 2 conditions, SB-RA and SB-FT, are initialized from the SB-Joint checkpoint.

For SB-RA:

- the shared word embedding matrix and shared backbone topic embeddings are frozen
- source-specific residual topic offsets and inference networks remain trainable
- residuals are initialized at zero
- `epochs=20`
- learning rate `1e-5`
- `adapt_warmup_epochs=5`
- `adapt_kl_theta_max=0.3`
- default regularization uses `lambda_anchor=1e-3` and `lambda_smooth=1e-3`

For SB-FT:

- each corpus is initialized from the SB-Joint checkpoint
- all model parameters are fine-tuned separately on each corpus
- this baseline tests what happens when the shared backbone is not preserved

## Data layout

The training scripts expect processed D-ETM inputs under `data_processing_scripts/`. Important processed-data roots include:

- `data_processing_scripts/merged_v2_min100_5year_v2/`
- `data_processing_scripts/individual_corpora_min100_5year_v3/`
- `data_processing_scripts/individual_corpora_specific_vocab_5year_v3/`

Each processed corpus directory is expected to contain the vocabulary, temporal bag-of-words splits, timestamp metadata, and embeddings used by `main.py`.

## Main training workflows

Run commands from the repository root.

### Stage 1: independent baselines and shared backbone

Train the merged multi-corpus SB-Joint model:

```bash
bash DETM_weights/run_merged_5year.sh
```

Train independent merged-vocabulary baselines:

```bash
bash DETM_weights/run_individual_coha_merged_5year.sh
bash DETM_weights/run_individual_hbr_merged_5year.sh
bash DETM_weights/run_individual_ilr_merged_5year.sh
```

Train independent corpus-specific-vocabulary baselines:

```bash
bash DETM_weights/run_individual_coha_specific_vocab_5year.sh
bash DETM_weights/run_individual_hbr_specific_vocab_5year.sh
bash DETM_weights/run_individual_ilr_specific_vocab_5year.sh
```

### Stage 2: source adaptation and full fine-tuning

Train the SB-RA source-adapted model from SB-Joint:

```bash
bash DETM_weights/adapt_source_topics.sh
```

Train SB-FT full fine-tuning baselines:

```bash
bash DETM_weights/full_finetune_coha.sh
bash DETM_weights/full_finetune_hbr.sh
bash DETM_weights/full_finetune_ilr.sh
```

## Seed robustness and K-sensitivity runs

The `multiseed_runs/` directory is used for seed robustness and K-sensitivity experiments.

Run one fixed-backbone SB-RA ablation seed:

```bash
bash DETM_weights/run_fixed_backbone_ablation_seed.sh 1 0
```

Run one fixed-backbone SB-FT seed:

```bash
bash DETM_weights/run_fixed_backbone_sbft_seed.sh 1 0
```

Run K-sensitivity experiments for seed `2019`:

```bash
bash DETM_weights/run_k_sensitivity_K10_seed2019.sh
bash DETM_weights/run_k_sensitivity_K30_seed2019.sh
```

## Evaluation scripts

Core evaluation scripts include:

- `DETM_weights/evaluate_npmi_robustness.py`: evaluates TD, TC/NPMI, UMass, C_V, and TQ for the training conditions.
- `DETM_weights/alignment_metrics_trajectory.py`: computes trajectory-level alignment metrics, including same-index JSD, Hungarian-matched JSD, margin, and retrieval@1.
- `DETM_weights/evaluate_ppl.py`: evaluates held-out perplexity.
- `DETM_weights/evaluate_all_models.py`: legacy fixed summary script for all model families.

Alignment evaluation wrappers:

```bash
bash DETM_weights/run_sbft_reviewer_alignment.sh
bash DETM_weights/run_k_sensitivity_reviewer_alignment.sh
bash DETM_weights/evaluate_fixed_backbone_sbft_multiseed_alignment.sh
```

## Paper figures and qualitative analysis

Paper-facing analysis scripts write outputs under `DETM_weights/paper_figures/`.

- `DETM_weights/case_study_sbra_heatmap.py`: generates SB-RA case-study heatmaps, selected-word audits, raw-beta line plots, source-distinctive trajectory figures, and LaTeX captions.
- `DETM_weights/topic0_source_local_summary.py`: generates Topic 0 source-local top-word tables and figures.
- `DETM_weights/topic_overlap_global_summary.py`: summarizes Top-30 overlap and source-local counts across all topics and five-year bins.
- `DETM_weights/export_all_topics.py`: exports all topics, sources, and time bins to CSV and Markdown for manual inspection.

Example:

```bash
python DETM_weights/case_study_sbra_heatmap.py
python DETM_weights/topic0_source_local_summary.py
python DETM_weights/topic_overlap_global_summary.py
```

## Reproducibility notes

- Shell launchers use `PYTHON="${PYTHON:-python}"`, so they work with the active environment or a user-specified interpreter.
- Training scripts pass random seeds through `--seed`; the default seed is `2019`.
- `main.py` logs runtime, device information, CUDA availability, and peak CUDA memory usage at process exit.
- Shell launchers write logs with `tee`, so stdout timing and CUDA-memory information are preserved.
- SB-RA and SB-FT Stage 2 runs are initialized from the SB-Joint checkpoint.

## Citation

Citation information will be added after publication.
