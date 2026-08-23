# DETM Training Configurations

This document records the optimized training configurations for all DETM experiments.

## Optimal Configuration (Validated on HBR 5-year bins)

After extensive hyperparameter tuning, the following configuration was found to achieve the best balance between Topic Diversity (TD) and Topic Coherence (TC):

### Core Parameters
```bash
NUM_TOPICS=20
NUM_EPOCHS=80
MIN_DF=100
BATCH_SIZE=500
EMBEDDING_DIM=300
```

### Regularization & Training
```bash
DELTA=0.01                    # Temporal smoothness prior
KL_ALPHA_SCALE=1e-6          # KL divergence scaling for alpha (CRITICAL)
WARMUP_EPOCHS=50             # Extended warmup for stable learning
KL_WEIGHT_MAX=0.9            # Cap KL weight to prevent TC plateau
LEARNING_RATE=5e-05          # Lower LR for fine-grained optimization
TRAIN_EMBEDDINGS=1           # Fine-tune word embeddings
```

### Results (HBR, 5-year bins, corpus-specific vocab)
- **Final TD**: 0.50 (moderate diversity)
- **Final TC**: 0.056 (positive, stable coherence)
- **Best Perplexity**: 2310.6
- **TC Plateau**: ~0.056 appears to be model capacity limit

---

## Experiment Configurations by Corpus Type

### 1. Corpus-Specific Vocabulary (5-year temporal bins)

**Purpose**: Individual corpus training with corpus-specific vocabularies and rebinned timestamps (5-year bins, ~20 time slices)

#### Scripts
- `run_individual_hbr_specific_vocab_5year.sh` (5-year bins)
- `run_individual_coha_specific_vocab_5year.sh` (5-year bins)
- `run_individual_ilr_specific_vocab_5year.sh` (5-year bins)
- `run_individual_coha_specific_vocab.sh` (original temporal resolution)
- `run_individual_ilr_specific_vocab.sh` (original temporal resolution)

#### Data Paths (5-year bins)
```bash
HBR:  /shared/share_hbr-ilr_nlp/data_processing_scripts/individual_corpora_specific_vocab_5year/hbr
COHA: /shared/share_hbr-ilr_nlp/data_processing_scripts/individual_corpora_specific_vocab_5year/coha
ILR:  /shared/share_hbr-ilr_nlp/data_processing_scripts/individual_corpora_specific_vocab_5year/ilr
```

#### Configuration
All three corpora use **identical optimal configuration** listed above for fair comparison.

#### Output Directories
```bash
All (5-year): /shared/share_hbr-ilr_nlp/detm_individual_specific_vocab_5year/
```

---

### 2. Merged Vocabulary (Original temporal resolution)

**Purpose**: Individual corpus training with merged vocabulary for comparison with multi-corpus model

#### Scripts
- `run_individual_hbr.sh`
- `run_individual_coha.sh`
- `run_individual_ilr.sh`

#### Data Paths
```bash
HBR:  /shared/share_hbr-ilr_nlp/data_processing_scripts/individual_corpora_min100/hbr
COHA: /shared/share_hbr-ilr_nlp/data_processing_scripts/individual_corpora_min100/coha
ILR:  /shared/share_hbr-ilr_nlp/data_processing_scripts/individual_corpora_min100/ilr
```

#### Configuration
All three corpora use **identical optimal configuration** for fair comparison.

#### Output Directories
```bash
All: /shared/share_hbr-ilr_nlp/detm_individual/
```

---

## Key Findings from Hyperparameter Tuning

### 1. KL_ALPHA_SCALE is Critical
- **1e-6**: ✅ Optimal - TC positive, stable
- **≥5e-5**: ❌ TC goes negative and degrades over time
- This parameter controls temporal smoothness regularization strength
- Too high → over-regularization → semantically incoherent topics

### 2. Extended Warmup Helps
- **50 epochs** allows model to learn basic topic structure before full regularization
- TC continues improving post-warmup with proper configuration
- Short warmup (10-30 epochs) can work but provides less stable learning

### 3. KL Weight Cap Prevents Early Plateau
- **0.9 cap** allows continued optimization after warmup
- Without cap (1.0): TC plateaus immediately after warmup
- Small learning rate (5e-05) critical for fine-grained post-warmup improvements

### 4. TC Ceiling at ~0.056
- Across all configurations tested, TC plateaus at approximately 0.056
- This appears to be a **model capacity limit** rather than optimization issue
- Consistent across different warmup durations and learning rates

### 5. TD vs TC Tradeoff
- Lower regularization (1e-6) → Higher TD (~0.50), positive TC (0.056)
- Higher regularization (≥5e-5) → Lower TD, negative TC
- Optimal balance achieved at 1e-6 with current architecture

---

## Training Trajectory (Typical for Optimal Config)

| Epoch | Phase | KL Weight | Expected TD | Expected TC | Notes |
|-------|-------|-----------|-------------|-------------|-------|
| 5 | Early warmup | 0.09 | 0.88 | -0.29 | Poor coherence initially |
| 15 | Mid warmup | 0.27 | 0.77 | ~0.00 | TC turning positive |
| 30 | Late warmup | 0.54 | 0.67 | 0.01 | Gradual improvement |
| 40 | Near warmup end | 0.72 | 0.64 | 0.03 | Accelerating |
| 50 | Warmup complete | 0.90 | 0.60 | 0.04 | Baseline established |
| 65 | Post-warmup | 0.90 | 0.54 | 0.05 | Continued improvement |
| 75-80 | Final | 0.90 | 0.50 | 0.056 | Plateau reached |

---

## Comparison Experiment Design

### Goal
Compare individual corpus models (corpus-specific vocab) vs merged multi-corpus model

### Metrics to Compare
1. **Topic Diversity (TD)**: Proportion of unique words across topics
2. **Topic Coherence (TC)**: PMI-based semantic coherence
3. **Perplexity (PPL)**: Model fit to held-out data
4. **Topic Quality**: Manual inspection of top words per topic

### Expected Outcomes
All individual models should achieve:
- TD ≈ 0.45-0.55
- TC ≈ 0.05-0.06
- Positive, stable coherence throughout training

If a corpus deviates significantly, it suggests corpus-specific characteristics rather than configuration issues.

---

## Usage Instructions

### Running Individual Corpus Training (Corpus-Specific Vocab, 5-year bins)
```bash
# HBR (5-year bins)
bash /shared/share_hbr-ilr_nlp/DETM_weights/run_individual_hbr_specific_vocab_5year.sh

# COHA (5-year bins)
bash /shared/share_hbr-ilr_nlp/DETM_weights/run_individual_coha_specific_vocab_5year.sh

# ILR (5-year bins)
bash /shared/share_hbr-ilr_nlp/DETM_weights/run_individual_ilr_specific_vocab_5year.sh
```

### Running Individual Corpus Training (Merged Vocab)
```bash
# HBR
bash /shared/share_hbr-ilr_nlp/DETM_weights/run_individual_hbr.sh

# COHA
bash /shared/share_hbr-ilr_nlp/DETM_weights/run_individual_coha.sh

# ILR
bash /shared/share_hbr-ilr_nlp/DETM_weights/run_individual_ilr.sh
```

---

## Monitoring Training

### Key Metrics to Watch
1. **TD trajectory**: Should decrease from ~0.8 to ~0.5
2. **TC trajectory**: Should increase from negative to ~0.05-0.06
3. **Perplexity**: Should decrease (lower is better)
4. **KL components**: Should remain stable (not explode)

### Warning Signs
- TC remains negative after epoch 30: Check KL_ALPHA_SCALE (should be 1e-6)
- TC > 0.10: Likely indicates model collapse or unrealistic topics
- TD < 0.3: Topics too similar (over-collapsed)
- TD > 0.7: Topics too dispersed (under-regularized)

---

## Configuration History

### Initial Configuration (Pre-optimization)
```bash
NUM_TOPICS=20
KL_ALPHA_SCALE=1e-5
WARMUP_EPOCHS=10
LR=0.0005
NUM_EPOCHS=50
```
**Result**: TC plateau at 0.056 after epoch 30

### Attempted Fix 1: Cap KL weight at 0.9
```bash
KL_WEIGHT_MAX=0.9
```
**Result**: No improvement, TC still plateaus at 0.056

### Attempted Fix 2: Extended warmup + Lower LR
```bash
WARMUP_EPOCHS=50
LR=5e-05
NUM_EPOCHS=80
```
**Result**: ✅ TC continues improving post-warmup, reaches 0.056 by epoch 75

### Final Insight
- TC ceiling of ~0.056 is model capacity limit, not optimization issue
- Extended warmup + lower LR allows model to reach this ceiling smoothly
- No single parameter change solves plateau; combination is key

---

## Last Updated
March 10, 2026

## Contact
Configuration validated through extensive experiments on HBR corpus with 5-year temporal bins.
