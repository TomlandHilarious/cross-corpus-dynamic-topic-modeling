#!/usr/bin/env python
"""
Qualitative case-study deviation-only heatmaps for SB-RA (source-adaptation D-ETM).

For each selected topic k in {7, 16}, produce a 3-panel heatmap of deviations
from the shared backbone (the shared backbone itself is NOT plotted):
    Panel 1: COHA deviation  D^{(COHA)}_{k,t}(w) = beta^{(COHA)}_{k,t}(w) - beta^{(0)}_{k,t}(w)
    Panel 2: HBR  deviation
    Panel 3: ILR  deviation

The three panels share a single symmetric diverging color scale per topic.

Word list per topic is built from two rule-based groups:
  G1 persistent source-emphasis: among each source's top-100 by avg beta^{(s)},
                                 top 2 by avg deviation D^{(s)} (positive)
  G2 rising source-emphasis:     among each source's late-period top-100 by
                                 avg beta^{(s)}, top 2 by (late dev - early dev)
Duplicates are removed; we aim for ~8-12 words and preserve source balance.

Outputs:
  - PNGs:           topic_{k}_heatmap.png
  - audit CSV:      selection_audit.csv
  - summary CSV:    selected_words_per_topic.csv
  - latex captions: latex_captions.txt
"""

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---- paths / config -------------------------------------------------------
DETM_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, DETM_DIR)
import detm  # noqa: F401  (registers detm.DETM for unpickling)

DEFAULT_CKPT = (
    f'{Path(__file__).resolve().parent.parent}/detm_source_adapted_5year/'
    'adapt_kl0.3_anchor1e-3_20260325_015039/'
    'detm_merged_K_20_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_'
    'Lr_1e-05_Bsz_500_RhoSize_300_L_3_minDF_100_trainEmbeddings_1_lora_r8.pt'
)
DEFAULT_VOCAB = (
    f'{Path(__file__).resolve().parent.parent}/data_processing_scripts/'
    'merged_v2_min100_5year_v2/min_df_100/vocab.pkl'
)
DEFAULT_OUT = f'{Path(__file__).resolve().parent}/paper_figures/case_study_sbra'

SOURCE_NAMES = ['COHA', 'HBR', 'ILR']  # src_id 0,1,2
SRC_COLORS = {'Shared': '#444444', 'COHA': '#1f77b4', 'HBR': '#d62728', 'ILR': '#2ca02c'}
SRC_LS = {'Shared': '--', 'COHA': '-', 'HBR': '-', 'ILR': '-'}
TOPICS = [0, 12, 15]

# Light editorial drop list for main-text figures: generic, low-semantic words
# that the rule may surface but which reviewers will rightly flag as artifacts.
# Rule itself is unchanged; these are simply skipped when building the display set.
DROP_WORDS_PER_TOPIC = {
    0:  {'year'},
    12: {'year'},
    15: {'year'},
}

# Short hand topic labels (used in figure suptitles).
TOPIC_LABELS = {
    0:  'Topic 0: MacroEconomics Transformation',
    12: 'unions / labor relations',
    15: 'management / quality / leadership',
}

# Source-distinctive top-word trajectory figure: one row per (topic, source).
# year list must use bin centers (1922 + 5k); top_n is words shown per time point.
SOURCE_DISTINCTIVE_CASES = [
    {'topic': 0,  'source': 'HBR', 'label': 'macroeconomic / industrial transformation'},
    {'topic': 12, 'source': 'ILR', 'label': 'labor relations and collective action'},
    {'topic': 15, 'source': 'HBR', 'label': 'corporate management and business organization'},
]
SOURCE_DISTINCTIVE_YEARS = [1932, 1952, 1972, 1987, 1997, 2017]
SOURCE_DISTINCTIVE_POOL  = 100   # top-N by beta^(s) at each (k,t) for candidate set
SOURCE_DISTINCTIVE_TOPN  = 7     # words per time point shown in figure (5-8)

# Manually selected word sets for the raw-beta line plot.
LINEPLOT_WORDS = {
    0: ['depression', 'internet', 'labor', 'wage', 'consumer'],
}

# 5-year bins: T=20 -> 1922,1927,...,2017
def time_year_labels(T):
    return [1922 + 5 * i for i in range(T)]

EARLY_YEARS = {1922, 1927, 1932, 1937, 1942, 1947, 1952}
LATE_YEARS  = {1992, 1997, 2002, 2007, 2012, 2017}

# Word-selection knobs
TOPN_PERSIST_PER_SRC = 2  # per source
TOPN_RISING_PER_SRC = 2   # per source
TOP_CANDIDATE_POOL = 100  # top-100 per-source restriction
MAX_WORDS = 12            # target ~8-12 words
SHARED_TOP_N = 10         # backbone top words reported alongside (caption text)

# Heatmap color-range clip: vmax = this percentile of |D| across plotted cells
HEATMAP_VMAX_PCT = 97.0   # clip the top ~3% so mid-range deviations are visible


# ---- model / vocab loading ------------------------------------------------
def load_model_and_vocab(ckpt_path, vocab_path, device):
    print(f"[load] ckpt:  {ckpt_path}")
    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.eval()
    print(f"[load] vocab: {vocab_path}")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    assert len(vocab) == model.vocab_size, (
        f"vocab/model mismatch: |V|={len(vocab)} vs model.V={model.vocab_size}"
    )
    print(f"[load] K={model.num_topics} T={model.num_times} V={model.vocab_size} "
          f"S={model.num_sources} src_adapt={model.source_adaptation_mode}")
    return model, vocab


@torch.no_grad()
def extract_betas(model):
    """Return shared (K,T,V) and per-source (S,K,T,V) numpy arrays."""
    # Use posterior means (eval mode) for determinism.
    alpha_global, _ = model.get_alpha()  # (K, T, L); eval => returns mu
    beta_shared = model.get_beta(alpha_global).cpu().numpy()  # (K, T, V)
    betas_src = []
    for s in range(model.num_sources):
        b = model.get_beta_source(s, alpha_global).cpu().numpy()
        betas_src.append(b)
    betas_src = np.stack(betas_src, axis=0)  # (S, K, T, V)
    return beta_shared, betas_src


# ---- word selection -------------------------------------------------------
def _vocab_to_idx2w(vocab, V):
    if isinstance(vocab, list):
        return list(vocab)
    # dict: could be word->idx or idx->word
    sample_key = next(iter(vocab))
    if isinstance(sample_key, str):
        return [w for w, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    return [vocab[i] for i in range(V)]


def select_words_for_topic(k, beta_shared, betas_src, vocab, years, drop_words=None):
    """
    Deviation-only selection (no shared-core group).

    Returns:
      ordered_idx:   list[int]   vocab indices, display order (rows of heatmap)
      ordered_words: list[str]   words for display
      audit:         list[dict]  one row per selected word
      shared_top:    list[str]   top-N shared-backbone words (for caption text)
    """
    drop_words = set(drop_words or [])
    V = beta_shared.shape[2]
    S = betas_src.shape[0]
    idx2w = _vocab_to_idx2w(vocab, V)

    yr = np.array(years)
    early_mask = np.array([y in EARLY_YEARS for y in yr])
    late_mask  = np.array([y in LATE_YEARS  for y in yr])

    # Shared backbone top words (reported separately)
    core_score = beta_shared[k].mean(axis=0)                           # (V,)
    shared_top_idx = np.argsort(-core_score)[:SHARED_TOP_N].tolist()
    shared_top = [idx2w[i] for i in shared_top_idx]

    # Per-source deviations / averages
    dev_full        = betas_src[:, k] - beta_shared[k][None]           # (S, T, V)
    source_dev_mean = dev_full.mean(axis=1)                            # (S, V)
    avg_src         = betas_src[:, k].mean(axis=1)                     # (S, V)
    early_dev       = dev_full[:, early_mask, :].mean(axis=1)          # (S, V)
    late_dev        = dev_full[:, late_mask, :].mean(axis=1)           # (S, V)
    rising_dev      = late_dev - early_dev                             # (S, V)
    late_avg_src    = betas_src[:, k][:, late_mask, :].mean(axis=1)    # (S, V)

    def _not_dropped(i):
        return idx2w[i] not in drop_words

    # ---- Group 1: persistent source emphasis (per source) ----
    persist_picks = []  # (idx, src_id, score)
    for s in range(S):
        cand = np.argsort(-avg_src[s])[:TOP_CANDIDATE_POOL]
        scored = [(int(i), float(source_dev_mean[s, i]))
                  for i in cand
                  if source_dev_mean[s, i] > 0 and _not_dropped(int(i))]
        scored.sort(key=lambda x: -x[1])
        for i, d in scored[:TOPN_PERSIST_PER_SRC]:
            persist_picks.append((i, s, d))

    # ---- Group 2: rising source emphasis (per source) ----
    rising_picks = []
    for s in range(S):
        cand = np.argsort(-late_avg_src[s])[:TOP_CANDIDATE_POOL]
        scored = [(int(i), float(rising_dev[s, i]))
                  for i in cand
                  if rising_dev[s, i] > 0 and _not_dropped(int(i))]
        scored.sort(key=lambda x: -x[1])
        for i, d in scored[:TOPN_RISING_PER_SRC]:
            rising_picks.append((i, s, d))

    # ---- merge with source-balanced round-robin, dedupe ----
    # Build per-source queues, alternate across sources for both groups.
    def round_robin(picks):
        per_src = {s: [] for s in range(S)}
        for i, s, d in picks:
            per_src[s].append((i, s, d))
        out = []
        depth = max((len(v) for v in per_src.values()), default=0)
        for k_ in range(depth):
            for s in range(S):
                if k_ < len(per_src[s]):
                    out.append(per_src[s][k_])
        return out

    persist_ordered = round_robin(persist_picks)
    rising_ordered  = round_robin(rising_picks)

    seen, ordered_idx, audit = set(), [], []

    def add(i, group, src_id, score):
        if i in seen or len(ordered_idx) >= MAX_WORDS:
            return
        seen.add(i)
        ordered_idx.append(i)
        audit.append({
            'topic': k, 'word': idx2w[i], 'group': group,
            'source': SOURCE_NAMES[src_id], 'score': float(score),
        })

    for i, s, d in persist_ordered:
        add(i, 'persistent source emphasis', s, d)
    for i, s, d in rising_ordered:
        add(i, 'rising source emphasis', s, d)

    ordered_words = [idx2w[i] for i in ordered_idx]
    # Carry per-source supporting stats for downstream line-plot selection.
    extras = {
        'avg_src': avg_src,                # (S, V)
        'late_avg_src': late_avg_src,      # (S, V)
        'source_dev_mean': source_dev_mean,# (S, V)
        'rising_dev': rising_dev,          # (S, V)
        'idx2w': idx2w,
    }
    return ordered_idx, ordered_words, audit, shared_top, extras


# ---- plotting -------------------------------------------------------------
def plot_topic_heatmap(k, word_idx, words, beta_shared, betas_src, years, out_png):
    """3-panel deviation-only heatmap for topic k."""
    shared_mat = beta_shared[k][:, word_idx].T   # (W, T)
    dev_mats = [(betas_src[s, k][:, word_idx].T - shared_mat)
                for s in range(betas_src.shape[0])]   # each (W, T)

    dev_max = max(np.abs(m).max() for m in dev_mats)
    if dev_max == 0:
        dev_max = 1e-12

    # Robust color range: clip the top (100-pct) percentile of |D| so mid-range
    # deviations are not washed out by a few extreme cells.
    all_abs = np.abs(np.concatenate([m.ravel() for m in dev_mats]))
    vmax = float(np.percentile(all_abs, HEATMAP_VMAX_PCT))
    if vmax <= 0:
        vmax = float(dev_max)
    # Make cells approximately square in the rendered figure.
    W, T = dev_mats[0].shape
    cell = 0.32  # inches per cell
    fig_w = max(13.0, 3 * cell * T + 3.5)
    fig_h = max(3.5, cell * W + 1.8)
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), constrained_layout=True)

    im = None
    for j, src in enumerate(SOURCE_NAMES):
        ax = axes[j]
        im = ax.imshow(dev_mats[j], aspect='equal', cmap='RdBu_r',
                       vmin=-vmax, vmax=+vmax, interpolation='nearest')
        ax.set_title(
            rf'{src} deviation $\beta^{{({src})}}_{{k,t}}(w)-\beta^{{(0)}}_{{k,t}}(w)$',
            fontsize=11)
        # Black gridlines between every cell
        ax.set_xticks(np.arange(-0.5, T, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, W, 1), minor=True)
        ax.grid(which='minor', color='black', linewidth=0.4)
        ax.tick_params(which='minor', length=0)
    fig.colorbar(im, ax=axes, shrink=0.85,
                 label=f'deviation (clipped at ±{HEATMAP_VMAX_PCT:.0f}th pct)')

    xticks = list(range(0, T, 2))
    xtlabels = [str(years[i]) for i in xticks]
    for j, ax in enumerate(axes):
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtlabels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words if j == 0 else [], fontsize=9)
        ax.set_xlabel('year', fontsize=9)

    fig.suptitle(f'SB-RA Topic {k}: source-specific deviations from shared backbone',
                 fontsize=13)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] saved {out_png}")


# ---- raw beta line plot ---------------------------------------------------
def _plot_one_topic_lines(k, words, beta_shared, betas_src, years, out_png):
    """One figure for topic k: 1 row x 3 corpus panels.

    Style:
    - Dashed lines with dot markers; no grid.
    - Per-panel legend inside each subplot.
    - Y-axis: 'Word probability'.
    - Subplot titles: corpus name only.
    - No x-axis label or tick labels.
    """
    cols = len(SOURCE_NAMES)
    base_palette = plt.get_cmap('tab10').colors
    word_colors = {w: base_palette[i % len(base_palette)]
                   for i, (w, _) in enumerate(words)}

    fig, axes = plt.subplots(1, cols, figsize=(13, 3.4),
                             sharey=True, constrained_layout=False)

    # Common y-limit across the 3 corpus panels for direct comparability
    max_y = 0.0
    for s_id in range(cols):
        for _, wi in words:
            max_y = max(max_y, float(betas_src[s_id, k, :, wi].max()))
    max_y = max_y * 1.08 if max_y > 0 else 1e-3

    yr_min, yr_max = years[0], years[-1]   # 1922 .. 2017

    for c, src_name in enumerate(SOURCE_NAMES):
        ax = axes[c]
        for w, wi in words:
            ax.plot(years, betas_src[c, k, :, wi],
                    color=word_colors[w], lw=1.6,
                    linestyle='--', marker='o', markersize=3.5,
                    label=w)
        # Vertical dashed year markers instead of grid
        for yr in years:
            ax.axvline(yr, color='#cccccc', linewidth=0.5, linestyle='--', zorder=0)
        ax.set_title(src_name, fontsize=12)
        ax.set_xlim(yr_min, yr_max)
        ax.set_xticks(years[::2])
        ax.set_xticklabels([str(y) for y in years[::2]],
                            rotation=45, ha='right', fontsize=7.5)
        ax.tick_params(axis='y', labelsize=8)
        ax.set_ylim(0, max_y)
        ax.spines['bottom'].set_linewidth(0.6)
        if c == 0:
            ax.set_ylabel('Word probability', fontsize=10)

    # Single shared legend centred below all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(words),
               fontsize=8.5, framealpha=0.9,
               handlelength=1.4, columnspacing=1.2,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        TOPIC_LABELS.get(k, f'Topic {k}'),
        fontsize=13, y=1.02,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88,
                        bottom=0.18, wspace=0.12)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] saved {out_png}")


def plot_raw_beta_lines(topics, audit_per_topic, extras_per_topic,
                        beta_shared, betas_src, years, out_dir):
    """One figure per topic; rows of 3 corpus panels with shared word legend."""
    idx2w = extras_per_topic[topics[0]]['idx2w']
    w2idx = {w: i for i, w in enumerate(idx2w)}

    all_records = []
    for k in topics:
        wlist = LINEPLOT_WORDS.get(k, [])
        if not wlist:
            continue
        words, missing = [], []
        for w in wlist:
            if w in w2idx:
                words.append((w, w2idx[w]))
            else:
                missing.append(w)
        if missing:
            print(f"[warn] topic {k}: words not in vocab, dropped: {missing}")
        if not words:
            continue
        out_png = out_dir / f'raw_beta_lineplot_topic_{k}.png'
        _plot_one_topic_lines(k, words, beta_shared, betas_src, years, out_png)
        for w, _ in words:
            all_records.append({'topic': k, 'word': w, 'words_set_kind': 'manual'})

    csv_path = out_dir / 'lineplot_selected_words.csv'
    with open(csv_path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['Topic', 'Word', 'Word set'])
        for r in all_records:
            wr.writerow([r['topic'], r['word'], r['words_set_kind']])
    print(f"[list]  wrote {csv_path}")
    return all_records


# ---- source-distinctive top-word trajectory figure ------------------------
def plot_source_distinctive_trajectories(
    cases, years_list, beta_shared, betas_src, vocab, out_dir,
    pool=SOURCE_DISTINCTIVE_POOL, top_n=SOURCE_DISTINCTIVE_TOPN,
):
    """Compact top-word trajectory figure for selected (topic, source) cases.

    For each case and each selected year t, we:
      - take the top `pool` words by beta^{(s)}_{k,t}(w);
      - keep only those with D = beta^{(s)} - beta^{(0)} > 0;
      - rank by Score = beta^{(s)} * D, take top `top_n` words;
      - render them as a vertical word list at column t.
    """
    V = beta_shared.shape[1] if beta_shared.ndim == 2 else beta_shared.shape[2]
    idx2w = _vocab_to_idx2w(vocab, V)
    all_years = years_list                 # full list 1922..2017
    year2col = {y: i for i, y in enumerate(all_years)}
    cols_idx = [year2col[y] for y in SOURCE_DISTINCTIVE_YEARS]
    sel_years = SOURCE_DISTINCTIVE_YEARS
    n_t = len(sel_years)
    n_rows = len(cases)

    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(1.65 * n_t + 1.5, 1.05 * top_n * n_rows + 1.5),
                             constrained_layout=False)
    if n_rows == 1:
        axes = [axes]

    records = []
    for r, case in enumerate(cases):
        k = case['topic']
        src_name = case['source']
        s_id = SOURCE_NAMES.index(src_name)
        ax = axes[r]
        ax.set_xlim(0.5, n_t + 0.5)
        ax.set_ylim(0, top_n + 1)
        ax.invert_yaxis()
        ax.set_xticks(range(1, n_t + 1))
        ax.set_xticklabels([str(y) for y in sel_years], fontsize=10)
        ax.tick_params(axis='x', length=0, pad=4)
        ax.set_yticks([])
        for spine in ('top', 'right', 'left'):
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.6)

        ax.set_title(
            f"Topic {k} \u2014 {src_name} ({case['label']})",
            fontsize=11, loc='left', pad=6,
        )

        for col_pos, (t_idx, yr) in enumerate(zip(cols_idx, sel_years), start=1):
            beta_s = betas_src[s_id, k, t_idx, :]      # (V,)
            beta_0 = beta_shared[k, t_idx, :]          # (V,)
            dev = beta_s - beta_0
            # Top `pool` candidates by beta^(s)
            cand = np.argpartition(-beta_s, min(pool, len(beta_s) - 1))[:pool]
            scored = []
            for i in cand:
                d = dev[i]
                if d <= 0:
                    continue
                bs = beta_s[i]
                scored.append((float(bs * d), int(i), float(bs), float(d)))
            scored.sort(key=lambda x: -x[0])
            picks = scored[:top_n]

            for rank, (score, i, bs, d) in enumerate(picks, start=1):
                w = idx2w[i]
                ax.text(col_pos, rank, w,
                        ha='center', va='center', fontsize=9.5)
                records.append({
                    'topic': k, 'source': src_name, 'year': yr, 'word': w,
                    'beta': bs, 'deviation': d, 'score': score, 'rank': rank,
                })

    fig.suptitle(
        'Source-distinctive top-word trajectories '
        r'(top-$n$ by $\beta^{(s)}_{k,t}(w)\cdot D^{(s)}_{k,t}(w)$, '
        r'$D>0$, within top-100 of $\beta^{(s)}$)',
        fontsize=12, y=0.995,
    )
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92,
                        bottom=0.06, hspace=0.55)

    out_png = out_dir / 'source_distinctive_top_word_trajectories.png'
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] saved {out_png}")

    csv_path = out_dir / 'source_distinctive_words.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Topic', 'Source', 'Year', 'Rank', 'Word',
                    'Beta', 'Deviation', 'Score'])
        for r in records:
            w.writerow([r['topic'], r['source'], r['year'], r['rank'], r['word'],
                        f"{r['beta']:.6e}", f"{r['deviation']:.6e}",
                        f"{r['score']:.6e}"])
    print(f"[list]  wrote {csv_path}")
    return records


# ---- main -----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=DEFAULT_CKPT)
    ap.add_argument('--vocab',      default=DEFAULT_VOCAB)
    ap.add_argument('--out_dir',    default=DEFAULT_OUT)
    ap.add_argument('--topics', type=int, nargs='+', default=TOPICS)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cpu')

    model, vocab = load_model_and_vocab(args.checkpoint, args.vocab, device)
    beta_shared, betas_src = extract_betas(model)
    K, T, V = beta_shared.shape
    years = time_year_labels(T)
    print(f"[info] years: {years[0]}..{years[-1]} (T={T})")

    all_audit = []
    selected_per_topic = {}
    shared_top_per_topic = {}
    extras_per_topic = {}
    audit_per_topic = {}
    for k in args.topics:
        print(f"\n[topic {k}] selecting words...")
        drop = DROP_WORDS_PER_TOPIC.get(k, set())
        if drop:
            print(f"  editorial drop list (skipped, not in figure): {sorted(drop)}")
        idxs, words, audit, shared_top, extras = select_words_for_topic(
            k, beta_shared, betas_src, vocab, years, drop_words=drop,
        )
        selected_per_topic[k] = words
        shared_top_per_topic[k] = shared_top
        extras_per_topic[k] = extras
        audit_per_topic[k] = audit
        all_audit.extend(audit)
        print(f"  shared-backbone top {SHARED_TOP_N}: {shared_top}")
        print(f"  selected ({len(words)}): {words}")
        out_png = out_dir / f"topic_{k}_heatmap.png"
        plot_topic_heatmap(k, idxs, words, beta_shared, betas_src, years, out_png)

    # ---- Raw beta line-plot figure (per topic) ----
    plot_raw_beta_lines(
        args.topics, audit_per_topic, extras_per_topic,
        beta_shared, betas_src, years, out_dir,
    )

    # ---- Source-distinctive top-word trajectory figure ----
    plot_source_distinctive_trajectories(
        SOURCE_DISTINCTIVE_CASES, years,
        beta_shared, betas_src, vocab, out_dir,
    )

    # ---- audit CSV ----
    audit_csv = out_dir / 'selection_audit.csv'
    with open(audit_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Topic', 'Word', 'Selection type', 'Main source', 'Score'])
        for r in all_audit:
            w.writerow([r['topic'], r['word'], r['group'], r['source'], f"{r['score']:.6e}"])
    print(f"\n[audit] wrote {audit_csv}")

    # ---- per-topic word list CSV ----
    list_csv = out_dir / 'selected_words_per_topic.csv'
    with open(list_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Topic', 'Words (display order)'])
        for k, ws in selected_per_topic.items():
            w.writerow([k, ', '.join(ws)])
    print(f"[list]  wrote {list_csv}")

    # ---- shared-backbone top words (caption-side reporting) ----
    sb_csv = out_dir / 'shared_backbone_top_words.csv'
    with open(sb_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Topic', f'Shared-backbone top-{SHARED_TOP_N} words (mean over time)'])
        for k, ws in shared_top_per_topic.items():
            w.writerow([k, ', '.join(ws)])
    print(f"[shared] wrote {sb_csv}")

    # ---- LaTeX captions ----
    cap_path = out_dir / 'latex_captions.txt'
    with open(cap_path, 'w') as f:
        for k in args.topics:
            sb = ', '.join(shared_top_per_topic[k])
            cap = (
                rf"\caption{{SB-RA Topic {k}: source-specific deviations from the shared "
                rf"backbone, $D^{{(s)}}_{{k,t}}(w)=\beta^{{(s)}}_{{k,t}}(w)-\beta^{{(0)}}_{{k,t}}(w)$, "
                rf"for $s\in\{{\mathrm{{COHA}},\mathrm{{HBR}},\mathrm{{ILR}}\}}$ over five-year "
                rf"bins from 1922 to 2017. The three panels share a symmetric diverging color "
                rf"scale centered at zero; red cells indicate words receiving higher probability "
                rf"in the source-specific topic than in the shared backbone, blue cells the "
                rf"opposite. Words are selected by a fixed two-group rule: persistent source "
                rf"emphasis (top-2 per source by mean positive $D^{{(s)}}$ among each source's "
                rf"top-100 words by avg $\beta^{{(s)}}$) and rising source emphasis (top-2 per "
                rf"source by late-minus-early growth in $D^{{(s)}}$ among each source's "
                rf"late-period top-100 words). Shared-backbone top-{SHARED_TOP_N} words for this "
                rf"topic (not plotted): \textit{{{sb}}}.}}"
            )
            f.write(f"% Topic {k}\n{cap}\n\n")
        # Caption for the raw-beta line plot (single figure for all topics)
        line_cap = (
            r"\caption{Raw topic-word probability trajectories for selected "
            r"source-emphasis words in SB-RA Topics 7 and 16. Each subplot shows "
            r"$\beta_{k,t}(w)$ over five-year time bins from 1922 to 2017 for the "
            r"shared backbone and the three source-specific trajectories "
            r"(COHA, HBR, ILR). Words are drawn from the source-specific "
            r"deviation analysis (Fig.~\ref{fig:sbra_heatmaps}), prioritizing "
            r"rising source-emphasis words and falling back to persistent "
            r"source-emphasis words; among candidates we prefer those with high "
            r"raw average probability in the source-specific topic. These plots "
            r"complement the deviation heatmaps by showing whether source-specific "
            r"deviation words are also substantively salient within the learned "
            r"topic-word distributions.}"
        )
        f.write(f"% Raw-beta line plot\n{line_cap}\n\n")

        # Caption for the source-distinctive top-word trajectory figure
        sd_cap = (
            r"\caption{Source-distinctive top-word trajectories for selected "
            r"aligned topics. Each row corresponds to a (topic, source) case; "
            r"each column is a time bin. Within each cell we list the top "
            rf"{SOURCE_DISTINCTIVE_TOPN} words ranked by "
            r"$\mathrm{Score}^{(s)}_{k,t}(w)=\beta^{(s)}_{k,t}(w)\cdot "
            r"D^{(s)}_{k,t}(w)$ with $D^{(s)}_{k,t}(w)=\beta^{(s)}_{k,t}(w)-"
            r"\beta^{(0)}_{k,t}(w)>0$, restricted to candidates in the top-"
            rf"{SOURCE_DISTINCTIVE_POOL} of $\beta^{{(s)}}_{{k,t}}$. The score "
            r"favors words that are both salient in the source-specific topic "
            r"and more emphasized than in the shared backbone. The figure "
            r"summarizes how source-specific realizations of the same aligned "
            r"topics acquire corpus-local lexical emphasis over time.}"
        )
        f.write(f"% Source-distinctive top-word trajectories\n{sd_cap}\n\n")
    print(f"[caps]  wrote {cap_path}")


if __name__ == '__main__':
    main()
