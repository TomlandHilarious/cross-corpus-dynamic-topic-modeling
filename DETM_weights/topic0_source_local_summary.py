#!/usr/bin/env python
"""
Topic-0 source-local top words via simple set-difference.

For each source s in {COHA, HBR, ILR}, topic k=0, and five-year bin t:

    TopK^{(s)}_{k,t} = top-K words by beta^{(s)}_{k,t}
    TopK^{(0)}_{k,t} = top-K words by beta^{(0)}_{k,t}   (shared backbone)
    SourceLocal^{(s)}_{k,t} = TopK^{(s)}_{k,t} \\ TopK^{(0)}_{k,t}

We rank source-local words by raw beta^{(s)}_{k,t}(w) and display the top 5.
No log ratios, no beta*deviation scores, no phase averages.

Outputs (in --out_dir):
  - source_local_topic0_all_bins.csv         : audit of all source-local words
  - source_local_topic0_compact_table.md     : 3 sources x 6 displayed bins
  - source_local_topic0_compact_figure.png   : same content as figure
  - source_local_topic0_caption.txt          : LaTeX caption
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

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

SOURCE_NAMES = ['COHA', 'HBR', 'ILR']
TOPIC_K = 0

K_TOP = 30                 # top-K cutoff for set difference (use 30 or 50)
DISPLAY_TOPN = 5           # words shown per cell in compact table/figure
DISPLAYED_YEARS = [1922, 1942, 1962, 1982, 2002, 2017]


# ---- helpers --------------------------------------------------------------
def time_year_labels(T):
    return [1922 + 5 * i for i in range(T)]


def vocab_to_idx2w(vocab, V):
    if isinstance(vocab, list):
        return list(vocab)
    sample_key = next(iter(vocab))
    if isinstance(sample_key, str):
        return [w for w, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    return [vocab[i] for i in range(V)]


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
    print(f"[load] K={model.num_topics} T={model.num_times} "
          f"V={model.vocab_size} S={model.num_sources}")
    return model, vocab


@torch.no_grad()
def extract_betas(model):
    alpha_global, _ = model.get_alpha()
    beta_shared = model.get_beta(alpha_global).cpu().numpy()  # (K, T, V)
    betas_src = []
    for s in range(model.num_sources):
        b = model.get_beta_source(s, alpha_global).cpu().numpy()
        betas_src.append(b)
    return beta_shared, np.stack(betas_src, axis=0)            # (S, K, T, V)


# ---- core selection -------------------------------------------------------
def top_k_indices(beta_row, K):
    """Return indices of top-K entries in descending order of beta_row."""
    cand = np.argpartition(-beta_row, min(K, len(beta_row) - 1))[:K]
    return cand[np.argsort(-beta_row[cand])]


def select_source_local(beta_shared, betas_src, idx2w, years, K=K_TOP):
    """For every (source, bin), compute the source-local top words.

    Returns:
      records (list of dicts): audit rows for ALL source-local words.
      per_bin (dict): (src_name, year) -> list[(word_idx, beta, source_rank)]
                      sorted by beta desc; only source-local words.
    """
    records = []
    per_bin = {}

    for s_id, src in enumerate(SOURCE_NAMES):
        for t_idx, yr in enumerate(years):
            beta_s = betas_src[s_id, TOPIC_K, t_idx, :]
            beta_0 = beta_shared[TOPIC_K, t_idx, :]

            src_top = top_k_indices(beta_s, K)             # ranked source top-K
            shared_top = set(top_k_indices(beta_0, K).tolist())
            src_rank_of = {int(i): r + 1 for r, i in enumerate(src_top)}

            # source-local = in source top-K, NOT in shared top-K
            local = [int(i) for i in src_top if int(i) not in shared_top]

            # Already sorted by beta_s desc (since src_top is sorted)
            entries = [(i, float(beta_s[i]), src_rank_of[i]) for i in local]
            per_bin[(src, yr)] = entries

            for rank, (i, b, src_rank) in enumerate(entries, start=1):
                records.append({
                    'topic': TOPIC_K, 'source': src, 'year': yr, 'rank': rank,
                    'word': idx2w[i], 'beta': b,
                    'source_rank': src_rank, 'in_shared_topk': False,
                })

    return records, per_bin


# ---- outputs --------------------------------------------------------------
def write_audit_csv(records, out_path):
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Topic', 'Source', 'Year', 'Rank', 'Word',
                    'Beta', 'SourceRank', 'InSharedTopK'])
        for r in records:
            w.writerow([r['topic'], r['source'], r['year'], r['rank'], r['word'],
                        f"{r['beta']:.6e}", r['source_rank'],
                        'False' if not r['in_shared_topk'] else 'True'])
    print(f"[csv]  wrote {out_path}")


def _cell_words(per_bin, idx2w, src, year, top_n):
    return [idx2w[i] for (i, _b, _r) in per_bin.get((src, year), [])[:top_n]]


def write_markdown_table(per_bin, idx2w, displayed_years, top_n, out_path):
    header = "| Source | " + " | ".join(str(y) for y in displayed_years) + " |"
    sep    = "|--------|" + "|".join("---" for _ in displayed_years) + "|"
    lines = [header, sep]
    for src in SOURCE_NAMES:
        cells = []
        for y in displayed_years:
            words = _cell_words(per_bin, idx2w, src, y, top_n)
            cells.append(", ".join(words) if words else "—")
        lines.append(f"| {src} | " + " | ".join(cells) + " |")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[md]   wrote {out_path}")


def plot_compact_figure(per_bin, idx2w, displayed_years, top_n, out_path):
    n_rows = len(SOURCE_NAMES)
    n_cols = len(displayed_years)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.0 * n_cols + 1.2, 1.0 * top_n * n_rows + 1.2),
        constrained_layout=False,
    )
    if n_rows == 1:
        axes = np.array([axes])

    for r_i, src in enumerate(SOURCE_NAMES):
        for c_i, yr in enumerate(displayed_years):
            ax = axes[r_i, c_i]
            ax.set_xlim(0, 1)
            ax.set_ylim(0, top_n + 1)
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ('top', 'right', 'left', 'bottom'):
                ax.spines[spine].set_linewidth(0.5)
            if r_i == 0:
                ax.set_title(str(yr), fontsize=10)
            if c_i == 0:
                ax.set_ylabel(src, fontsize=11, rotation=0,
                              ha='right', va='center', labelpad=18)
            for rank, (i, b, _sr) in enumerate(
                    per_bin.get((src, yr), [])[:top_n], start=1):
                ax.text(0.5, rank, idx2w[i],
                        ha='center', va='center', fontsize=10)

    fig.suptitle(f'Topic 0 source-local top words '
                 f'(top-{top_n} of source top-{K_TOP} not in shared top-{K_TOP})',
                 fontsize=12, y=0.995)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88,
                        bottom=0.04, wspace=0.08, hspace=0.25)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def write_caption(out_path):
    cap = (
        r"\caption{This table reports source-local top words for Topic 0. "
        r"Words are selected from the source-specific top-" rf"{K_TOP} "
        r"topic words after removing words that also appear in the shared-"
        r"backbone top-" rf"{K_TOP} " r"list for the same time bin. Within "
        r"each cell we list the top-" rf"{DISPLAY_TOPN} " r"remaining words "
        r"ranked by $\beta^{(s)}_{0,t}(w)$. These words are not strictly "
        r"unique to a corpus; they are top words in a source-specific "
        r"realization of the topic that are not part of the shared top-word "
        r"list.}"
    )
    out_path.write_text(cap + '\n')
    print(f"[caps] wrote {out_path}")


# ---- main -----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=DEFAULT_CKPT)
    ap.add_argument('--vocab',      default=DEFAULT_VOCAB)
    ap.add_argument('--out_dir',    default=DEFAULT_OUT)
    ap.add_argument('--top_k',      type=int, default=K_TOP,
                    help='K for top-K set difference (e.g. 30 or 50)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cpu')

    model, vocab = load_model_and_vocab(args.checkpoint, args.vocab, device)
    beta_shared, betas_src = extract_betas(model)
    K, T, V = beta_shared.shape
    years = time_year_labels(T)
    idx2w = vocab_to_idx2w(vocab, V)
    print(f"[info] K_topics={K} T={T} V={V} years {years[0]}..{years[-1]} "
          f"K_TOP={args.top_k}")

    records, per_bin = select_source_local(beta_shared, betas_src,
                                           idx2w, years, K=args.top_k)

    # Console preview for displayed bins
    print(f"\n[Topic {TOPIC_K}] top-{DISPLAY_TOPN} source-local words "
          f"per displayed bin (K={args.top_k}):")
    for src in SOURCE_NAMES:
        print(f"  -- {src} --")
        for y in DISPLAYED_YEARS:
            words = _cell_words(per_bin, idx2w, src, y, DISPLAY_TOPN)
            print(f"    {y}: {', '.join(words) if words else '(none)'}")

    write_audit_csv(records, out_dir / 'source_local_topic0_all_bins.csv')
    write_markdown_table(per_bin, idx2w, DISPLAYED_YEARS, DISPLAY_TOPN,
                         out_dir / 'source_local_topic0_compact_table.md')
    plot_compact_figure(per_bin, idx2w, DISPLAYED_YEARS, DISPLAY_TOPN,
                        out_dir / 'source_local_topic0_compact_figure.png')
    write_caption(out_dir / 'source_local_topic0_caption.txt')


if __name__ == '__main__':
    main()
