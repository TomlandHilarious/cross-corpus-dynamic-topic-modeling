#!/usr/bin/env python
"""
Save per‑topic top words **and** report topic diversity.

Output CSV columns
    source   : GLOBAL | COHA | HBR | ILR
    topic    : topic index (0‑based)
    method   : UNION | AVG | REL
    keywords : comma‑separated top‑`top_n` words

After CSV is written the script prints a small summary table:
    ┌────────┬─────────┬────────────┐
    │ source │ method  │ diversity │
    └────────┴─────────┴────────────┘
where
    diversity = #unique(top_n words across K topics) / (K*top_n).
The closer to 1 the less overlap among topics.
"""
from pathlib import Path
import torch
import pandas as pd
from detm import DETM
from typing import List, Dict
from tqdm import tqdm
from collections import defaultdict

# -----------------------------------------------------------------------------
# paths & parameters
# -----------------------------------------------------------------------------
ckpt_path = f"{Path(__file__).resolve().parent.parent}/detm_weighted/topic_50_min_df_10_delta_{0.05}_time_20250520_162831/lora_rank_16/detm_merged_K_50_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_Lr_0.0_Bsz_500_RhoSize_300_L_3_minDF_10_trainEmbeddings_1_lora_r16.pt"
vocab_path = f"{Path(__file__).resolve().parent.parent}/merged_max_df_0.6/min_df_10/vocab.txt"
output_csv = "topic_keywords.csv"

# number of keywords per topic & per variant
top_n = 25
λ_relevance = 0.6               # weight for mean vs lift

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def load_vocab(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [w.strip() for w in f]


def collect_keywords_union(beta: torch.Tensor, vocab: List[str]) -> List[List[str]]:
    K, T, _ = beta.shape
    out: List[List[str]] = []
    for k in range(K):
        ids = set()
        for t in range(T):
            ids.update(beta[k, t].topk(top_n).indices.tolist())
        out.append([vocab[i] for i in list(ids)[:top_n]])
    return out


def collect_keywords_avg(beta: torch.Tensor, vocab: List[str]) -> List[List[str]]:
    beta_avg = beta.mean(dim=1)  # (K, V)
    out: List[List[str]] = []
    for row in beta_avg:
        ids = row.topk(top_n).indices.tolist()
        out.append([vocab[i] for i in ids])
    return out


def collect_keywords_rel(beta: torch.Tensor, bg_probs: torch.Tensor, vocab: List[str]) -> List[List[str]]:
    beta_avg = beta.mean(dim=1)
    lift = beta_avg / (bg_probs + 1e-12)
    score = λ_relevance * beta_avg + (1 - λ_relevance) * torch.log(lift + 1e-12)
    out: List[List[str]] = []
    for row in score:
        ids = row.topk(top_n).indices.tolist()
        out.append([vocab[i] for i in ids])
    return out


def topic_diversity(list_of_lists: List[List[str]]) -> float:
    """TD = |unique words| / (K * top_n)"""
    flat = [w for sub in list_of_lists for w in sub]
    return len(set(flat)) / len(flat)

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt_path, map_location=device)
    if not isinstance(state, DETM):
        raise RuntimeError("Checkpoint is not a full model object (state_dict not supported).")
    model: DETM = state.to(device).eval()

    with torch.no_grad():
        alpha = model.mu_q_alpha
        beta_global = model.get_beta(alpha)  # (K,T,V)
        src_names = ["COHA", "HBR", "ILR"]
        name2id = {s: i for i, s in enumerate(src_names)}
        beta_src: Dict[str, torch.Tensor] = {s: model.get_beta(alpha, src_id=name2id[s]) for s in src_names}

    vocab = load_vocab(vocab_path)

    # corpus background probs (for relevance)
    bg_counts = torch.cat([beta_global]).sum(dim=(0, 1))  # sum over K,T
    bg_probs = bg_counts / bg_counts.sum()

    collectors = {
        "UNION": lambda b: collect_keywords_union(b, vocab),
        "AVG":   lambda b: collect_keywords_avg(b, vocab),
        "REL":   lambda b: collect_keywords_rel(b, bg_probs, vocab),
    }

    keyword_records = []
    diversity_rows = []

    # --- global ---
    for m, fn in collectors.items():
        kw = fn(beta_global)
        for k, words in enumerate(tqdm(kw, desc=f"GLOBAL {m}")):
            keyword_records.append({"source": "GLOBAL", "topic": k, "method": m, "keywords": ", ".join(words)})
        diversity_rows.append(("GLOBAL", m, topic_diversity(kw)))

    # --- per source ---
    for src, b in beta_src.items():
        for m, fn in collectors.items():
            kw = fn(b)
            for k, words in enumerate(tqdm(kw, desc=f"{src} {m}", leave=False)):
                keyword_records.append({"source": src, "topic": k, "method": m, "keywords": ", ".join(words)})
            diversity_rows.append((src, m, topic_diversity(kw)))

    # CSV
    pd.DataFrame(keyword_records).to_csv(output_csv, index=False)
    print(f"[✓] Keyword CSV written → {output_csv}")

    # diversity summary
    df_div = pd.DataFrame(diversity_rows, columns=["source", "method", "diversity"]).sort_values(["source", "method"])
    print("\nTopic‑Diversity summary (unique(top_n)/(K*top_n))\n", df_div.to_string(index=False))


if __name__ == "__main__":
    main()
