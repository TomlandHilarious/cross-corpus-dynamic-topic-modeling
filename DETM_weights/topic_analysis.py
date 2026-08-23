"""topic_analysis.py

Command‑line utility to compare and visualise *source‑specific* topic
word‑distributions produced by a Dynamic‑ETM model with LoRA offsets.

---------------------------------------------------------------
Core features
------------
1. **Distance report** – given a topic *k*, compute Jensen–Shannon (or KL)
   divergence between β_{k,t,src} of each non‑reference corpus and that of
   the reference (default: COHA) at every time‑slice.  Saves a CSV and
   prints a summary table.
2. **Top‑words chart** – bar chart of the top‑N words for the chosen topic
   in each corpus (most recent time by default, or user‑specified).
3. **Common‑word trajectories** – for words that appear in the *union* of
   the three top‑N lists, plot p(word | topic_k, t, src) over time for the
   three corpora.

Usage
-----
```bash
python topic_analysis.py \
    --ckpt PATH_TO_LORA_CKPT.pt \
    --data_path /path/to/corpus_root \
    --topic 5 \
    --top_n 10 \
    --ref COHA \
    --out_dir results/topic5
```

Dependencies: torch, numpy, pandas, matplotlib, scipy
"""

import argparse
import os
from pathlib import Path

import data
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy 
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle
from typing import Union, List, Optional         # ← add this

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen–Shannon divergence (base‑2, bounded in [0,1])."""
    p = (p + eps) / (p.sum() + eps * len(p))
    q = (q + eps) / (q.sum() + eps * len(q))
    return float(jensenshannon(p, q, base=2.0) ** 2)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = (p + eps) / (p.sum() + eps * len(p))
    q = (q + eps) / (q.sum() + eps * len(q))
    return float(entropy(p, q))


# -----------------------------------------------------------------------------
# Main analysis class
# -----------------------------------------------------------------------------
class TopicAnalyser:
    def __init__(self, ckpt_path: str, data_path: str, min_df: int, device: str = "cpu") -> None:
        self.device = device
        print(f"Loading model from {ckpt_path} …")
        model = torch.load(ckpt_path, map_location=device)
        
        if isinstance(model, torch.nn.Module):
            self.model = model.to(device)
        else:  # state_dict only
            from detm import DETM  # assuming same codebase in PYTHONPATH
            raise RuntimeError("state_dict‑only ckpt – please instantiate DETM first")
        self.model.eval()
        # meta
        self.src_names = ["COHA", "HBR", "ILR"]  # order used when training
        if hasattr(self.model, "vocab"):
            self.vocab = self.model.vocab
            print("Using vocab embedded in checkpoint")
        else:
            data_dir = Path(data_path) / f"min_df_{str(min_df)}"  # will come from CLI
            vocab_pkl = data_dir / "vocab.pkl"
            print(f"Loading vocab from {vocab_pkl}")
            with open(vocab_pkl, "rb") as f:
                self.vocab = pickle.load(f)
            vocab_loaded, _, _, _ = data.get_data(str(data_dir), temporal=True)
            ts_path = data_dir / "timestamps.pkl"
            with open(ts_path, "rb") as f:
                year_list = pickle.load(f)          # len == T
            self.year_arr = np.asarray(year_list)    
            if vocab_loaded != self.vocab:
                raise ValueError("Vocabulary mismatch between pickle and corpus splits")
        

    # ------------------------------------------------------------------
    def _beta_src(self, src_id: int) -> torch.Tensor:
        """Return β^{src}:  tensor shape (T, K, V)."""
        with torch.no_grad():
            alpha = self.model.mu_q_alpha           # (K, T, L)
            beta  = self.model.get_beta(alpha, src_id=src_id)  # (K, T, V)
        return beta.cpu()    # (K, T, V)

    def collect_betas(self):
        print("Collecting β for each corpus …")
        raw = {s: self._beta_src(i) for i, s in enumerate(self.src_names)}  # (K,T,V)
        self.beta_table = {s: b.permute(1, 0, 2).contiguous()
                       for s, b in raw.items()}

        # Cache common shapes
        any_beta       = next(iter(self.beta_table.values()))
        self.T, self.K, self.V = any_beta.shape

    # ------------------------------------------------------------------
    def distance_report(self, topic: int, ref: str = "COHA", metric: str = "JS") -> pd.DataFrame:
        assert hasattr(self, "beta_table"), "run collect_betas() first"
        ref_beta = self.beta_table[ref][:, topic, :].numpy()   # (T,V)
        rows = []
        for src, beta in self.beta_table.items():
            if src == ref:
                continue
            dist_func = js_divergence if metric.upper() == "JS" else kl_divergence
            dists = [dist_func(beta[:, topic, t].numpy(), ref_beta[t]) for t in range(self.T)]
            rows.append(pd.Series(dists, name=src))
        df = pd.concat(rows, axis=1)
        return df  # rows = time, cols = corpus vs ref

    # ------------------------------------------------------------------
    def plot_top_words(self, topic: int, top_n: int = 10, time_idx: int = -1, out_dir: Union[Path, str] = "plots"):
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        vocab = self.vocab  # expect the DETM object to carry vocab list

        fig, axes = plt.subplots(1, len(self.src_names), figsize=(4*len(self.src_names), 4))
        for ax, src in zip(axes, self.src_names):
            beta_tk = self.beta_table[src][topic, time_idx]     # (V,)
            top_ids  = torch.topk(beta_tk, top_n).indices.numpy()
            words    = [vocab[i] for i in top_ids]
            probs    = beta_tk[top_ids].numpy()
            ax.barh(range(top_n)[::-1], probs[::-1])
            ax.set_yticks(range(top_n)[::-1])
            ax.set_yticklabels(words[::-1])
            ax.set_title(src)
        fig.suptitle(f"Top {top_n} words – topic {topic} – time {time_idx}")
        fig.tight_layout()
        fig.savefig(out_dir / f"top_words_topic{topic}_t{time_idx}.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    def plot_word_trajectories(self, topic: int, top_n: int = 10, ref: str = "COHA", 
                               query_words: Optional[List[str]] = None,
                               out_dir: Union[Path, str] = "plots",
                               rel_to='COHA'):
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        vocab = self.vocab
        vocab2id = {w: i for i, w in enumerate(self.vocab)}
        if query_words:
            cand_ids = []
            for w in query_words:
                if w not in vocab2id:
                    print(f"[WARN] '{w}' not in vocab – skipped")
                else:
                    cand_ids.append(vocab2id[w])
            if not cand_ids:
                print("No valid query words – aborting plot.")
                return
        else:
        # words common to all corpora in their top‑N list at final time
            cand_ids = None
            for src in self.src_names:
                beta_last = self.beta_table[src][-1, topic]
                top_ids   = set(torch.topk(beta_last, top_n).indices.tolist())
                cand_ids = top_ids if cand_ids is None else cand_ids & top_ids
            if not cand_ids:
                print("No common words in top-N – aborting plot.")
                return
            cand_ids = sorted(cand_ids)
        time_axis = self.year_arr   
        for wid in cand_ids:
            word = vocab[wid]
            fig, ax = plt.subplots(figsize=(6, 4))
            base = self.beta_table[rel_to][topic, :, wid].numpy()
            for src in self.src_names:
                probs = self.beta_table[src][topic, :, wid].numpy()
                ratio = np.log((probs + 1e-12) / (base + 1e-12))


                ax.plot(time_axis, ratio, label=src)
            ax.set_title(f"{word} – topic {topic}")
            ax.set_xlabel("time index")
            ax.set_ylabel(f"log p(w| k, src, t) - log p(w|k, {rel_to}, t)")

            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10, integer=True))
            ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"traj_{word}_topic{topic}.png", dpi=150)
            plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Source‑aware topic analysis & visualisation")
    p.add_argument("--ckpt", required=True, help="Path to LoRA‑tuned DETM ckpt (.pt)")
    p.add_argument("--data_path", required=True, help="Corpus root – only needed if DETM class needs it")
    p.add_argument("--topic", type=int, required=True, help="Topic index k to analyse")
    p.add_argument("--top_n", type=int, default=10)
    p.add_argument("--ref", default="COHA", choices=["COHA", "HBR", "ILR"], help="Reference corpus for distance calc")
    p.add_argument("--metric", default="JS", choices=["JS", "KL"])
    p.add_argument("--out_dir", default="analysis_results")
    p.add_argument("--min_df", type=int, default=10, help="min-df used when training")
    p.add_argument("--show_global_top", action="store_true",
              help="Print / plot the global β (no LoRA) top-N words")
    p.add_argument("--global_times", type=int, nargs="*", default=[0, -1],
              help="Time indices to inspect in β_global; default first & last")
    p.add_argument("--words", nargs="+", default=None,
               help="User-specified query words (space-separated). "
                    "If omitted, use the intersection-of-topN heuristic.")


    return p.parse_args()


def main():
    args = parse_args()
    analyser = TopicAnalyser(args.ckpt, data_path=args.data_path, min_df=args.min_df, device="cpu")
    analyser.collect_betas()

    if args.show_global_top:
        with torch.no_grad():
            alpha = analyser.model.mu_q_alpha        # (K, T, L)

            beta_global = analyser.model.get_beta(alpha)        # (K,T,V)
            beta_global = beta_global.permute(1, 0, 2).cpu()    # ② → (T,K,V)

            # ─── ︙ ───
            # debug ↓
            beta_c = analyser.model.get_beta(alpha, src_id=0) 
            diff_norm = (beta_g - beta_c).norm().item()
            print(f"[DEBUG] ‖β_global – β_COHA‖ = {diff_norm:.6f}")  
            W_base = analyser.model.rho.weight                        # (V,L)
            delta0 = analyser.model.lora_B[0] @ analyser.model.lora_A[0]
            W_eff0 = W_base + delta0                                  # (V,L)
            w_diff = (W_eff0 - W_base).norm().item()
            print(f"[DEBUG] ‖ΔW_COHA‖            = {delta0.norm():.6f}")
            print(f"[DEBUG] ‖W_eff0 – W_base‖     = {w_diff:.6f}")
      
           
    top_n = args.top_n
    for t in args.global_times:
        
        t_idx = t if t >= 0 else analyser.T + t
        year  = analyser.year_arr[t_idx]
        gamma = beta_global[args.topic, t_idx]   # (V,)
        top_ids = torch.topk(gamma, args.top_n).indices 
        words   = [analyser.vocab[i] for i in top_ids]
        probs   = gamma[top_ids].numpy()
        print(f"\n[β_global] topic {args.topic}, time {t_idx} (year={year})")
        for w,pv in zip(words, probs):
            print(f"  {w:15s} {pv:.4f}")

        plt.figure(figsize=(4,3))
        plt.barh(range(top_n)[::-1], probs[::-1])
        plt.yticks(range(top_n)[::-1], words[::-1])
        plt.title(f"β_global  t={t_idx}")
        plt.tight_layout()
        (Path(args.out_dir)/f"beta_global_topic{args.topic}_t{t_idx}.png"
        ).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(Path(args.out_dir)/
                    f"beta_global_topic{args.topic}_t{t_idx}.png", dpi=150)
        plt.close()


    # 1) distance CSV & print
    df = analyser.distance_report(args.topic, args.ref, args.metric)
    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True, parents=True)
    csv_path = out_dir / f"dist_topic{args.topic}.csv"
    df.to_csv(csv_path, index_label="time_idx")
    print("\nJensen–Shannon (rows=time)\n", df.head())
    print(f"Saved full CSV to {csv_path}")

    # 2) bar‑chart top words at last time
    analyser.plot_top_words(args.topic, args.top_n, time_idx=-1, out_dir=out_dir)

    # 3) common‑word trajectories
    analyser.plot_word_trajectories(args.topic, args.top_n, query_words=args.words, out_dir=out_dir)


if __name__ == "__main__":
    main()
