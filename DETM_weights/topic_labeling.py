#!/usr/bin/env python
# coding: utf-8
"""
Generate unified topic labels for a dynamic-ETM checkpoint.

Usage
-----
python topic_labeling.py \
       --ckpt_path  path/to/detm_lora_r16.pt \
       --vocab_path path/to/vocab.txt        \
       --top_n      20                      \
       --out_dir    output folder
"""
from dotenv import load_dotenv
import argparse, os, json, torch
from tqdm import tqdm
from typing import List
from detm import DETM  
import pandas as pd
import textwrap
from google import genai
from scipy.special import rel_entr
from google.genai import types
import time
import numpy as np
# -------------------------------------------------- #
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
# model_name=""
EPS = 1e-12  # smoothing to avoid log0

def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen_Shannon divergence between two probability vectors (base‑2)."""
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(rel_entr(p, m))
    kl_qm = np.sum(rel_entr(q, m))
    return 0.5 * (kl_pm + kl_qm) / np.log(2)   # convert to bits (0‑1)



def load_vocab(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [w.strip() for w in f]

def collect_keywords(beta: torch.Tensor,
                     top_n: int,
                     vocab: List[str]) -> List[List[str]]:
    """
    beta : (K, T, V)
    Returns K lists of unique keywords (length ≤ top_n)
    """
    K, T, V = beta.shape
    keywords = []
    for k in range(K):
        ids = set()
        for t in range(T):
            ids |= set(beta[k, t].topk(top_n).indices.tolist())
        ids = list(ids)[: top_n]
        keywords.append([vocab[i] for i in ids])
    return keywords

def build_prompt(keywords: list[str]) -> str:
    """Return a formatted prompt for the LLM."""
    tpl = """
    You are a domain expert.
    Task: Given a list of keywords, propose a short (2-5 words)
    ENGLISH topic title that best summarizes the shared theme.

    Requirements:
    - Reflect the main idea covered by the keywords.
    - Use common academic or journalistic phrasing.
    - Avoid vague words like "miscellaneous" or "general".
    - Output **only** the title, no extra commentary.

    Example
    -------
    Keywords: stock, equity, dividend, market, portfolio
    Title: Stock Market Investing

    Now label this set:
    Keywords: {kw}
    Title:"""
    return textwrap.dedent(tpl).strip().format(kw=", ".join(keywords))


def ask_llm(keywords,
            api_key,
            model_name="gemini-2.0-flash",
            temperature=0.2):
    """
    keywords : list[str]  -  the word list to label
    api_key  : str        -  Gemini API key from Google AI Studio
    """
    

    # configure client (runs once per process)
    client = genai.Client(api_key=api_key)

    prompt = build_prompt(keywords)           # same as before
    
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt],
        config=types.GenerateContentConfig(
        max_output_tokens=10,
        temperature=0.2
        )
    )
    time.sleep(4.5)
    return response.text

def main(args):
    # ------ load model & beta ------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state  = torch.load(args.ckpt_path, map_location=device)
    if isinstance(state, DETM):
        model = state
    else:                 # saved with torch.save(model.state_dict())
        # Need to rebuild the DETM skeleton first (left as TODO)
        raise RuntimeError("Checkpoint is not a full model object")
    model.to(device).eval()
    with torch.no_grad():
        alpha = model.mu_q_alpha                       # (K, T, L)
        beta  = model.get_beta(alpha)                  # (K, T, V)
        # average over time to  (K, V)
        beta_global_avg = beta.mean(dim=1).cpu().numpy()
        src_names = ["COHA", "HBR", "ILR"]
        name2id   = {s: i for i, s in enumerate(src_names)}


        beta_src  = {
            s: model.get_beta(alpha, src_id=name2id[s])   # (K, T, V)
            for s in src_names
        }
        beta_src_avg = {s: b.mean(dim=1).cpu().numpy() for s, b in beta_src.items()}


    # ------ load vocab ------
    vocab = load_vocab(args.vocab_path)

    # ======================= 1. GLOBAL (merged) =======================
    kw_global = collect_keywords(beta, args.top_n, vocab)      # (K, list[str])
    labels_all = []
    for k, words in enumerate(tqdm(kw_global, desc="Global topics")):
        title = ask_llm(words, api_key=api_key, temperature=0.2)
        labels_all.append({
            "source":  "GLOBAL",
            "topic":   k,
            "title":   title,
            "keywords": ", ".join(words),
            "jsd_to_global": 0.0
        })

    # ======================= 2. PER-SOURCE (LoRA) =====================
    src_names = ["COHA", "HBR", "ILR"]
    name2id   = {s: i for i, s in enumerate(src_names)}
    beta_src  = {s: model.get_beta(alpha, src_id=name2id[s]) for s in src_names}

    for src, beta_s in beta_src.items():                       # beta_s: (K,T,V)
        kw_lists = collect_keywords(beta_s, args.top_n, vocab)
        for k, words in enumerate(tqdm(kw_lists, desc=f"{src} topics", leave=False)):
            title = ask_llm(words,
                        api_key=api_key,   
                        temperature=0.2)
            js = jsd(beta_global_avg[k] + EPS, beta_src_avg[src][k] + EPS)  # 0‑1 bits
            labels_all.append({
                "source":  src,
                "topic":   k,
                "title":   title,
                "keywords": ", ".join(words),
                "jsd_to_global": round(float(js), 4)
            })

    # ======================= 3. SAVE =======================
    os.makedirs(args.out_dir, exist_ok=True)
    all_path = os.path.join(args.out_dir, "topic_labels_all.csv")
    global_path = os.path.join(args.out_dir, "topic_labels_global.csv")
    finetune_path = os.path.join(args.out_dir, "topic_labels_per_source.csv")
    
    df = pd.DataFrame(labels_all)                #  (K·4) rows
    df.to_csv(all_path, index=False)
    df[df.source == "GLOBAL"].to_csv(global_path, index=False)
    df[df.source != "GLOBAL"].to_csv(finetune_path, index=False)
    print("All files saved.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path",  required=True)
    p.add_argument("--vocab_path", required=True)
    p.add_argument("--top_n",      type=int, default=20)
    p.add_argument("--out_dir",    required=True)
    args = p.parse_args()
    main(args)
