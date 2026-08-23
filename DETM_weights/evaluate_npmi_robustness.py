#!/usr/bin/env python
"""
NPMI robustness check for all 5 conditions (A, B, C, D, E).

For every (model, source) pair:
  - Extract beta at each time slice (source-specific for D, global otherwise)
  - Use the SAME reference docs (single corpus, that source's docs only)
  - Compute TD (top-25), TC = NPMI (top-10), TQ = TD * TC

Reference choice: per-source single-corpus reference (most fair across A/B/D/E,
because A/B are trained on a single corpus already).
For C-ALL the global beta is one set of topics, so we report per-source TC
using each source's own docs (so C is also directly comparable).

Usage examples are at the bottom of this file (see __main__).
"""
from pathlib import Path

import torch
import numpy as np
import argparse
import json
import sys
import os
from collections import Counter
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data
from utils import get_topic_coherence_npmi

_CV_IMPORT_WARNING_SHOWN = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _norm_src(x):
    # Unwrap 1-element arrays/lists e.g. ['COHA']
    while True:
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) == 0:
                return ''
            x = x[0]
        else:
            break
    if isinstance(x, bytes):
        x = x.decode()
    return str(x).strip().upper()


def _diversity(beta_t, num_top):
    K = beta_t.shape[0]
    seen = set()
    for k in range(K):
        seen.update(np.argsort(beta_t[k])[-num_top:].tolist())
    return len(seen) / (K * num_top)


def _normalize_doc_tokens(doc):
    if torch.is_tensor(doc):
        doc = doc.detach().cpu().view(-1).tolist()
    elif isinstance(doc, np.ndarray):
        doc = doc.reshape(-1).tolist()
    elif np.isscalar(doc):
        doc = [doc]
    else:
        doc = list(doc)
    return [int(x) for x in doc]


def _top_word_ids(beta_t, top_n):
    arr = np.asarray(beta_t)
    if np.issubdtype(arr.dtype, np.integer):
        return arr[:, :top_n].astype(np.int64)
    return np.argsort(arr, axis=1)[:, -top_n:][:, ::-1].astype(np.int64)


def _cooccurrence_counts(top_ids, docs):
    needed_words = set(top_ids.reshape(-1).tolist())
    df = Counter()
    pair_df = Counter()
    for doc in docs:
        doc_set = set(w for w in _normalize_doc_tokens(doc) if w in needed_words)
        for w in doc_set:
            df[w] += 1
        for wi, wj in combinations(sorted(doc_set), 2):
            pair_df[(wi, wj)] += 1
    return df, pair_df


def get_topic_coherence_umass(beta_t, docs, top_n=10, eps=1.0):
    top_ids = _top_word_ids(beta_t, top_n)
    if len(docs) == 0:
        return float('nan'), []
    df, pair_df = _cooccurrence_counts(top_ids, docs)
    return get_topic_coherence_umass_from_counts(top_ids, df, pair_df, eps=eps)


def get_topic_coherence_umass_from_counts(top_ids, df, pair_df, eps=1.0):
    topic_scores = []
    for topic in top_ids:
        scores = []
        words = topic.tolist()
        for i in range(1, len(words)):
            for j in range(i):
                pair_key = tuple(sorted((int(words[i]), int(words[j]))))
                denom = max(float(df[int(words[j])]), eps)
                scores.append(float(np.log((pair_df[pair_key] + eps) / denom)))
        topic_scores.append(float(np.mean(scores)) if scores else float('nan'))
    return float(np.nanmean(topic_scores)), topic_scores


def prepare_cv_reference(docs, vocab):
    global _CV_IMPORT_WARNING_SHOWN
    if vocab is None or len(docs) == 0:
        return None, None
    try:
        from gensim.corpora import Dictionary
    except Exception as e:
        if not _CV_IMPORT_WARNING_SHOWN:
            print(f"WARNING: C_V coherence unavailable because gensim could not be imported: {e}")
            _CV_IMPORT_WARNING_SHOWN = True
        return None, None

    texts = []
    for doc in docs:
        words = []
        for token_id in _normalize_doc_tokens(doc):
            if 0 <= token_id < len(vocab):
                words.append(str(vocab[token_id]))
        if words:
            texts.append(words)
    if len(texts) == 0:
        return None, None
    try:
        return texts, Dictionary(texts)
    except Exception as e:
        if not _CV_IMPORT_WARNING_SHOWN:
            print(f"WARNING: C_V coherence unavailable because dictionary construction failed: {e}")
            _CV_IMPORT_WARNING_SHOWN = True
        return None, None


def get_topic_coherence_cv_from_reference(beta_t, vocab, texts, dictionary, top_n=10):
    global _CV_IMPORT_WARNING_SHOWN
    if vocab is None or texts is None or dictionary is None:
        return float('nan')
    try:
        from gensim.models import CoherenceModel
    except Exception as e:
        if not _CV_IMPORT_WARNING_SHOWN:
            print(f"WARNING: C_V coherence unavailable because gensim could not be imported: {e}")
            _CV_IMPORT_WARNING_SHOWN = True
        return float('nan')

    top_ids = _top_word_ids(beta_t, top_n)
    topics = []
    for topic in top_ids:
        words = [str(vocab[int(token_id)]) for token_id in topic if 0 <= int(token_id) < len(vocab)]
        if words:
            topics.append(words)
    if len(topics) == 0:
        return float('nan')

    try:
        cm = CoherenceModel(
            topics=topics,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v',
            processes=1
        )
        return float(cm.get_coherence())
    except Exception as e:
        if not _CV_IMPORT_WARNING_SHOWN:
            print(f"WARNING: C_V coherence unavailable because computation failed: {e}")
            _CV_IMPORT_WARNING_SHOWN = True
        return float('nan')


def get_topic_coherence_cv(beta_t, docs, vocab, top_n=10):
    texts, dictionary = prepare_cv_reference(docs, vocab)
    return get_topic_coherence_cv_from_reference(beta_t, vocab, texts, dictionary, top_n=top_n)


def npmi_penalty_diagnostics(beta_t, docs, top_n):
    top_ids = _top_word_ids(beta_t, top_n)
    if len(docs) == 0:
        return {
            f'zero_pair_rate@{top_n}': float('nan'),
            f'avg_pair_count@{top_n}': float('nan'),
        }
    _, pair_df = _cooccurrence_counts(top_ids, docs)
    return npmi_penalty_diagnostics_from_counts(top_ids, pair_df, top_n)


def npmi_penalty_diagnostics_from_counts(top_ids, pair_df, top_n):
    counts = []
    for topic in top_ids:
        for wi, wj in combinations(topic.tolist(), 2):
            pair_key = tuple(sorted((int(wi), int(wj))))
            counts.append(int(pair_df[pair_key]))
    if not counts:
        return {
            f'zero_pair_rate@{top_n}': float('nan'),
            f'avg_pair_count@{top_n}': float('nan'),
        }
    counts = np.asarray(counts, dtype=float)
    return {
        f'zero_pair_rate@{top_n}': float(np.mean(counts == 0)),
        f'avg_pair_count@{top_n}': float(np.mean(counts)),
    }


def _nanmean(values):
    return float(np.nanmean(values)) if len(values) > 0 else float('nan')


def load_model_cpu(ckpt_path):
    print(f"  Loading: {ckpt_path}", flush=True)
    with open(ckpt_path, 'rb') as f:
        m = torch.load(f, map_location='cpu', weights_only=False)
    print("  Loaded checkpoint; moving model to CPU/eval", flush=True)
    m = m.cpu().eval()
    for p in m.parameters():
        p.data = p.data.cpu()
    if hasattr(m, 'mu_q_alpha'):
        m.mu_q_alpha = m.mu_q_alpha.cpu()
    if hasattr(m, 'logsigma_q_alpha'):
        m.logsigma_q_alpha = m.logsigma_q_alpha.cpu()
    print("  Model ready", flush=True)
    return m


def get_beta_global(model):
    """Return global beta [K, T, V] from model."""
    print("  Extracting global beta", flush=True)
    with torch.no_grad():
        alpha, _ = model.get_alpha()
        beta = model.get_beta(alpha).cpu().numpy()
    print(f"  beta shape={beta.shape}", flush=True)
    return beta


def get_beta_source(model, src_id):
    """Return source-specific beta [K, T, V] from D-* model."""
    print(f"  Extracting source beta for src_id={src_id}", flush=True)
    with torch.no_grad():
        alpha, _ = model.get_alpha()
        beta = model.get_beta_source(src_id, alpha).cpu().numpy()
    print(f"  beta shape={beta.shape}", flush=True)
    return beta


# ---------------------------------------------------------------------------
# Reference doc loaders (one per condition style)
# ---------------------------------------------------------------------------
def load_reference_docs(data_dir):
    """Load merged data once. Returns vocab, train tokens/times/src_ids."""
    print(f"\nLoading data from: {data_dir}")
    vocab, train, _, _ = data.get_data(data_dir, temporal=True)
    train_tokens = train['tokens']
    train_times = np.asarray(train['times'])
    if 'sources' in train:
        srcs_norm = [_norm_src(s) for s in train['sources']]
        name2id = {'COHA': 0, 'HBR': 1, 'ILR': 2}
        train_src_ids = np.array([name2id.get(s, -1) for s in srcs_norm])
    else:
        train_src_ids = None
    print(f"  vocab={len(vocab)}  docs={len(train_tokens)}  has_src_labels={train_src_ids is not None}")
    return vocab, train_tokens, train_times, train_src_ids


def get_docs_for(train_tokens, train_times, train_src_ids, t, src_id=None):
    """Return list of token lists for time slice t (optionally filtered by src_id)."""
    if src_id is not None and train_src_ids is not None:
        mask = (train_times == t) & (train_src_ids == src_id)
    else:
        mask = (train_times == t)
    idx = np.where(mask)[0]
    return [train_tokens[i] for i in idx]


# ---------------------------------------------------------------------------
# Per-source TD/TC/TQ given a beta and the reference loader
# ---------------------------------------------------------------------------
def per_source_metrics(beta, train_tokens, train_times, train_src_ids,
                       src_id, vocab=None, num_tops_div=25, num_tops_coh=10,
                       compute_diagnostics=False, compute_cv=True, compute_umass=True):
    """D-ETM-style topic quality.

    For each time slice t:
        - TD_t  = topic diversity over top-`num_tops_div` of beta[:, t, :]
        - TC_t  = NPMI coherence (top-`num_tops_coh`) of beta[:, t, :], scored
                  against the FULL reference corpus (this source's full train
                  tokens), NOT the docs from time slice t. This matches the
                  D-ETM repo, which loops time-specific betas but always
                  evaluates coherence against `train_tokens`.
    Returns aggregate TD = mean_t TD_t, TC = mean_t TC_t, TQ = TD * TC.
    """
    K, T, V = beta.shape

    # Build the SINGLE full reference corpus once (filtered by source if given).
    if src_id is not None and train_src_ids is not None:
        idx = np.where(train_src_ids == src_id)[0]
    else:
        idx = np.arange(len(train_tokens))
    ref_docs = [train_tokens[i] for i in idx]
    print(f"  Computing metrics for ref_docs={len(ref_docs)} diagnostics={compute_diagnostics}", flush=True)
    if compute_cv:
        cv_texts, cv_dictionary = prepare_cv_reference(ref_docs, vocab)
        if cv_texts is not None:
            print(f"  Prepared C_V reference docs={len(cv_texts)}", flush=True)
    else:
        cv_texts, cv_dictionary = None, None

    td_per_t, tc_per_t, umass_per_t, cv_per_t = [], [], [], []
    zero10_per_t, zero15_per_t, avg_count10_per_t = [], [], []
    for t in range(T):
        print(f"    time {t + 1}/{T}", flush=True)
        beta_t = beta[:, t, :]
        td_per_t.append(_diversity(beta_t, num_tops_div))
        if len(ref_docs) == 0:
            tc_per_t.append(float('nan'))
            umass_per_t.append(float('nan'))
            cv_per_t.append(float('nan'))
            zero10_per_t.append(float('nan'))
            zero15_per_t.append(float('nan'))
            avg_count10_per_t.append(float('nan'))
            continue
        npmi_t, _ = get_topic_coherence_npmi(beta_t, ref_docs, top_n=num_tops_coh)
        tc_per_t.append(npmi_t)
        if compute_cv:
            cv_t = get_topic_coherence_cv_from_reference(beta_t, vocab, cv_texts, cv_dictionary,
                                                         top_n=num_tops_coh)
            cv_per_t.append(cv_t)
        if compute_umass or compute_diagnostics:
            top_n_counts = 15 if compute_diagnostics else num_tops_coh
            top_ids = _top_word_ids(beta_t, top_n_counts)
            df, pair_df = _cooccurrence_counts(top_ids, ref_docs)
            top_ids_coh = top_ids[:, :num_tops_coh]
            if compute_umass:
                umass_t, _ = get_topic_coherence_umass_from_counts(top_ids_coh, df, pair_df)
                umass_per_t.append(umass_t)
            if compute_diagnostics:
                top_ids10 = top_ids[:, :10]
                diag10 = npmi_penalty_diagnostics_from_counts(top_ids10, pair_df, top_n=10)
                diag15 = npmi_penalty_diagnostics_from_counts(top_ids, pair_df, top_n=15)
                zero10_per_t.append(diag10['zero_pair_rate@10'])
                zero15_per_t.append(diag15['zero_pair_rate@15'])
                avg_count10_per_t.append(diag10['avg_pair_count@10'])
    td = _nanmean(td_per_t)
    tc = _nanmean(tc_per_t)
    cv = _nanmean(cv_per_t)
    return {
        'TD': td,
        'TC': tc,
        'UMass': _nanmean(umass_per_t),
        'C_V': cv,
        'TQ': td * tc,
        'zero_pair_rate@10': _nanmean(zero10_per_t),
        'zero_pair_rate@15': _nanmean(zero15_per_t),
        'avg_pair_count@10': _nanmean(avg_count10_per_t),
        'n_ref_docs': int(len(ref_docs)),
    }


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------
def eval_condition_single(label, ckpt, data_dir, src_id, src_name,
                          compute_diagnostics=False, compute_cv=True, compute_umass=True):
    """A-*, B-*, E-*: single-corpus model. data_dir is that corpus's data dir.
    Use global beta. Reference docs = all train docs of that data dir at time t.
    For B-*/E-* the data_dir is the merged dir but the ckpt was trained on
    a filtered subset; we re-filter the reference to be that source only."""
    vocab, tokens, times, sids = load_reference_docs(data_dir)
    model = load_model_cpu(ckpt)
    beta = get_beta_global(model)

    # If src_ids exist, filter to that source for the reference. Otherwise
    # the data_dir is single-corpus already so no filter needed.
    use_src = src_id if (sids is not None) else None
    m = per_source_metrics(beta, tokens, times, sids, src_id=use_src, vocab=vocab,
                           compute_diagnostics=compute_diagnostics,
                           compute_cv=compute_cv,
                           compute_umass=compute_umass)
    print(f"\n[{label}-{src_name}]  TD={m['TD']:.4f}  "
          f"TC={m['TC']:+.4f}  UMass={m['UMass']:+.4f}  C_V={m['C_V']:+.4f}  TQ={m['TQ']:+.4f}  "
          f"zero@10={m['zero_pair_rate@10']:.3f}  zero@15={m['zero_pair_rate@15']:.3f}  "
          f"avgcnt@10={m['avg_pair_count@10']:.2f}  "
          f"(ref_docs={m['n_ref_docs']})")
    return {'condition': label, 'source': src_name,
            **m, 'checkpoint': ckpt}


def eval_condition_C(ckpt, data_dir, ref='source',
                     compute_diagnostics=False, compute_cv=True, compute_umass=True):
    """C-ALL: joint model with global beta.
    ref='source' -> per-source single-corpus reference (comparable to D/E)
    ref='all'    -> all corpora at time t as reference (original merged TC)
    """
    vocab, tokens, times, sids = load_reference_docs(data_dir)
    model = load_model_cpu(ckpt)
    beta = get_beta_global(model)
    out = []
    if ref == 'all':
        m = per_source_metrics(beta, tokens, times, sids, src_id=None, vocab=vocab,
                               compute_diagnostics=compute_diagnostics,
                               compute_cv=compute_cv,
                               compute_umass=compute_umass)
        print(f"\n[C-ALL ref=all]  TD={m['TD']:.4f}  "
              f"TC={m['TC']:+.4f}  UMass={m['UMass']:+.4f}  C_V={m['C_V']:+.4f}  TQ={m['TQ']:+.4f}  "
              f"zero@10={m['zero_pair_rate@10']:.3f}  zero@15={m['zero_pair_rate@15']:.3f}  "
              f"avgcnt@10={m['avg_pair_count@10']:.2f}  (ref_docs={m['n_ref_docs']})")
        out.append({'condition': 'C-ALL', 'source': 'ALL',
                    **m, 'checkpoint': ckpt})
        return out
    for sid, sname in enumerate(['COHA', 'HBR', 'ILR']):
        m = per_source_metrics(beta, tokens, times, sids, src_id=sid, vocab=vocab,
                               compute_diagnostics=compute_diagnostics,
                               compute_cv=compute_cv,
                               compute_umass=compute_umass)
        print(f"\n[C-ALL view from {sname}]  TD={m['TD']:.4f}  "
              f"TC={m['TC']:+.4f}  UMass={m['UMass']:+.4f}  C_V={m['C_V']:+.4f}  TQ={m['TQ']:+.4f}  "
              f"zero@10={m['zero_pair_rate@10']:.3f}  zero@15={m['zero_pair_rate@15']:.3f}  "
              f"avgcnt@10={m['avg_pair_count@10']:.2f}  (ref_docs={m['n_ref_docs']})")
        out.append({'condition': 'C-ALL', 'source': sname,
                    **m, 'checkpoint': ckpt})
    return out


def eval_condition_D(ckpt, data_dir, compute_diagnostics=False, compute_cv=True, compute_umass=True):
    """D-*: source adaptation. beta = beta_source(s). Reference = source s docs."""
    vocab, tokens, times, sids = load_reference_docs(data_dir)
    model = load_model_cpu(ckpt)
    out = []
    for sid, sname in enumerate(['COHA', 'HBR', 'ILR']):
        beta = get_beta_source(model, sid)
        m = per_source_metrics(beta, tokens, times, sids, src_id=sid, vocab=vocab,
                               compute_diagnostics=compute_diagnostics,
                               compute_cv=compute_cv,
                               compute_umass=compute_umass)
        print(f"\n[D-{sname}]  TD={m['TD']:.4f}  "
              f"TC={m['TC']:+.4f}  UMass={m['UMass']:+.4f}  C_V={m['C_V']:+.4f}  TQ={m['TQ']:+.4f}  "
              f"zero@10={m['zero_pair_rate@10']:.3f}  zero@15={m['zero_pair_rate@15']:.3f}  "
              f"avgcnt@10={m['avg_pair_count@10']:.2f}  (ref_docs={m['n_ref_docs']})")
        out.append({'condition': 'D', 'source': sname,
                    **m, 'checkpoint': ckpt})
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True,
                    choices=['A', 'B', 'C', 'D', 'E', 'all'])
    ap.add_argument('--data_dir', help='Reference data dir (per-condition)')
    ap.add_argument('--checkpoint', help='Single ckpt (modes C, D)')
    ap.add_argument('--src_name', help='COHA/HBR/ILR (modes A, B, E)')

    # For mode A / B: single-corpus model + that corpus\'s data dir
    # For mode E: merged data dir + per-corpus ckpt
    ap.add_argument('--coha_ckpt')
    ap.add_argument('--hbr_ckpt')
    ap.add_argument('--ilr_ckpt')
    ap.add_argument('--coha_data_dir')
    ap.add_argument('--hbr_data_dir')
    ap.add_argument('--ilr_data_dir')

    # For mode 'all': pass paths via JSON config
    ap.add_argument('--config', help='JSON config with all paths')

    ap.add_argument('--ref', choices=['source', 'all'], default='source',
                    help='For mode C: per-source reference (default) or all-corpora reference')
    ap.add_argument('--diagnostics', action='store_true',
                    help='Compute zero-pair NPMI diagnostics. Disabled by default because it adds extra co-occurrence work.')
    ap.add_argument('--skip-cv', action='store_true',
                    help='Skip C_V coherence entirely. Use this to quickly reproduce TD/TC/TQ.')
    ap.add_argument('--skip-umass', action='store_true',
                    help='Skip UMass coherence entirely. Use with --skip-cv to reproduce TD/TC/TQ only.')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    results = []

    if args.mode == 'C':
        if not args.checkpoint:
            ap.error("--checkpoint is required for mode C")
        results = eval_condition_C(args.checkpoint, args.data_dir, ref=args.ref,
                                   compute_diagnostics=args.diagnostics,
                                   compute_cv=not args.skip_cv,
                                   compute_umass=not args.skip_umass)

    elif args.mode == 'D':
        if not args.checkpoint:
            ap.error("--checkpoint is required for mode D")
        results = eval_condition_D(args.checkpoint, args.data_dir,
                                   compute_diagnostics=args.diagnostics,
                                   compute_cv=not args.skip_cv,
                                   compute_umass=not args.skip_umass)

    elif args.mode in ('A', 'B'):
        if not any([args.coha_ckpt, args.hbr_ckpt, args.ilr_ckpt]):
            ap.error(f"mode {args.mode} requires at least one of --coha_ckpt/--hbr_ckpt/--ilr_ckpt")
        # Single-corpus models with per-source ckpt + per-source data dir
        for sname, ck, dd in [
            ('COHA', args.coha_ckpt, args.coha_data_dir),
            ('HBR',  args.hbr_ckpt,  args.hbr_data_dir),
            ('ILR',  args.ilr_ckpt,  args.ilr_data_dir),
        ]:
            if ck is None or dd is None:
                continue
            sid = {'COHA': 0, 'HBR': 1, 'ILR': 2}[sname]
            results.append(eval_condition_single(args.mode, ck, dd, sid, sname,
                                                 compute_diagnostics=args.diagnostics,
                                                 compute_cv=not args.skip_cv,
                                                 compute_umass=not args.skip_umass))

    elif args.mode == 'E':
        if not any([args.coha_ckpt, args.hbr_ckpt, args.ilr_ckpt]):
            ap.error("mode E requires at least one of --coha_ckpt/--hbr_ckpt/--ilr_ckpt")
        # E uses MERGED data dir but each ckpt was trained with source_filter.
        # Reference = that source\'s docs in the merged dir.
        merged_dir = args.data_dir
        for sname, ck in [('COHA', args.coha_ckpt),
                          ('HBR',  args.hbr_ckpt),
                          ('ILR',  args.ilr_ckpt)]:
            if ck is None:
                continue
            sid = {'COHA': 0, 'HBR': 1, 'ILR': 2}[sname]
            results.append(eval_condition_single('E', ck, merged_dir, sid, sname,
                                                 compute_diagnostics=args.diagnostics,
                                                 compute_cv=not args.skip_cv,
                                                 compute_umass=not args.skip_umass))

    elif args.mode == 'all':
        cfg = json.load(open(args.config))
        # cfg = {'A': {'COHA':{'ckpt':..., 'data_dir':...}, ...},
        #        'B': {...}, 'C': {'ckpt':..., 'data_dir':...},
        #        'D': {'ckpt':..., 'data_dir':...},
        #        'E': {'data_dir':..., 'COHA':'ckpt', 'HBR':'ckpt', 'ILR':'ckpt'}}
        for mode in ('A', 'B'):
            for sname, sub in cfg.get(mode, {}).items():
                sid = {'COHA': 0, 'HBR': 1, 'ILR': 2}[sname]
                results.append(eval_condition_single(mode, sub['ckpt'],
                                                     sub['data_dir'], sid, sname,
                                                     compute_diagnostics=args.diagnostics,
                                                     compute_cv=not args.skip_cv,
                                                     compute_umass=not args.skip_umass))
        if 'C' in cfg:
            results += eval_condition_C(cfg['C']['ckpt'], cfg['C']['data_dir'],
                                        ref=args.ref,
                                        compute_diagnostics=args.diagnostics,
                                        compute_cv=not args.skip_cv,
                                        compute_umass=not args.skip_umass)
        if 'D' in cfg:
            results += eval_condition_D(cfg['D']['ckpt'], cfg['D']['data_dir'],
                                        compute_diagnostics=args.diagnostics,
                                        compute_cv=not args.skip_cv,
                                        compute_umass=not args.skip_umass)
        if 'E' in cfg:
            md = cfg['E']['data_dir']
            for sname in ('COHA', 'HBR', 'ILR'):
                if sname in cfg['E']:
                    sid = {'COHA': 0, 'HBR': 1, 'ILR': 2}[sname]
                    results.append(eval_condition_single('E',
                                                         cfg['E'][sname], md, sid, sname,
                                                         compute_diagnostics=args.diagnostics,
                                                         compute_cv=not args.skip_cv,
                                                         compute_umass=not args.skip_umass))

    # Print summary
    print("\n" + "=" * 96)
    if args.diagnostics:
        print(f"{'Cond':<6} {'Source':<6} {'TD':>8} {'TC(NPMI)':>11} {'UMass':>10} {'C_V':>8} {'TQ':>9} "
              f"{'zero@10':>9} {'zero@15':>9} {'avgcnt@10':>10} {'n_ref':>9}")
        print("-" * 105)
        for r in results:
            print(f"{r['condition']:<6} {r['source']:<6} "
                  f"{r['TD']:>8.4f} {r['TC']:>+11.4f} {r['UMass']:>+10.4f} {r['C_V']:>8.4f} {r['TQ']:>+9.4f} "
                  f"{r['zero_pair_rate@10']:>9.3f} {r['zero_pair_rate@15']:>9.3f} "
                  f"{r['avg_pair_count@10']:>10.2f} {r.get('n_ref_docs', 0):>9d}")
    else:
        print(f"{'Cond':<6} {'Source':<6} {'TD':>8} {'TC(NPMI)':>11} {'UMass':>10} {'C_V':>8} {'TQ':>9} {'n_ref':>9}")
        print("-" * 78)
        for r in results:
            print(f"{r['condition']:<6} {r['source']:<6} "
                  f"{r['TD']:>8.4f} {r['TC']:>+11.4f} {r['UMass']:>+10.4f} {r['C_V']:>8.4f} "
                  f"{r['TQ']:>+9.4f} {r.get('n_ref_docs', 0):>9d}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.output}")


if __name__ == '__main__':
    main()
