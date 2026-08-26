#!/usr/bin/env python

from __future__ import print_function

import argparse
import atexit
import time
from datetime import datetime
import torch
import pickle 
import numpy as np 
import os 
import math 
import random 
import sys
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.io

import data 
from scipy.spatial.distance import jensenshannon, cosine

from sklearn.decomposition import PCA
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd
from detm import DETM
from utils import nearest_neighbors, get_topic_coherence


# =========== Source-specific topic diagnostics ======= #
def get_source_topic_words(model, vocab, topic_id, time_id, num_words=10):
    """
    Get top words for a specific topic/time across all sources.
    
    Returns:
        dict: {src_name: [top_words]}
    """
    with torch.no_grad():
        alpha_global, _ = model.get_alpha()
        
        results = {}
        # Shared backbone
        beta_shared = model.get_beta(alpha_global)
        top_indices = beta_shared[topic_id, time_id].topk(num_words)[1].cpu().numpy()
        results['shared'] = [vocab[i] for i in top_indices]
        
        # Source-specific if in adaptation mode
        if model.source_adaptation_mode and model.delta_alpha is not None:
            for src_id in range(model.num_sources):
                beta_src = model.get_beta_source(src_id, alpha_global)
                top_indices = beta_src[topic_id, time_id].topk(num_words)[1].cpu().numpy()
                results[f'source_{src_id}'] = [vocab[i] for i in top_indices]
        
        return results

def compute_beta_distances(model, metric='js'):
    """
    Compute pairwise distances between source-specific betas.
    
    Args:
        model: DETM model
        metric: 'js' (Jensen-Shannon) or 'cosine'
    
    Returns:
        dict: {(src_i, src_j): distance}
    """
    if not model.source_adaptation_mode or model.delta_alpha is None:
        return {}
    
    with torch.no_grad():
        alpha_global, _ = model.get_alpha()
        betas = []
        for src_id in range(model.num_sources):
            beta_src = model.get_beta_source(src_id, alpha_global)
            betas.append(beta_src.cpu().numpy())  # (K, T, V)
        
        distances = {}
        for i in range(model.num_sources):
            for j in range(i+1, model.num_sources):
                # Flatten to (K*T*V,) for distance computation
                beta_i = betas[i].reshape(-1)
                beta_j = betas[j].reshape(-1)
                
                if metric == 'js':
                    # Jensen-Shannon divergence
                    dist = jensenshannon(beta_i, beta_j)
                elif metric == 'cosine':
                    dist = cosine(beta_i, beta_j)
                else:
                    dist = np.linalg.norm(beta_i - beta_j)
                
                distances[(i, j)] = dist
        
        return distances

# =========== quick exam of topic stats, including diversity and coherence ======= #
def quick_topic_stats(beta_full, vocab, train_tokens, train_times,
                      num_tops_div=25, num_tops_coh=10, sample_docs_per_time=4000):
    """
    Fast training-time stats with the SAME metric definitions as final eval:
    - TD: top-25 diversity averaged over time
    - TC: top-10 PMI coherence averaged over time
    """
    K, T, V = beta_full.shape
    TD_all, TC_all = [], []

    train_times_np = np.asarray(train_times)

    for tt in range(T):
        beta_t = beta_full[:, tt, :]

        # Diversity
        td_t = _diversity_helper(beta_t, num_tops_div)
        TD_all.append(td_t)

        # Coherence docs from this time slice only
        doc_idx_t = np.where(train_times_np == tt)[0]
        if len(doc_idx_t) > sample_docs_per_time:
            doc_idx_t = np.random.choice(doc_idx_t, sample_docs_per_time, replace=False)

        docs_t = [train_tokens[i] for i in doc_idx_t]

        tc_t, _ = get_topic_coherence(
            beta_t.cpu().numpy(),
            docs_t,
            vocab=vocab,
            top_n=num_tops_coh
        )
        TC_all.append(tc_t)

    TD = float(np.mean(TD_all))
    TC = float(np.nanmean(TC_all))
    return TD, TC


def quick_topic_stats_per_source(model, vocab, train_tokens, train_times, train_src_ids, src_names,
                                 num_tops_div=25, num_tops_coh=10, sample_docs_per_time=4000,
                                 use_source_specific_beta=False):
    """
    Compute TD/TC separately for each source corpus.
    
    Args:
        model: DETM model instance
        vocab: vocabulary list
        train_tokens, train_times, train_src_ids: training data
        src_names: list of source names
        use_source_specific_beta: if True, use source-specific beta for each source
    
    Returns dict with per-source metrics and overall metrics.
    """
    train_times_np = np.asarray(train_times) if not isinstance(train_times, torch.Tensor) else train_times.cpu().numpy()
    train_src_np = np.asarray(train_src_ids) if not isinstance(train_src_ids, torch.Tensor) else train_src_ids.cpu().numpy()
    
    results = {}
    
    # Get shared alpha for overall metrics
    with torch.no_grad():
        alpha_global, _ = model.get_alpha()
        beta_shared = model.get_beta(alpha_global)
    
    K, T, V = beta_shared.shape
    
    # Compute overall metrics using shared beta
    overall_td, overall_tc = quick_topic_stats(beta_shared, vocab, train_tokens, train_times, 
                                                num_tops_div, num_tops_coh, sample_docs_per_time)
    results['overall'] = {'TD': overall_td, 'TC': overall_tc}
    
    # Compute per-source metrics
    for src_id, src_name in enumerate(src_names):
        # Get source-specific beta if in adaptation mode
        if use_source_specific_beta and model.source_adaptation_mode:
            with torch.no_grad():
                beta_source = model.get_beta_source(src_id, alpha_global)
        else:
            beta_source = beta_shared
        
        TD_all, TC_all = [], []
        
        for tt in range(T):
            beta_t = beta_source[:, tt, :]
            
            # Diversity
            td_t = _diversity_helper(beta_t, num_tops_div)
            TD_all.append(td_t)
            
            # Coherence: only use docs from this source AND time slice
            doc_idx_t = np.where((train_times_np == tt) & (train_src_np == src_id))[0]
            
            if len(doc_idx_t) == 0:
                TC_all.append(float('nan'))
                continue
                
            if len(doc_idx_t) > sample_docs_per_time:
                doc_idx_t = np.random.choice(doc_idx_t, sample_docs_per_time, replace=False)
            
            docs_t = [train_tokens[i] for i in doc_idx_t]
            
            tc_t, _ = get_topic_coherence(
                beta_t.cpu().numpy(),
                docs_t,
                vocab=vocab,
                top_n=num_tops_coh
            )
            TC_all.append(tc_t)
        
        TD = float(np.mean(TD_all))
        TC = float(np.nanmean(TC_all))
        results[src_name] = {'TD': TD, 'TC': TC}
    
    return results



parser = argparse.ArgumentParser(description='The Embedded Topic Model')

### data and file related arguments
parser.add_argument('--dataset', type=str, default='un', help='name of corpus')
parser.add_argument('--data_path', type=str, default='un/', help='directory containing data')
parser.add_argument('--emb_path', type=str, default='skipgram/embeddings.txt', help='directory containing embeddings')
parser.add_argument('--save_path', type=str, default='./results', help='path to save results')
parser.add_argument('--batch_size', type=int, default=1000, help='number of documents in a batch for training')
parser.add_argument('--min_df', type=int, default=100, help='to get the right data..minimum document frequency')

### model-related arguments
parser.add_argument('--num_topics', type=int, default=50, help='number of topics')
parser.add_argument('--rho_size', type=int, default=300, help='dimension of rho')
parser.add_argument('--emb_size', type=int, default=300, help='dimension of embeddings')
parser.add_argument('--t_hidden_size', type=int, default=800, help='dimension of hidden space of q(theta)')
parser.add_argument('--theta_act', type=str, default='relu', help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
parser.add_argument('--train_embeddings', type=int, default=1, help='whether to fix rho or train it')
parser.add_argument('--eta_nlayers', type=int, default=3, help='number of layers for eta')
parser.add_argument('--eta_hidden_size', type=int, default=200, help='number of hidden units for rnn')
parser.add_argument('--delta', type=float, default=0.005, help='prior variance')

### optimization-related arguments
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--mode', type=str, default='train', help='train or eval model')
parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
parser.add_argument('--seed', type=int, default=2019, help='random seed (default: 2019)')
parser.add_argument('--enc_drop', type=float, default=0.1, help='dropout rate on encoder')
parser.add_argument('--eta_dropout', type=float, default=0.2, help='dropout rate on rnn for eta')
parser.add_argument('--clip', type=float, default=2.0, help='gradient clipping')
parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
parser.add_argument('--wdecay', type=float, default=1e-6, help='some l2 regularization')
parser.add_argument('--anneal_lr', type=int, default=1, help='whether to anneal the learning rate or not')
parser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')

### evaluation, visualization, and logging-related arguments
parser.add_argument('--num_words', type=int, default=20, help='number of words for topic viz')
parser.add_argument('--log_interval', type=int, default=10, help='when to log training')
parser.add_argument('--visualize_every', type=int, default=5, help='when to visualize results')
parser.add_argument('--eval_batch_size', type=int, default=1000, help='input batch size for evaluation')
parser.add_argument('--load_from', type=str, default='', help='the name of the ckpt to eval from')
parser.add_argument('--tc', type=int, default=0, help='whether to compute tc or not')
parser.add_argument('--num_sources', type=int, default=None,help='auto filled after corpus is loaded')
parser.add_argument('--save_checkpoint_every', type=int, default=10, 
                    help='save checkpoint every N epochs (0 = only save best model)')

### LoRa and fine-tune arguments
parser.add_argument('--lora_rank', type=int, default=8)
parser.add_argument('--lora_alpha', type=float, default=1.0)
parser.add_argument('--stage',      type=str,   default='pretrain',
                    choices=['pretrain','lora'])
parser.add_argument('--freeze_rho', action='store_true',
                    help='freeze rho during LoRA stage')
parser.add_argument('--lora_lr',    type=float, default=1e-3)
parser.add_argument('--full_finetune', action='store_true',
                    help='Full fine-tuning mode: unfreeze ALL parameters including rho and alpha')
parser.add_argument('--source_filter', type=str, default=None,
                    help='Filter training to single source (e.g., COHA, HBR, ILR) for full fine-tuning baseline')

### KL divergence weighting arguments
parser.add_argument('--kl_alpha_scale', type=float, default=0.001, 
                    help='scaling factor for KL_alpha term')

### Source-specific topic adaptation arguments
parser.add_argument('--source_adaptation_mode', type=int, default=0,
                    help='Enable source-specific topic adaptation via delta_alpha (1=on, 0=off)')
parser.add_argument('--lambda_anchor', type=float, default=1e-3,
                    help='Anchor regularization: penalize ||delta_alpha||^2')
parser.add_argument('--lambda_smooth', type=float, default=1e-3,
                    help='Temporal smoothness: penalize ||delta_alpha[t] - delta_alpha[t-1]||^2')
parser.add_argument('--freeze_rho_in_adaptation', type=int, default=1,
                    help='Freeze rho (word embeddings) during adaptation (1=freeze, 0=train)')
parser.add_argument('--freeze_alpha_in_adaptation', type=int, default=1,
                    help='Freeze alpha_global during adaptation (1=freeze, 0=train)')
parser.add_argument('--warmup_epochs', type=int, default=30,
                    help='number of epochs for KL warmup in pretraining (0 = no warmup)')
parser.add_argument('--kl_weight_max', type=float, default=1.0,
                    help='maximum KL weight after warmup (default 1.0, try 0.9 to prevent TC plateau)')

# Adaptation-specific KL warmup
parser.add_argument('--adapt_kl_theta_max', type=float, default=0.3,
                    help='Maximum KL_theta weight during adaptation (lower than pretraining)')
parser.add_argument('--adapt_warmup_epochs', type=int, default=5,
                    help='KL warmup epochs during adaptation')

args = parser.parse_args()

RUN_START_TIME = time.perf_counter()
RUN_START_WALL = datetime.now()


def format_elapsed(seconds):
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def log_run_timing():
    elapsed_seconds = time.perf_counter() - RUN_START_TIME
    run_end_wall = datetime.now()
    print('\n' + '=' * 80)
    print(f'[run_timing] start_time: {RUN_START_WALL.isoformat(timespec="seconds")}')
    print(f'[run_timing] end_time:   {run_end_wall.isoformat(timespec="seconds")}')
    print(f'[run_timing] elapsed:    {format_elapsed(elapsed_seconds)} ({elapsed_seconds:.2f} seconds)')
    print(f'[run_timing] mode:       {args.mode}')
    print(f'[run_timing] stage:      {args.stage}')
    print(f'[run_timing] seed:       {args.seed}')
    print(f'[run_timing] torch:      {torch.__version__}')
    print(f'[run_timing] device:     {globals().get("device", "uninitialized")}')
    print(f'[run_timing] cuda_available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f'[run_timing] cuda_device: {torch.cuda.get_device_name(current_device)}')
        print(f'[run_timing] cuda_version: {torch.version.cuda}')
        print(f'[run_timing] max_cuda_memory_allocated_mb: {torch.cuda.max_memory_allocated(current_device) / 1024**2:.2f}')
        print(f'[run_timing] max_cuda_memory_reserved_mb: {torch.cuda.max_memory_reserved(current_device) / 1024**2:.2f}')
    if args.mode == 'train':
        print(f'[run_timing] save_path:  {args.save_path}')
    elif args.load_from:
        print(f'[run_timing] load_from:  {args.load_from}')
    print('=' * 80)


atexit.register(log_run_timing)
print(f'[run_timing] start_time: {RUN_START_WALL.isoformat(timespec="seconds")}')
print(f'[run_timing] seed:       {args.seed}')

pca = PCA(n_components=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['PYTHONHASHSEED'] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

## get data
# 1. vocabulary
print('Getting vocabulary ...')
data_file = os.path.join(args.data_path, 'min_df_{}'.format(args.min_df))
vocab, train, valid, test = data.get_data(data_file, temporal=True)
vocab_size = len(vocab)
args.vocab_size = vocab_size

# 1. training data
print('Getting training data ...')
train_tokens = train['tokens']
train_counts = train['counts']
train_times = train['times']
# Helper function to normalize source names
def normalize_source(x):
    if isinstance(x, np.ndarray):
        x = x.item()
    if isinstance(x, bytes):
        x = x.decode()
    return str(x).strip().upper()

train_sources = [normalize_source(s) for s in train['sources']]
valid_sources_normalized = [normalize_source(s) for s in valid['sources']]
test_sources_normalized  = [normalize_source(s) for s in test['sources']]

# build the source and id table
src_names = sorted(set(train_sources))        # ['COHA','HBR','ILR']
args.num_sources = len(src_names)
name2id = {s:i for i,s in enumerate(src_names)}

train_src_ids = torch.tensor([name2id[s] for s in train_sources],
                             dtype=torch.long, device=device)
valid_src_ids = torch.tensor([name2id[s] for s in valid_sources_normalized],
                             dtype=torch.long, device=device)
test_src_ids = torch.tensor([name2id[s] for s in test_sources_normalized],
                             dtype=torch.long, device=device)

args.num_times = len(np.unique(train_times))
print("train_times shape:", train_times.shape)
counts_per_time = np.bincount(train_times)
for t_idx, c in enumerate(counts_per_time):
    print(f"Time {t_idx} has {c} docs in train")
args.num_docs_train = len(train_tokens)
train_rnn_inp = data.get_rnn_input(
    train_tokens, train_counts, train_times, args.num_times, args.vocab_size, args.num_docs_train)


time_path = os.path.join(data_file, 'timestamps.pkl')
with open(time_path, 'rb') as f:
    timelist = pickle.load(f)
# print('timelist: ', timelist)
T = len(timelist)
ticks = [str(x) for x in timelist]

train_year = [ticks[t] for t in train_times]  
meta = pd.DataFrame({'year': train_year, 'src': train_sources})

# 2. dev set
print('Getting validation data ...')
valid_tokens = valid['tokens']
valid_counts = valid['counts']
valid_times = valid['times']
args.num_docs_valid = len(valid_tokens)
valid_rnn_inp = data.get_rnn_input(
    valid_tokens, valid_counts, valid_times, args.num_times, args.vocab_size, args.num_docs_valid)
counts_per_time_val = np.bincount(valid_times)
for t_idx, c in enumerate(counts_per_time_val):
    print(f"Time {t_idx} has {c} docs in valid")
# 3. test data
print('Getting testing data ...')
test_tokens = test['tokens']
test_counts = test['counts']
test_times = test['times']
args.num_docs_test = len(test_tokens)
test_rnn_inp = data.get_rnn_input(
    test_tokens, test_counts, test_times, args.num_times, args.vocab_size, args.num_docs_test)

test_1_tokens = test['tokens_1']
test_1_counts = test['counts_1']
test_1_times = test_times
args.num_docs_test_1 = len(test_1_tokens)
test_1_rnn_inp = data.get_rnn_input(
    test_1_tokens, test_1_counts, test_1_times, args.num_times, args.vocab_size, args.num_docs_test_1)

test_2_tokens = test['tokens_2']
test_2_counts = test['counts_2']
test_2_times = test_times
args.num_docs_test_2 = len(test_2_tokens)
test_2_rnn_inp = data.get_rnn_input(
    test_2_tokens, test_2_counts, test_2_times, args.num_times, args.vocab_size, args.num_docs_test_2)

# Source filtering for full fine-tuning baseline (AFTER all data is loaded)
if args.source_filter is not None:
    source_filter_normalized = args.source_filter.strip().upper()
    print(f"\n{'='*80}")
    print(f"SOURCE FILTERING: Only training on {source_filter_normalized} documents")
    print(f"{'='*80}\n")
    
    # Filter training data
    train_mask = np.array([s == source_filter_normalized for s in train_sources])
    train_indices = np.where(train_mask)[0]
    
    print(f"Original training set: {len(train_tokens)} docs")
    print(f"Filtered to {source_filter_normalized}: {len(train_indices)} docs")
    
    # Apply filter
    train_tokens = [train_tokens[i] for i in train_indices]
    train_counts = [train_counts[i] for i in train_indices]
    train_times = train_times[train_indices]
    train_sources = [train_sources[i] for i in train_indices]
    train_src_ids = train_src_ids[train_indices]
    
    # Rebuild RNN input with filtered data
    args.num_docs_train = len(train_tokens)
    train_rnn_inp = data.get_rnn_input(
        train_tokens, train_counts, train_times, args.num_times, args.vocab_size, args.num_docs_train)
    
    # Filter validation data
    valid_mask = np.array([s == source_filter_normalized for s in valid_sources_normalized])
    valid_indices = np.where(valid_mask)[0]
    valid_tokens = [valid_tokens[i] for i in valid_indices]
    valid_counts = [valid_counts[i] for i in valid_indices]
    valid_times = valid_times[valid_indices]
    valid_src_ids = valid_src_ids[valid_indices]
    args.num_docs_valid = len(valid_tokens)
    valid_rnn_inp = data.get_rnn_input(
        valid_tokens, valid_counts, valid_times, args.num_times, args.vocab_size, args.num_docs_valid)
    
    # Filter test data
    test_mask = np.array([s == source_filter_normalized for s in test_sources_normalized])
    test_indices = np.where(test_mask)[0]
    test_tokens = [test_tokens[i] for i in test_indices]
    test_counts = [test_counts[i] for i in test_indices]
    test_times = test_times[test_indices]
    test_src_ids = test_src_ids[test_indices]
    args.num_docs_test = len(test_tokens)
    test_rnn_inp = data.get_rnn_input(
        test_tokens, test_counts, test_times, args.num_times, args.vocab_size, args.num_docs_test)
    
    # Filter test_1 and test_2
    test_1_tokens = [test_1_tokens[i] for i in test_indices]
    test_1_counts = [test_1_counts[i] for i in test_indices]
    test_1_times = test_times
    args.num_docs_test_1 = len(test_1_tokens)
    test_1_rnn_inp = data.get_rnn_input(
        test_1_tokens, test_1_counts, test_1_times, args.num_times, args.vocab_size, args.num_docs_test_1)
    
    test_2_tokens = [test_2_tokens[i] for i in test_indices]
    test_2_counts = [test_2_counts[i] for i in test_indices]
    test_2_times = test_times
    args.num_docs_test_2 = len(test_2_tokens)
    test_2_rnn_inp = data.get_rnn_input(
        test_2_tokens, test_2_counts, test_2_times, args.num_times, args.vocab_size, args.num_docs_test_2)
    
    print(f"Filtered validation set: {args.num_docs_valid} docs")
    print(f"Filtered test set: {args.num_docs_test} docs\n")

# make the contribution for each individual document balanced,
# even with unbalanced training data
target_share = {'COHA': 1/3, 'HBR': 1/3, 'ILR': 1/3}
from collections import Counter
src_counts = Counter(train_sources)          # {'COHA':195367, 'HBR':22984, ...}
w_list = [target_share[s] / src_counts[s]  for s in train_sources ]
w_table = torch.tensor(w_list, dtype=torch.float32, device=device)

## get embeddings 
print('Getting embeddings ...')
emb_path = args.emb_path
vect_path = os.path.join(args.data_path.split('/')[0], 'embeddings.pkl')   
vectors = {}
file_ext = os.path.splitext(emb_path)[-1]
print(f"File ending in {file_ext}")
if file_ext == ".npy":
    embedding_matrix = np.load(emb_path)
    assert embedding_matrix.shape[0] == len(vocab), \
        f"Embedding vocab size ({embedding_matrix.shape[0]}) and current vocab size ({len(vocab)}) mismatch!"
    embedding_matrix = np.ascontiguousarray(embedding_matrix, dtype=np.float32)
    embeddings = torch.tensor(embedding_matrix, device=device)
else:
    with open(emb_path, "rb") as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            if word in vocab:
                vect = np.array(line[1:]).astype(np.float32)
                vectors[word] = vect
    embeddings = np.zeros((vocab_size, args.emb_size))
    words_found = 0
    for i, word in enumerate(vocab):
        try: 
            embeddings[i] = vectors[word]
            words_found += 1
        except KeyError:
            embeddings[i] = np.random.normal(scale=0.6, size=(args.emb_size, ))
    embeddings = torch.from_numpy(embeddings).to(device)

args.embeddings_dim = embeddings.size()

print('\n')
print('=*'*100)
print('Training a Dynamic Embedded Topic Model on {} with the following settings: {}'.format(args.dataset.upper(), args))
print('=*'*100)

## define checkpoint
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if args.mode == 'eval':
    ckpt = args.load_from
else:
    base_ckpt = os.path.join(args.save_path, 
        'detm_{}_K_{}_Htheta_{}_Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_L_{}_minDF_{}_trainEmbeddings_{}'.format(
        args.dataset, args.num_topics, args.t_hidden_size, args.optimizer, args.clip, args.theta_act, 
            args.lr, args.batch_size, args.rho_size, args.eta_nlayers, args.min_df, args.train_embeddings))

    # adding post-fix to avoid overlapping of the model checkpoint
    if args.stage == 'pretrain':
        ckpt = base_ckpt + '_pretrain.pt'
    else:          # stage == 'lora'
        ckpt = base_ckpt + f'_lora_r{args.lora_rank}.pt'
## define model and optimizer
# if args.load_from != '':
#     print('Loading checkpoint from {}'.format(args.load_from))
#     with open(args.load_from, 'rb') as f:
#         model = torch.load(f)
# else:
#     model = DETM(args, embeddings)
# print('\nDETM architecture: {}'.format(model))
model = DETM(args, embeddings).to(device)
print('\nDETM architecture: {}'.format(model))
# adding lora 
if args.stage == 'pretrain':
    if args.load_from:
        print(f"Loading checkpoint {args.load_from}")
        state = torch.load(args.load_from, map_location=device)
        if isinstance(state, torch.nn.Module):  # Full model object
            model = state
        else:  # State dict
            model.load_state_dict(state, strict=False)
elif args.stage == 'lora':
    assert args.load_from, "LoRA/adaptation stage requires --load_from ckpt"
    print(f"Loading backbone from {args.load_from}")
    base_state = torch.load(args.load_from, map_location=device)
    if isinstance(base_state, torch.nn.Module):          # old ckpt is saving the complete model
        base_state = base_state.state_dict()             # get state_dict
    model.load_state_dict(base_state, strict=False) # old checkpoint doesn't have delta_alpha or lora_*

    # Freeze/unfreeze based on adaptation mode
    if args.full_finetune:
        # Full fine-tuning baseline: unfreeze ALL parameters including rho and alpha
        print("FULL FINE-TUNING MODE: unfreezing ALL model parameters (including rho and alpha)")
        for n, p in model.named_parameters():
            p.requires_grad = True
            if 'rho' in n or 'mu_q_alpha' in n or 'logsigma_q_alpha' in n:
                print(f"  Unfreezing (backbone): {n}")
    elif args.source_adaptation_mode:
        # Source-specific topic adaptation mode
        print(f"Source adaptation mode: learning delta_alpha residuals")
        print(f"  lambda_anchor={args.lambda_anchor}, lambda_smooth={args.lambda_smooth}")
        for n, p in model.named_parameters():
            if 'delta_alpha' in n:
                p.requires_grad = True
                print(f"  Unfreezing: {n} (shape={list(p.shape)})")
            elif 'mu_q_alpha' in n or 'logsigma_q_alpha' in n:
                if args.freeze_alpha_in_adaptation:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
                    print(f"  Unfreezing: {n}")
            elif 'rho' in n:
                if args.freeze_rho_in_adaptation:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
                    print(f"  Unfreezing: {n}")
            else:
                # Unfreeze inference networks (q_theta, q_eta)
                p.requires_grad = True
    elif args.lora_rank == 0:
        # Full fine-tuning: unfreeze all parameters except embeddings
        print("Full fine-tuning mode (LORA_RANK=0): unfreezing all model parameters except rho/alpha")
        for n, p in model.named_parameters():
            if ('rho' in n) or ('mu_q_alpha' in n) or ('logsigma_q_alpha' in n):
                p.requires_grad = False
            else:
                p.requires_grad = True
    else:
        # LoRA fine-tuning: only unfreeze LoRA parameters
        print(f"LoRA fine-tuning mode (LORA_RANK={args.lora_rank}): only unfreezing lora_* parameters")
        for n, p in model.named_parameters():
            if 'lora_' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
print('\nDETM architecture: {}'.format(model))
# model.to(device)

# build optimizer
if args.stage =='pretrain':
    train_params = model.parameters()
    lr_main = args.lr
else:
    train_params = [p for p in model.parameters() if p.requires_grad]
    lr_main = args.lora_lr


if args.optimizer == 'adam':
    optimizer = optim.Adam(train_params, lr=lr_main, weight_decay=args.wdecay)
elif args.optimizer == 'adagrad':
    optimizer = optim.Adagrad(train_params, lr=lr_main, weight_decay=args.wdecay)
elif args.optimizer == 'adadelta':
    optimizer = optim.Adadelta(train_params, lr=lr_main,weight_decay=args.wdecay)
elif args.optimizer == 'rmsprop':
    optimizer = optim.RMSprop(train_params, lr=lr_main, weight_decay=args.wdecay)
elif args.optimizer == 'asgd':
    optimizer = optim.ASGD(train_params, lr=lr_main, t0=0, lambd=0., weight_decay=args.wdecay)
else:
    print('Defaulting to vanilla SGD')
    optimizer = optim.SGD(train_params, lr=lr_main)

def train(epoch):
    """Train DETM on data for one epoch.
    """
    model.train()
    acc_loss = 0
    acc_nll = 0
    acc_kl_theta_loss = 0
    acc_kl_eta_loss = 0
    acc_kl_alpha_loss = 0
    cnt = 0

    src_stats = {s: dict(nll=0., kl_theta=0., kl_alpha=0., kl_eta=0., wsum=0.) for s in src_names}

    indices = torch.randperm(args.num_docs_train)
    indices = torch.split(indices, args.batch_size) 

    for idx, ind in enumerate(indices):
        optimizer.zero_grad()
        model.zero_grad()
        data_batch, times_batch = data.get_batch(
            train_tokens, train_counts, ind, args.vocab_size, args.emb_size, temporal=True, times=train_times)
        sums = data_batch.sum(1).unsqueeze(1)

        if (sums == 0).any():
            bad_idx = torch.nonzero(sums.squeeze() == 0).squeeze()
            print(f"[WARN] zero-length docs in batch {idx}: indices {bad_idx}")
            data_batch = data_batch[sums.squeeze() != 0]
            times_batch = times_batch[sums.squeeze() != 0]
            sums = sums[sums != 0]
        if args.bow_norm:
            normalized_data_batch = data_batch / sums
        else:
            normalized_data_batch = data_batch

         
        # loss, nll_vec, kl_alpha, kl_eta, kl_theta = model(data_batch, normalized_data_batch, times_batch, train_rnn_inp, args.num_docs_train)
        # ------------ sample-weight ------------
        # origianl coeff
        # re-distribute the weight for each individual document
        bsz   = len(ind)                         
        coeff = args.num_docs_train / bsz 
        batch_w = w_table[ind]  
        src_batch = train_src_ids[ind] 
        nll_vec, kl_alpha, kl_eta, kl_theta_vec = model(
            data_batch, normalized_data_batch, times_batch, train_rnn_inp, args.num_docs_train, src_batch)
        
        # Compute weighted NLL and KL_theta
        nll = (nll_vec * batch_w).sum() * coeff
        kl_theta = (kl_theta_vec * batch_w).sum() * coeff
        
        # KL warmup: different strategies for pretraining vs adaptation
        if args.source_adaptation_mode:
            # Source adaptation: use KL_theta only, ignore KL_alpha and KL_eta
            # Warmup KL_theta to adapt_kl_theta_max
            if args.adapt_warmup_epochs > 0 and epoch <= args.adapt_warmup_epochs:
                kl_theta_w = (epoch / args.adapt_warmup_epochs) * args.adapt_kl_theta_max
            else:
                kl_theta_w = args.adapt_kl_theta_max
            
            # KL_alpha and KL_eta set to 0 (frozen components)
            kl_alpha_w = 0.0
            kl_eta_w = 0.0
            kl_w = kl_theta_w  # For per-source stats compatibility
            
            loss = nll + kl_theta_w * kl_theta
            disp_kl_theta = (kl_theta_w * kl_theta).item()
            disp_kl_alpha = 0.0
            disp_kl_eta = 0.0
            
        elif args.stage == 'lora':
            # Legacy LoRA mode: no KL regularization
            kl_w = 0.0
            loss = nll + kl_w * (kl_theta + kl_alpha + kl_eta)
            disp_kl_theta = 0.0
            disp_kl_alpha = 0.0
            disp_kl_eta = 0.0
            
        else:
            # Pretraining mode: standard KL warmup
            kl_alpha = args.kl_alpha_scale * kl_alpha
            if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
                kl_w = (epoch / args.warmup_epochs) * args.kl_weight_max
            else:
                kl_w = args.kl_weight_max
            
            loss = nll + kl_w * (kl_theta + kl_alpha + kl_eta)
            disp_kl_theta = (kl_w * kl_theta).item()
            disp_kl_alpha = (kl_w * kl_alpha).item()
            disp_kl_eta = (kl_w * kl_eta).item()

        # Add regularization based on adaptation mode
        if args.source_adaptation_mode:
            # Source-specific topic adaptation regularization
            loss_anchor, loss_smooth = model.get_delta_alpha_regularization(
                lambda_anchor=args.lambda_anchor,
                lambda_smooth=args.lambda_smooth
            )
            loss = loss + loss_anchor + loss_smooth
        elif args.stage == 'lora' and args.lora_rank > 0:
            # LoRA regularization
            reg = 1e-3 * (model.lora_A.pow(2).sum() + model.lora_B.pow(2).sum())
            loss = loss + reg
    
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        acc_loss += torch.sum(loss).item()
        acc_nll += torch.sum(nll).item()
        acc_kl_theta_loss += disp_kl_theta
        acc_kl_eta_loss += disp_kl_eta
        acc_kl_alpha_loss += disp_kl_alpha
        cnt += 1
        


        for s_id, s_name in enumerate(src_names):
            m = (src_batch == s_id)
            if m.any():
                w_s = batch_w[m]
                share = w_s.sum().item() * coeff          
                src_stats[s_name]['nll']       += (nll_vec[m]      * w_s).sum().item() * coeff
                src_stats[s_name]['kl_theta']  += (kl_theta_vec[m] * w_s).sum().item() * coeff
                src_stats[s_name]['kl_alpha']  += kl_alpha.item() * share / args.num_docs_train
                src_stats[s_name]['kl_eta']    += kl_eta.item()   * share / args.num_docs_train
                src_stats[s_name]['wsum']      += share


        
        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = round(acc_loss / cnt, 2) 
            cur_nll = round(acc_nll / cnt, 2) 
            cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
            cur_kl_eta = round(acc_kl_eta_loss / cnt, 2) 
            cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2) 
            lr = optimizer.param_groups[0]['lr']
            print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. KL_eta: {} .. KL_alpha: {} .. Rec_loss: {} .. NELBO: {}'.format(
                epoch, idx, len(indices), lr, cur_kl_theta, cur_kl_eta, cur_kl_alpha, cur_nll, cur_loss))
        
            check = 0.0
            total_w = sum(d['wsum'] for d in src_stats.values()) or 1.0
            for s in src_names:
                w_share = src_stats[s]['wsum'] / total_w 
                nll_s   = src_stats[s]['nll']      / cnt
                kl_th_s = src_stats[s]['kl_theta'] / cnt
                kl_al_s = cur_kl_alpha * w_share     #  curr_kl_alpha / curr_kl_eta
                kl_et_s = cur_kl_eta   * w_share
                nelbo_s = nll_s + kl_w * kl_th_s + kl_al_s + kl_et_s
                check  += nelbo_s
                print(f"    ↳ {s:4s} NLL {nll_s:.2f} | KL_theta {kl_th_s:.2f} | "
                        f"KL_eta {kl_et_s:.2f} | KL_alpha {kl_al_s:.2f} | NELBO {nelbo_s:.2f}")
            print(f"    Σ per-source NELBO = {check:.2f}  (global {cur_loss:.2f})\n")
                
        
    cur_loss = round(acc_loss / cnt, 2) 
    cur_nll = round(acc_nll / cnt, 2) 
    cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
    cur_kl_eta = round(acc_kl_eta_loss / cnt, 2) 
    cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2) 
    lr = optimizer.param_groups[0]['lr']
    
    # Calculate current KL weight for display
    if args.source_adaptation_mode:
        if args.adapt_warmup_epochs > 0 and epoch <= args.adapt_warmup_epochs:
            display_kl_w = (epoch / args.adapt_warmup_epochs) * args.adapt_kl_theta_max
        else:
            display_kl_w = args.adapt_kl_theta_max
    elif args.stage == 'lora':
        display_kl_w = 0.0
    else:
        if args.warmup_epochs > 0:
            display_kl_w = min((epoch / args.warmup_epochs) * args.kl_weight_max, args.kl_weight_max)
        else:
            display_kl_w = args.kl_weight_max
    
    print('*'*100)
    print('Epoch----->{} .. LR: {} .. KL_weight: {:.3f} .. KL_theta: {} .. KL_eta: {} .. KL_alpha: {} .. Rec_loss: {} .. NELBO: {}'.format(
            epoch, lr, display_kl_w, cur_kl_theta, cur_kl_eta, cur_kl_alpha, cur_nll, cur_loss))
    print('*'*100)

def visualize():
    """Visualizes topics and embeddings and word usage evolution.
    """
    model.eval()
    with torch.no_grad():
        alpha = model.mu_q_alpha
        beta = model.get_beta(alpha) 
        print('beta: ', beta.size())
        print('\n')
        print('#'*100)
        print('Visualize topics...')
        # Dynamically select time indices based on actual num_times
        # Select 2 representative times: early (~20%) and mid (~40% of timeline)
        max_time = args.num_times - 1
        if args.num_times >= 40:
            # For yearly data (98 times): use original [20, 40]
            times = [20, 40]
        else:
            # For coarser bins: select proportionally
            times = [max(0, int(0.2 * max_time)), max(0, int(0.4 * max_time))]
        print(f'Showing topics at time indices: {times} (out of {args.num_times} total)')
        
        topics_words = []
        for k in range(args.num_topics):
            for t in times:
                gamma = beta[k, t, :]
                top_words = list(gamma.cpu().numpy().argsort()[-args.num_words+1:][::-1])
                topic_words = [vocab[a] for a in top_words]
                topics_words.append(' '.join(topic_words))
                print('Topic {} .. Time: {} ===> {}'.format(k, t, topic_words)) 
            print("="*50)
        # fine tuned topics
        if args.source_adaptation_mode:
            print('\nPer-source β after adaptation (using delta_alpha)')
        else:
            print('\nPer-source β after LoRA')
        
        for s_name in src_names:                         # e.g. 'COHA', 'HBR', 'ILR'
            s_id   = name2id[s_name]
            
            # Use source-specific beta if in adaptation mode
            if args.source_adaptation_mode and hasattr(model, 'get_beta_source'):
                beta_s = model.get_beta_source(s_id, alpha)  # (K,T,V) - source-specific!
            else:
                beta_s = model.get_beta(alpha, src_id=s_id)  # (K,T,V) - legacy LoRA

            print('\n' + '#' * 120)
            print(f'[{s_name}] beta size: {beta_s.size()}')
            for k in range(args.num_topics):
                for t in times:
                    top_ids = beta_s[k, t].topk(args.num_words).indices
                    words   = [vocab[i] for i in top_ids]
                    print(f'{s_name[:3]} | topic {k:02d} @ t={t:02d} : {words}')
                print('-'*120)
        print('='*120 + '\n')
        print('\n')
        print('Visualize word embeddings ...')
        queries = ['economic', 'assembly', 'security', 'management', 'debt', 'rights',  'africa',
                   'strike', 'factory', 'china',
                   'union']

        try:
            embeddings = model.rho.weight  # Vocab_size x E
        except:
            embeddings = model.rho         # Vocab_size x E
        neighbors = []
        for word in queries:
            print('word: {} .. neighbors: {}'.format(
                word, nearest_neighbors(word, embeddings, vocab, args.num_words)))
        print('#'*100)
        
        # ---------- 3. source-specific neighbors ----------
        # Skip if source filtering is active (only one source present)
        if args.source_filter is None:
            print('\nSource-specific neighbors (topic space)\n')
            
            theta_mat, _ = model.infer_all_theta(
                train_tokens, train_counts, train_times, train_rnn_inp, train_src_ids)
            theta_bar = get_theta_bar_by_source(
                theta_mat, train_counts, np.array(train_sources),
                length_weight=False)
            theta_bar_time = get_theta_bar_by_source_time(
                        theta_mat, train_counts, train_times,
                        np.array(train_sources), T=args.num_times,
                        length_weight=False)
            
            g_mats = build_g_mats(theta_bar, beta) 
        
            print("\n[DEBUG] theta_mat sample rows")
            print(theta_mat[:5, :8])            
            print("row-means :", theta_mat[:5].mean(1))
            print("row-vars  :", theta_mat[:5].var(1))
            for s, th in theta_bar.items():
                print(f"{s:4s}  mean={th.mean():.4f}  var={th.var():.4f}  "
                    f"top3_idx={th.topk(3).indices.tolist()}")

            to_np_prob = lambda t: (t.detach().cpu().float().numpy() + 1e-12) / t.sum()

            # Only compute cross-corpus comparisons if multiple sources exist
            if 'HBR' in theta_bar and 'COHA' in theta_bar:
                print("JS(HBR‖COHA):", jensenshannon(
                        to_np_prob(theta_bar['HBR']),
                        to_np_prob(theta_bar['COHA'])))
            if 'ILR' in theta_bar and 'COHA' in theta_bar:
                print("JS(ILR‖COHA):", jensenshannon(
                        to_np_prob(theta_bar['ILR']),
                        to_np_prob(theta_bar['COHA'])))
            if 'HBR' in theta_bar and 'ILR' in theta_bar:
                print("cos(HBR,ILR):", cosine(
                        to_np_prob(theta_bar['HBR']),
                        to_np_prob(theta_bar['ILR'])))

            g_mats = build_g_mats(theta_bar, beta, l2_norm=False)
            g_mats_time = build_g_mats_times(theta_bar_time, beta, l2_norm=False)

            # for w in queries:
            #     for src in g_mats.keys():
            #         nbs = src_neighbors(w, src, g_mats, vocab, topk=10)
            #         print(f"{src:4s} | {w:10s} → {nbs}")
            #     print('-' * 60)
            
            # Dynamically select time indices for neighbor visualization
            if args.num_times >= 80:
                # For yearly data (98 times): use original [0, 20, 40, 60, 80]
                time_indices = [0, 20, 40, 60, 80]
            else:
                # For coarser bins: select evenly spaced indices
                max_time = args.num_times - 1
                time_indices = [0, 
                               max(1, int(0.2 * max_time)), 
                               max(2, int(0.4 * max_time)), 
                               max(3, int(0.6 * max_time)), 
                               max_time]
            print(f'Neighbor visualization at time indices: {time_indices}')
            
            for w in queries:
                for src in g_mats.keys():
                    for time in time_indices:
                        nbs = src_neighbors_time(w, src, 
                                            t=time,
                                            g_mats_time=g_mats_time, 
                                            vocab=vocab, 
                                            topk=10)
                        print(f"{src:4s} at time_{time} | {w:10s} → {nbs}")
                print('-' * 60)
    print("-"*80)



        # print('\n')
        # print('Visualize word evolution ...')
        # topic_0 = None ### k 
        # queries_0 = ['woman', 'gender', 'man', 'mankind', 'humankind'] ### v 

        # topic_1 = None
        # queries_1 = ['africa', 'colonial', 'racist', 'democratic']

        # topic_2 = None
        # queries_2 = ['poverty', 'sustainable', 'trade']

        # topic_3 = None
        # queries_3 = ['soviet', 'convention', 'iran']

        # topic_4 = None # climate
        # queries_4 = ['environment', 'impact', 'threats', 'small', 'global', 'climate']

def _eta_helper(rnn_inp):
    inp = model.q_eta_map(rnn_inp).unsqueeze(1)
    if torch.isnan(inp).any():
        print("[DEBUG] q_eta_map(rnn_inp) contains NaN!")
    hidden = model.init_hidden()
    output, _ = model.q_eta(inp, hidden)
    if torch.isnan(output).any():
        print("[DEBUG] RNN output has NaN before squeeze!")
    output = output.squeeze()
    if torch.isnan(output).any():
        print("[DEBUG] RNN output has NaN after squeeze!")
    if len(output.shape) != 2:
        print("[DEBUG] Unexpected output shape:", output.shape,
              " (Should be [num_times, hidden_dim])")
    etas = torch.zeros(model.num_times, model.num_topics).to(device)
    inp_0 = torch.cat([output[0], torch.zeros(model.num_topics,).to(device)], dim=0)
    if torch.isnan(inp_0).any():
        print("[DEBUG] inp_0 has NaN before mu_q_eta at time 0!")

    etas[0] = model.mu_q_eta(inp_0)
    if torch.isnan(etas[0]).any():
        print("[DEBUG] etas[0] is NaN after mu_q_eta at time 0!")
    for t in range(1, model.num_times):
        if torch.isnan(output[t]).any():
            print(f"[DEBUG] RNN output is NaN at time step {t}!")
        if torch.isnan(etas[t-1]).any():
            print(f"[DEBUG] etas[{t-1}] already NaN before forming inp_t at time {t}!")
        inp_t = torch.cat([output[t], etas[t-1]], dim=0)
        if torch.isnan(inp_t).any():
            print(f"[DEBUG] inp_t has NaN before mu_q_eta at time {t}!")
        etas[t] = model.mu_q_eta(inp_t)
        if torch.isnan(etas[t]).any():
            print(f"[DEBUG] etas[{t}] is NaN after mu_q_eta at time {t}!")
    return etas

def get_eta(source):
    model.eval()
    with torch.no_grad():
        if source == 'val':
            rnn_inp = valid_rnn_inp
            return _eta_helper(rnn_inp)
        else:
            rnn_1_inp = test_1_rnn_inp
            return _eta_helper(rnn_1_inp)

def get_theta(eta, bows):
    model.eval()
    with torch.no_grad():
        inp = torch.cat([bows, eta], dim=1)
        q_theta = model.q_theta(inp)
        mu_theta = model.mu_q_theta(q_theta)
        theta = F.softmax(mu_theta, dim=-1)
        return theta    

def get_theta_bar_by_source(theta_mat, counts, src_arr, length_weight=True):
    """
    Return the theta (document proportion average for
    each sources)
    """
    out = {}
    counts = np.array(counts, dtype=object)   
    for s in np.unique(src_arr):
        mask = (src_arr == s)
        print(f"[DEBUG] {s}  docs = {mask.sum()}")
        lengths = torch.tensor(
            [row.sum() if hasattr(row, "sum") else sum(row)
             for row in counts[mask]],
            dtype=torch.float32
        ).to(theta_mat.device)
        print(f"[DEBUG] {s}  len min/max :", lengths.min().item(), lengths.max().item())
    
        theta_src = theta_mat[mask] 
        if length_weight:
            theta_bar = (theta_src.T * lengths).sum(1) / lengths.sum()
        else:                                       
            theta_bar = theta_src.mean(dim=0)
        out[s] = theta_bar 

    return out


def get_theta_bar_by_source_time(theta_mat: torch.Tensor,
                                 counts:   list,
                                 times:    np.ndarray,
                                 src_arr:  np.ndarray,
                                 T:        int,
                                 length_weight: bool = True):
    """
    Returns:
        theta_bar_time : dict[src] → torch.Tensor (T, K)
    """
    counts = np.array(counts, dtype=object)
    out = {}
    for s in np.unique(src_arr):
        mask_src = (src_arr == s)
        theta_s  = theta_mat[mask_src]        # (D_s, K)
        times_s  = times[mask_src]            # (D_s,)
        K        = theta_mat.size(1)
        theta_avg_t = torch.zeros(T, K, device=device)
        for t in range(T):
            mask_t = (times_s == t)
            if mask_t.sum() == 0:
                continue                      # leave zeros if no docs at t
            theta_t = theta_s[mask_t]         # (d_t, K)
            if length_weight:
                lens = torch.tensor(
                    [row.sum() if hasattr(row, "sum") else sum(row)
                     for row in counts[mask_src][mask_t]],
                    dtype=torch.float32, device=device)
                theta_avg_t[t] = (theta_t.T * lens).sum(1) / lens.sum()
            else:
                theta_avg_t[t] = theta_t.mean(0)
        out[s] = theta_avg_t                  # (T, K)
    return out


def build_g_mats(theta_bar, beta, l2_norm=False):
    """
    theta_bar : dict{src: torch.Tensor (K,)}
    beta      : torch.Tensor (K, T, V), the topic-word distribution across time
    l2_norm   : l2_norm for calculating cosine similarity
    ------
    return g_mats : dict{src: torch.Tensor (K, V)},
    each column corresponds to g_{v,src}
    """
    g_mats = {}
    beta_avg = beta.mean(dim=1).to(next(iter(theta_bar.values())).device)
    for src, th in theta_bar.items():
        g = th.unsqueeze(1) * beta_avg # average beta 
        if l2_norm:
            g = g / g.norm(dim=0, keepdim=True).clamp_min(1e-12)
        g_mats[src] = g
    return g_mats

def build_g_mats_times(theta_bar_time, beta, l2_norm=False):
    beta_t = beta.permute(1, 0, 2).contiguous()  
    g_mats_time = {}
    for src, theta_t in theta_bar_time.items():    # theta_t : T × K
        if theta_t.shape[0] != beta_t.shape[0]:
            raise ValueError(f"theta[{src}].shape={theta_t.shape} "
                             f" vs beta.shape={beta_t.shape}")

        # broadcast: (T, K, 1)  ×  (T, K, V)  →  (T, K, V)
        g = theta_t.unsqueeze(2) * beta_t

        if l2_norm:
            # for each time, for each v do L2：||g_{:,v}||₂=1
            g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)

        g_mats_time[src] = g                     #  T × K × V

    return g_mats_time



def src_neighbors(word, src, g_mats, vocab, topk=12):
    """
    with a specific source, return top-k most similar 
    words with the query word.

    word   : str
    src    : "HBR" | "ILR" | "COHA"
    g_mats : dict{src: (K,V)} —— get_g_mat 
    vocab  : list[str]        —— idx to token
    """
    g = g_mats[src]                          # (K,V)
    wid = vocab.index(word)                  # keyword index
    vec = g[:, wid]                          # (K,)
    # due to l2 normalization, dot = cosine
    sims = torch.matmul(vec, g)
    idx = sims.argsort(descending=True)[1:topk+1] 
    return [vocab[i] for i in idx]


def src_neighbors_time(word: str,
                       src: str,
                       t: int,
                       g_mats_time: dict,
                       vocab: list,
                       topk: int = 12):
    """
    Same idea as `src_neighbors`, but restricted to a single time‑slice.

    Parameters
    ----------
    word, src, vocab, topk
        (See `src_neighbors`.)
    t     : int
        Zero_based time index. 0 == first timestamp used by DETM.

    g_mats_time : dict[str, torch.Tensor]
        Maps each corpus to a tensor of shape (T, K, V) created by
        `build_g_mats_times`.

    Returns
    -------
    list[str]
        Nearest neighbours of *word* inside the (src, t) slice.
    """
    g_t = g_mats_time[src][t]       # (K, V)
    wid = vocab.index(word)
    vec = g_t[:, wid]               # (K,)
    sims = torch.matmul(vec, g_t)   # (V,)
    idx  = sims.argsort(descending=True)[1: topk + 1]
    return [vocab[i] for i in idx]


def get_completion_ppl(source):
    """Returns document completion perplexity.
    """
    model.eval()
    with torch.no_grad():
        alpha = model.mu_q_alpha
        if source == 'val':
            indices = torch.split(torch.tensor(range(args.num_docs_valid)), args.eval_batch_size)
            tokens = valid_tokens
            counts = valid_counts
            times = valid_times
            eta = get_eta('val')

            acc_loss = 0
            cnt = 0
            for idx, ind in enumerate(indices):
                data_batch, times_batch = data.get_batch(
                    tokens, counts, ind, args.vocab_size, args.emb_size, temporal=True, times=times)
                print("[DEBUG] times_batch range:", times_batch.min().item(), times_batch.max().item())
                assert times_batch.max().item() < args.num_times, f"times_batch out of range!"
                sums = data_batch.sum(1).unsqueeze(1)
                if torch.isnan(data_batch).any():
                    print(f"[DEBUG] data_batch has NaN in batch {idx}")
                if (sums == 0).any():
                    print(f"[DEBUG] sums==0 in batch {idx}, times_batch: {times_batch}")

                if args.bow_norm:
                    normalized_data_batch = data_batch / sums
                else:
                    normalized_data_batch = data_batch

                times_batch_long = times_batch.long()
                eta_td = eta[times_batch_long]
                if torch.isnan(eta_td).any():
                    print(f"[DEBUG] eta_td has NaN in batch {idx}")
                theta = get_theta(eta_td, normalized_data_batch)
                if torch.isnan(theta).any():
                    print(f"[DEBUG] theta has NaN in batch {idx}")
                alpha_td = alpha[:, times_batch_long, :]
                beta = model.get_beta(alpha_td).permute(1, 0, 2)
                if torch.isnan(beta).any():
                    print(f"[DEBUG] beta has NaN in batch {idx}")
                loglik = theta.unsqueeze(2) * beta
                loglik = loglik.sum(1)

                if torch.isnan(loglik).any():
                    print(f"[DEBUG] loglik (before log) has NaN in batch {idx}")
                if (loglik < 0).any():
                    print(f"[DEBUG] loglik < 0 in batch {idx} (should be >= 0)")
                loglik = torch.log(loglik + 1e-10)
                if torch.isnan(loglik).any():
                    print(f"[DEBUG] loglik (after log) has NaN in batch {idx}")
                nll = -loglik * data_batch
                if torch.isnan(nll).any():
                    print(f"[DEBUG] nll has NaN in batch {idx}")

                nll = nll.sum(-1)
                safe_sums = torch.clamp(sums.squeeze(), min=1e-10)
                loss = nll / safe_sums
                if torch.isnan(loss).any():
                    print(f"[DEBUG] loss has NaN in batch {idx}")
                loss = loss.mean().item()
                if math.isnan(loss):
                    print(f"[DEBUG] loss is NaN in batch {idx}")
                acc_loss += loss
                cnt += 1
            cur_loss = acc_loss / max(cnt, 1)  # avoid dividing by zero
            # limit curr 
            cur_loss = min(cur_loss, 100) 
            ppl_all = round(math.exp(cur_loss), 1)
            print('*'*100)
            print('{} PPL: {}'.format(source.upper(), ppl_all))
            print('*'*100)
            return ppl_all
        else: 
            indices = torch.split(torch.tensor(range(args.num_docs_test)), args.eval_batch_size)
            tokens_1 = test_1_tokens
            counts_1 = test_1_counts

            tokens_2 = test_2_tokens
            counts_2 = test_2_counts

            eta_1 = get_eta('test')

            acc_loss = 0
            cnt = 0
            indices = torch.split(torch.tensor(range(args.num_docs_test)), args.eval_batch_size)
            for idx, ind in enumerate(indices):
                data_batch_1, times_batch_1 = data.get_batch(
                    tokens_1, counts_1, ind, args.vocab_size, args.emb_size, temporal=True, times=test_times)
                

                sums_1 = data_batch_1.sum(1).unsqueeze(1)
                if args.bow_norm:
                    normalized_data_batch_1 = data_batch_1 / sums_1
                else:
                    normalized_data_batch_1 = data_batch_1

                times_batch_1_long = times_batch_1.long()
                eta_td_1 = eta_1[times_batch_1_long]
                theta = get_theta(eta_td_1, normalized_data_batch_1)

                data_batch_2, times_batch_2 = data.get_batch(
                    tokens_2, counts_2, ind, args.vocab_size, args.emb_size, temporal=True, times=test_times)
                sums_2 = data_batch_2.sum(1).unsqueeze(1)

                times_batch_2_long = times_batch_2.long()
                alpha_td = alpha[:, times_batch_2_long, :]
                beta = model.get_beta(alpha_td).permute(1, 0, 2)
                loglik = theta.unsqueeze(2) * beta
                loglik = loglik.sum(1)
                # Add a small epsilon to avoid log(0)
                loglik = torch.log(loglik + 1e-10)
                nll = -loglik * data_batch_2
                nll = nll.sum(-1)
                # Ensure the divisor is nonzero
                safe_sums = torch.clamp(sums_2.squeeze(), min=1e-10)
                loss = nll / safe_sums
                loss = loss.mean().item()
                acc_loss += loss
                cnt += 1
            cur_loss = acc_loss / max(cnt, 1)  # Avoid division by zero
            # Bound cur_loss to avoid overflow in exp
            cur_loss = min(cur_loss, 100)  # Cap the maximum value
            ppl_dc = round(math.exp(cur_loss), 1)
            print('*'*100)
            print('{} Doc Completion PPL: {}'.format(source.upper(), ppl_dc))
            print('*'*100)
            return ppl_dc

def _diversity_helper(beta, num_tops):
    """
    beta: shape (K, V) for one time slice
    returns topic diversity = unique top words / (K * num_tops)
    """
    list_w = np.zeros((args.num_topics, num_tops), dtype=np.int64)
    for k in range(args.num_topics):
        gamma = beta[k, :]
        top_words = gamma.cpu().numpy().argsort()[-num_tops:][::-1]
        list_w[k, :] = top_words

    n_unique = len(np.unique(list_w.reshape(-1)))
    diversity = n_unique / (args.num_topics * num_tops)
    return float(diversity)

def get_topic_quality():
    """
    Final DETM-style evaluation function.

    - TD: top-25 diversity, averaged over time slices
    - TC: top-10 PMI coherence, computed within each time slice using
          only documents from that time slice, then averaged over time
    - Topic Quality = TD * TC
    """
    model.eval()
    with torch.no_grad():
        alpha = model.mu_q_alpha
        beta = model.get_beta(alpha)   # (K, T, V)

        print('beta:', beta.size())
        print('\n' + '#' * 100)

        # ---- Topic Diversity ----
        print('Get topic diversity...')
        num_tops_div = 25
        TD_all = []

        for tt in range(args.num_times):
            beta_t = beta[:, tt, :]   # (K, V)
            td_t = _diversity_helper(beta_t, num_tops_div)
            TD_all.append(td_t)

        TD = float(np.mean(TD_all))
        print(f'Topic Diversity is: {TD:.6f}')

        # ---- Topic Coherence ----
        print('\nGet topic coherence...')
        num_tops_coh = 10
        TC_all = []

        train_times_np = np.asarray(train_times)

        for tt in range(args.num_times):
            beta_t = beta[:, tt, :]   # (K, V)

            # Only use docs from this time slice
            doc_idx_t = np.where(train_times_np == tt)[0]
            docs_t = [train_tokens[i] for i in doc_idx_t]

            tc_t, _ = get_topic_coherence(
                beta_t.cpu().numpy(),
                docs_t,
                vocab=vocab,
                top_n=num_tops_coh
            )
            TC_all.append(tc_t)

        TC = float(np.nanmean(TC_all))
        print(f'Topic Coherence is: {TC:.6f}')

        # ---- Topic Quality ----
        quality = TD * TC
        print('\nGet topic quality...')
        print(f'Topic Quality is: {quality:.6f}')
        print('#' * 100)

        return TD, TC, quality


if args.mode == 'train':
    ## train model on data by looping through multiple epochs
    best_epoch = 0
    best_val_ppl = 1e9
    all_val_ppls = []
    for epoch in range(1, args.epochs):
        train(epoch)
        if epoch % args.visualize_every == 0:
            visualize()
            with torch.no_grad():
                # Compute per-source metrics if multi-source dataset
                if args.num_sources > 1:
                    use_src_beta = args.source_adaptation_mode and model.source_adaptation_mode
                    stats = quick_topic_stats_per_source(
                        model, vocab, train_tokens, train_times, 
                        train_src_ids, src_names,
                        use_source_specific_beta=use_src_beta
                    )
                    print(f"[Topic-Stats] Overall: TD={stats['overall']['TD']:.3f} TC={stats['overall']['TC']:.3f}")
                    for src_name in src_names:
                        print(f"  {src_name:6s}: TD={stats[src_name]['TD']:.3f} TC={stats[src_name]['TC']:.3f}")
                    
                    # Comprehensive diagnostics if in adaptation mode
                    if args.source_adaptation_mode and model.delta_alpha is not None:
                        print("\n[Delta-Alpha Diagnostics]")
                        for src_id, src_name in enumerate(src_names):
                            delta_norm = torch.norm(model.delta_alpha[src_id]).item()
                            print(f"  {src_name:6s}: ||delta_alpha|| = {delta_norm:.4f}")
                        
                        # Pairwise beta distances
                        print("\n[Pairwise Beta Distances (JS divergence)]")
                        distances = compute_beta_distances(model, metric='js')
                        for (i, j), dist in distances.items():
                            print(f"  {src_names[i]} <-> {src_names[j]}: {dist:.4f}")
                        
                        # Topic word comparison for a few representative topics
                        print("\n[Topic Word Comparison]")
                        # Sample topics: business (topic 1), labor (topic 3), one mid-time
                        sample_topics = [(1, args.num_times//2, 'Business-Mid'),
                                        (3, args.num_times//2, 'Labor-Mid')]
                        
                        for topic_id, time_id, label in sample_topics:
                            if topic_id < args.num_topics and time_id < args.num_times:
                                print(f"\n  {label} (Topic {topic_id}, Time {time_id}):")
                                topic_words = get_source_topic_words(model, vocab, topic_id, time_id, num_words=10)
                                print(f"    Shared : {', '.join(topic_words['shared'][:10])}")
                                for src_id, src_name in enumerate(src_names):
                                    src_key = f'source_{src_id}'
                                    if src_key in topic_words:
                                        words = topic_words[src_key]
                                        print(f"    {src_name:6s}: {', '.join(words[:10])}")
                else:
                    # Single-source or non-adaptation mode
                    beta_now = model.get_beta(model.mu_q_alpha)
                    td, tc = quick_topic_stats(beta_now, vocab, train_tokens, train_times)
                    print(f"[Topic-Stats] TD={td:.3f} TC={tc:.3f}")
                
                print()  # Empty line for readability

        try:
            val_ppl = get_completion_ppl('val')
            print('val_ppl: ', val_ppl)
            
            # check if val_ppl is NaN
            is_valid_ppl = not (math.isnan(val_ppl) or math.isinf(val_ppl))
            
            # if val_ppl is valid and is better than the current optimal
            # or the current optimal is NaN，save the model
            if is_valid_ppl and (val_ppl < best_val_ppl or math.isnan(best_val_ppl)):
                with open(ckpt, 'wb') as f:
                    torch.save(model, f)
                best_epoch = epoch
                best_val_ppl = val_ppl
                print(f'Model saved at epoch {epoch} with val_ppl: {val_ppl}')
            
            # Save checkpoint at regular intervals if configured
            if args.save_checkpoint_every > 0 and epoch % args.save_checkpoint_every == 0:
                checkpoint_path = f"{ckpt}_epoch_{epoch}"
                with open(checkpoint_path, 'wb') as f:
                    torch.save(model, f)
                print(f'Regular checkpoint saved at epoch {epoch}')
        except Exception as e:
            print(f'Error during validation: {e}')
            # even when there's error, stilll save the model checkpoint
            checkpoint_path = f"{ckpt}_epoch_{epoch}_error"
            with open(checkpoint_path, 'wb') as f:
                torch.save(model, f)
            print(f'Error checkpoint saved at epoch {epoch}')
        else:
            ## check whether to anneal lr
            lr = optimizer.param_groups[0]['lr']
            if args.anneal_lr and (len(all_val_ppls) > args.nonmono and val_ppl > min(all_val_ppls[:-args.nonmono]) and lr > 1e-5):
                optimizer.param_groups[0]['lr'] /= args.lr_factor
        all_val_ppls.append(val_ppl)
    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        print('saving topic matrix beta...')
        alpha = model.mu_q_alpha
        beta = model.get_beta(alpha).cpu().numpy()
        scipy.io.savemat(ckpt+'_beta.mat', {'values': beta}, do_compression=True)
        if args.train_embeddings:
            print('saving word embedding matrix rho...')
            rho = model.rho.weight.cpu().numpy()
            scipy.io.savemat(ckpt+'_rho.mat', {'values': rho}, do_compression=True)
        print('computing validation perplexity...')
        val_ppl = get_completion_ppl('val')
        print('computing test perplexity...')
        test_ppl = get_completion_ppl('test')
else: 
    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)
        
    print('saving alpha...')
    with torch.no_grad():
        alpha = model.mu_q_alpha.cpu().numpy()
        scipy.io.savemat(ckpt+'_alpha.mat', {'values': alpha}, do_compression=True)

    print('computing validation perplexity...')
    val_ppl = get_completion_ppl('val')
    print('computing test perplexity...')
    test_ppl = get_completion_ppl('test')
    print('computing topic coherence and topic diversity...')
    get_topic_quality()
    print('visualizing topics and embeddings...')
    visualize()
