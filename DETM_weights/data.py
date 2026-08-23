import os
import random
import pickle
import numpy as np
import torch 
import scipy.io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _fetch(path, name):
    if name == 'train':
        token_file = os.path.join(path, 'bow_tr_tokens.mat')
        count_file = os.path.join(path, 'bow_tr_counts.mat')
    elif name == 'valid':
        token_file = os.path.join(path, 'bow_va_tokens.mat')
        count_file = os.path.join(path, 'bow_va_counts.mat')
    else:
        token_file = os.path.join(path, 'bow_ts_tokens.mat')
        count_file = os.path.join(path, 'bow_ts_counts.mat')
    tokens = scipy.io.loadmat(token_file)['tokens'].squeeze()
    counts = scipy.io.loadmat(count_file)['counts'].squeeze()
    if name == 'test':
        token_1_file = os.path.join(path, 'bow_ts_h1_tokens.mat')
        count_1_file = os.path.join(path, 'bow_ts_h1_counts.mat')
        token_2_file = os.path.join(path, 'bow_ts_h2_tokens.mat')
        count_2_file = os.path.join(path, 'bow_ts_h2_counts.mat')
        tokens_1 = scipy.io.loadmat(token_1_file)['tokens'].squeeze()
        counts_1 = scipy.io.loadmat(count_1_file)['counts'].squeeze()
        tokens_2 = scipy.io.loadmat(token_2_file)['tokens'].squeeze()
        counts_2 = scipy.io.loadmat(count_2_file)['counts'].squeeze()
        return {'tokens': tokens, 'counts': counts, 'tokens_1': tokens_1, 'counts_1': counts_1, 'tokens_2': tokens_2, 'counts_2': counts_2}
    return {'tokens': tokens, 'counts': counts}


def _fetch_temporal(path, name):
    if name == 'train':
        pref = 'tr'
    elif name == 'valid':
        pref = 'va'
    else:
        pref = 'ts'         

    token_file = os.path.join(path, f'bow_{pref}_tokens.mat')
    count_file = os.path.join(path, f'bow_{pref}_counts.mat')
    time_file  = os.path.join(path, f'bow_{pref}_timestamps.mat')
    src_file   = os.path.join(path, f'bow_{pref}_sources.mat')   

    tokens  = scipy.io.loadmat(token_file)['tokens'    ].squeeze()
    counts  = scipy.io.loadmat(count_file)['counts'    ].squeeze()
    times   = scipy.io.loadmat(time_file )['timestamps'].squeeze()
    sources = scipy.io.loadmat(src_file )['sources'    ].squeeze()  

    if name == 'test':         
        token_1_file = os.path.join(path, 'bow_ts_h1_tokens.mat')
        count_1_file = os.path.join(path, 'bow_ts_h1_counts.mat')
        token_2_file = os.path.join(path, 'bow_ts_h2_tokens.mat')
        count_2_file = os.path.join(path, 'bow_ts_h2_counts.mat')
        tokens_1 = scipy.io.loadmat(token_1_file)['tokens'].squeeze()
        counts_1 = scipy.io.loadmat(count_1_file)['counts'].squeeze()
        tokens_2 = scipy.io.loadmat(token_2_file)['tokens'].squeeze()
        counts_2 = scipy.io.loadmat(count_2_file)['counts'].squeeze()
        return {'tokens': tokens, 'counts': counts, 'times': times,
                'sources': sources,                       
                'tokens_1': tokens_1, 'counts_1': counts_1,
                'tokens_2': tokens_2, 'counts_2': counts_2}

    return {'tokens': tokens, 'counts': counts, 'times': times,
            'sources': sources}                             


def get_data(path, temporal=False):
    ### load vocabulary
    with open(os.path.join(path, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    if not temporal:
        train = _fetch(path, 'train')
        valid = _fetch(path, 'valid')
        test = _fetch(path, 'test')
    else:
        train = _fetch_temporal(path, 'train')
        valid = _fetch_temporal(path, 'valid')
        test = _fetch_temporal(path, 'test')

    return vocab, train, valid, test

def get_batch(tokens, counts, ind, vocab_size, emsize=300, temporal=False, times=None):
    """fetch input data by batch."""
    batch_size = len(ind)
    data_batch = np.zeros((batch_size, vocab_size))
    if temporal:
        times_batch = np.zeros((batch_size, ), dtype=np.int64)
    for i, doc_id in enumerate(ind):
        doc = tokens[doc_id]
        count = counts[doc_id]
        if temporal:
            timestamp = times[doc_id]
            times_batch[i] = timestamp
        if len(doc) == 1: 
            doc = [doc.squeeze()]
            count = [count.squeeze()]
        else:
            doc = doc.squeeze()
            count = count.squeeze()
        if doc_id != -1:
            for j, word in enumerate(doc):
                data_batch[i, word] = count[j]
    data_batch = torch.from_numpy(data_batch).float().to(device)
    if temporal:
        times_batch = torch.from_numpy(times_batch).long().to(device)
        return data_batch, times_batch
    return data_batch

def get_rnn_input(tokens, counts, times, num_times, vocab_size, num_docs):
    """
    Corpus-level lexical summary. 
    """
    indices = torch.randperm(num_docs)
    indices = torch.split(indices, 1000) 
    rnn_input = torch.zeros(num_times, vocab_size, device=device)
    cnt = torch.zeros(num_times, device=device)
    for idx, ind in enumerate(indices):
        data_batch, times_batch = get_batch(tokens, counts, ind, vocab_size, temporal=True, times=times)
        for t in range(num_times):
            idx_t = (times_batch == t).nonzero(as_tuple=True)[0]
            if idx_t.numel() == 0:
                continue
            rnn_input[t] += data_batch[idx_t].sum(0)
            cnt[t] += idx_t.numel()
        if idx % 20 == 0:
            print('idx: {}/{}'.format(idx, len(indices)))
    # rnn_input = rnn_input / cnt.unsqueeze(1)
    # indicating that the year is missing 
    nonzero = cnt > 0 
    rnn_input[nonzero] /= cnt[nonzero].unsqueeze(1)
    return rnn_input
