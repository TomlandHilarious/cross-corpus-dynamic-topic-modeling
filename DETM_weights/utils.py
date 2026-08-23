from sklearn.manifold import TSNE
import torch 
import numpy as np
import bokeh.plotting as bp

from bokeh.plotting import save
from bokeh.models import HoverTool
import matplotlib.pyplot as plt 
import matplotlib 

tiny = 1e-6

def _reparameterize(mu, logvar, num_samples):
    """Applies the reparameterization trick to return samples from a given q"""
    std = torch.exp(0.5 * logvar) 
    bsz, zdim = logvar.size()
    eps = torch.randn(num_samples, bsz, zdim).to(mu.device)
    mu = mu.unsqueeze(0)
    std = std.unsqueeze(0)
    res = eps.mul_(std).add_(mu)
    return res

from collections import Counter
from itertools import combinations

def _normalize_doc_tokens(doc):
    """
    Convert one document into a flat Python list of int token ids.
    Compatible with torch tensors / numpy arrays / Python lists.
    """
    if torch.is_tensor(doc):
        doc = doc.detach().cpu().view(-1).tolist()
    elif isinstance(doc, np.ndarray):
        doc = doc.reshape(-1).tolist()
    else:
        doc = list(doc)
    return [int(x) for x in doc]


def get_topic_coherence_npmi(beta_or_top_ids, data, vocab=None, top_n=10, eps=1e-12):
    """
    Document-level NPMI topic coherence.

    NPMI(w_i, w_j) = log(p_ij / (p_i * p_j)) / -log(p_ij)
        - Bounded in [-1, 1]
        - Standard D-ETM / Lau et al. (2014) convention: when pair_df = 0
          assign NPMI = -1 (the lower bound). Do NOT use eps-floor smoothing
          for the joint count, because that injects unbounded large negatives
          into a metric that must remain bounded.

    Parameters
    ----------
    beta_or_top_ids : np.ndarray or torch.Tensor
        Either (K, V) topic-word distributions or (K, top_n) top-word ids.
    data : list of token lists
        One time slice's documents (token ids).
    top_n : int
        Top words per topic for coherence (default 10).
    """
    if isinstance(beta_or_top_ids, torch.Tensor):
        arr = beta_or_top_ids.detach().cpu().numpy()
    else:
        arr = np.asarray(beta_or_top_ids)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    if np.issubdtype(arr.dtype, np.integer):
        top_ids = arr[:, :top_n].astype(np.int64)
    else:
        top_ids = np.argsort(arr, axis=1)[:, -top_n:][:, ::-1].astype(np.int64)

    docs = [_normalize_doc_tokens(doc) for doc in data]
    D = len(docs)
    if D == 0:
        return float("nan"), []

    needed_words = set(top_ids.reshape(-1).tolist())

    df = Counter()
    pair_df = Counter()
    for doc in docs:
        doc_set = set(w for w in doc if w in needed_words)
        for w in doc_set:
            df[w] += 1
        for wi, wj in combinations(sorted(doc_set), 2):
            pair_df[(wi, wj)] += 1

    tc_list = []
    for topic in top_ids:
        scores = []
        for wi, wj in combinations(topic.tolist(), 2):
            wi, wj = sorted((int(wi), int(wj)))
            cnt_ij = pair_df[(wi, wj)]
            if cnt_ij == 0:
                # D-ETM / Lau et al. (2014) standard NPMI convention.
                scores.append(-1.0)
                continue
            p_i  = df[wi] / D
            p_j  = df[wj] / D
            p_ij = cnt_ij / D
            pmi = np.log(p_ij / (p_i * p_j))
            npmi = pmi / (-np.log(p_ij))
            scores.append(float(npmi))
        tc_list.append(float(np.mean(scores)) if scores else float("nan"))
    tc_mean = float(np.nanmean(tc_list))
    return tc_mean, tc_list


def get_topic_coherence(beta_or_top_ids, data, vocab=None, top_n=10, eps=1e-12):
    """
    Paper-style PMI topic coherence.

    Parameters
    ----------
    beta_or_top_ids : np.ndarray or torch.Tensor
        Either:
        - shape (K, V): topic-word distributions
        - shape (K, top_n): precomputed top word ids
    data : list
        List of documents, where each document is a list / tensor of token ids.
        IMPORTANT: for DETM evaluation this should be the docs from ONE time slice.
    vocab : unused, kept only for backward compatibility
    top_n : int
        Number of top words per topic for coherence calculation. Use 10.
    eps : float
        Numerical floor.

    Returns
    -------
    tc_mean : float
        Mean PMI coherence across topics.
    tc_list : list[float]
        Per-topic PMI coherence.
    """
    if isinstance(beta_or_top_ids, torch.Tensor):
        arr = beta_or_top_ids.detach().cpu().numpy()
    else:
        arr = np.asarray(beta_or_top_ids)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    # If integer matrix, assume these are already top word ids.
    # Otherwise assume shape (K, V) and extract top_n ids from topic-word probs.
    if np.issubdtype(arr.dtype, np.integer):
        top_ids = arr[:, :top_n].astype(np.int64)
    else:
        top_ids = np.argsort(arr, axis=1)[:, -top_n:][:, ::-1].astype(np.int64)

    docs = [_normalize_doc_tokens(doc) for doc in data]
    D = len(docs)
    if D == 0:
        return float("nan"), []

    needed_words = set(top_ids.reshape(-1).tolist())

    df = Counter()
    pair_df = Counter()

    for doc in docs:
        doc_set = set(w for w in doc if w in needed_words)
        for w in doc_set:
            df[w] += 1
        for wi, wj in combinations(sorted(doc_set), 2):
            pair_df[(wi, wj)] += 1

    tc_list = []
    for topic in top_ids:
        scores = []
        for wi, wj in combinations(topic.tolist(), 2):
            wi, wj = sorted((int(wi), int(wj)))

            p_i = max(df[wi] / D, eps)
            p_j = max(df[wj] / D, eps)
            p_ij = max(pair_df[(wi, wj)] / D, eps)

            # PMI(wi, wj)
            scores.append(np.log(p_ij / (p_i * p_j)))

        tc_list.append(float(np.mean(scores)) if scores else float("nan"))

    tc_mean = float(np.nanmean(tc_list))
    return tc_mean, tc_list

def log_gaussian(z, mu=None, logvar=None):
    sz = z.size()
    d = z.size(2)
    bsz = z.size(1)
    if mu is None or logvar is None:
        mu = torch.zeros(bsz, d).to(z.device)
        logvar = torch.zeros(bsz, d).to(z.device)
    mu = mu.unsqueeze(0)
    logvar = logvar.unsqueeze(0)
    var = logvar.exp()
    log_density = ((z - mu)**2 / (var+tiny)).sum(2) # b
    log_det = logvar.sum(2) # b
    log_density = log_density + log_det + d*np.log(2*np.pi)
    return -0.5*log_density

def logsumexp(x, dim=0):
    d = torch.max(x, dim)[0]   
    if x.dim() == 1:
        return torch.log(torch.exp(x - d).sum(dim)) + d
    else:
        return torch.log(torch.exp(x - d.unsqueeze(dim).expand_as(x)).sum(dim) + tiny) + d

def flatten_docs(docs): #to get words and doc_indices
    words = [x for y in docs for x in y]
    doc_indices = [[j for _ in doc] for j, doc in enumerate(docs)]
    doc_indices = [x for y in doc_indices for x in y]
    return words, doc_indices
    
def onehot(data, min_length):
    return list(np.bincount(data, minlength=min_length))

def nearest_neighbors(word, embeddings, vocab, num_words):
    vectors = embeddings.cpu().numpy() 
    index = vocab.index(word)
    query = embeddings[index].cpu().numpy() 
    ranks = vectors.dot(query).squeeze()
    denom = query.T.dot(query).squeeze()
    denom = denom * np.sum(vectors**2, 1)
    denom = np.sqrt(denom)
    ranks = ranks / denom
    mostSimilar = []
    [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
    nearest_neighbors = mostSimilar[:num_words]
    nearest_neighbors = [vocab[comp] for comp in nearest_neighbors]
    return nearest_neighbors

def visualize(docs, _lda_keys, topics, theta):
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    # project to 2D
    tsne_lda = tsne_model.fit_transform(theta)
    colormap = []
    for name, hex in matplotlib.colors.cnames.items():
        colormap.append(hex)

    colormap = colormap[:len(theta[0, :])]
    colormap = np.array(colormap)

    title = '20 newsgroups TE embedding V viz'
    num_example = len(docs)

    plot_lda = bp.figure(plot_width=1400, plot_height=1100,
                     title=title,
                     tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                     x_axis_type=None, y_axis_type=None, min_border=1)

    plt.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1],
                 color=colormap[_lda_keys][:num_example])
    plt.show()
