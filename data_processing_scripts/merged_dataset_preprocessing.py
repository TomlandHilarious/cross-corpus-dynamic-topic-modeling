#!/user/rl3403/.conda/envs/nlp_kogut/bin/python

"""
Merged dataset preprocessing script with corpus-specific handling.

Features:
- Takes root_folder as command-line argument
- Handles COHA, HBR, ILR differently based on document length
- Provides downsampling for COHA to reduce noise
- Reports corpus-specific statistics

Usage: 
  ./merged_dataset_preprocessing.py [root_folder] [output_folder] [downsample_coha_ratio]
  
  Examples:
  ./merged_dataset_preprocessing.py Merged_1920plus_v2_phrase merged_v2_phrase 0.5
  ./merged_dataset_preprocessing.py Merged_1920plus merged_standard 1.0
"""

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pickle
import random
from scipy import sparse
import itertools
from scipy.io import savemat, loadmat
import string
import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from gensim.models import Word2Vec

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process merged corpus data with corpus-specific handling')
parser.add_argument('root_folder', nargs='?', default='Merged_1920plus', 
                    help='Root folder containing the merged corpus data')
parser.add_argument('out_folder', nargs='?', default='merged_max_df_0.6',
                    help='Output folder for processed data')
parser.add_argument('--downsample_coha', type=float, default=0.3,
                    help='Downsampling ratio for COHA corpus (0.1-1.0)')
parser.add_argument('--min_doc_len_coha', type=int, default=50,
                    help='Minimum document length for COHA corpus')
parser.add_argument('--min_doc_len_hbr', type=int, default=50,
                    help='Minimum document length for HBR corpus')
parser.add_argument('--min_doc_len_ilr', type=int, default=50,
                    help='Minimum document length for ILR corpus')
parser.add_argument('--min_df', type=int, default=10, 
                    help='Minimum document frequency for vocabulary')
parser.add_argument('--max_df', type=float, default=0.6,
                    help='Maximum document frequency for vocabulary (0.0-1.0)')
args = parser.parse_args()

# Configuration
root_folder = args.root_folder
YEAR_MIN, YEAR_MAX = 1922, 2019
min_df, max_df = args.min_df, args.max_df
out_folder = args.out_folder
downsample_coha_ratio = args.downsample_coha

# Make sure downsampling ratio is valid
if not (0.0 < downsample_coha_ratio <= 1.0):
    print(f"Warning: Invalid downsampling ratio {downsample_coha_ratio}. Using 1.0 (no downsampling)")
    downsample_coha_ratio = 1.0

# Create output directory
print(f"Processing {root_folder} with COHA downsampling ratio: {downsample_coha_ratio}")
print(f"Output folder: {out_folder}")
os.makedirs(out_folder, exist_ok=True)

# Set minimum document lengths by corpus from command line args
min_doc_lengths = {
    'COHA': args.min_doc_len_coha,
    'HBR': args.min_doc_len_hbr,
    'ILR': args.min_doc_len_ilr
}
print(f"Minimum document lengths: COHA={min_doc_lengths['COHA']}, HBR={min_doc_lengths['HBR']}, ILR={min_doc_lengths['ILR']}")


# remove non-printable characters
def remove_not_printable(in_str):
    return "".join([c for c in in_str if c in string.printable])

print('reading merged meta-data …')
all_filepaths, all_timestamps, all_sources = [], [], []

for year_folder in sorted(os.listdir(root_folder)):
    # skip 1920, 1921, 2020+
    if not (year_folder.isdigit() and YEAR_MIN <= int(year_folder) <= YEAR_MAX):
        continue                                   
    year_dir = os.path.join(root_folder, year_folder)
    for fname in os.listdir(year_dir):
        if not fname.endswith('.txt'):
            continue
        full = os.path.join(year_dir, fname)
        src  = fname.split('_', 1)[0]              # COHA / HBR / ILR
        all_filepaths.append(full)
        all_timestamps.append(year_folder)    
        all_sources.append(src)                    # keep for sample-weight later

print(f'  total files: {len(all_filepaths)}')

# Group files by source for corpus-specific handling
files_by_source = {'COHA': [], 'HBR': [], 'ILR': []}
times_by_source = {'COHA': [], 'HBR': [], 'ILR': []}

for path, tt, ss in zip(all_filepaths, all_timestamps, all_sources):
    if ss in files_by_source:
        files_by_source[ss].append((path, tt))
        
# Print initial corpus statistics
print('\nInitial corpus statistics:')
for src, files in files_by_source.items():
    print(f"  - {src}: {len(files)} files")

# Implement COHA downsampling if requested
if downsample_coha_ratio < 1.0 and 'COHA' in files_by_source:
    original_count = len(files_by_source['COHA'])
    # Stratify by year to maintain temporal distribution
    coha_by_year = defaultdict(list)
    for path, tt in files_by_source['COHA']:
        coha_by_year[tt].append((path, tt))
    
    # Downsample each year proportionally
    downsampled_coha = []
    for year, files in coha_by_year.items():
        sample_size = max(1, int(len(files) * downsample_coha_ratio))
        downsampled_coha.extend(random.sample(files, sample_size))
    
    files_by_source['COHA'] = downsampled_coha
    print(f"\nCOHA downsampling: {original_count} → {len(files_by_source['COHA'])} files ({downsample_coha_ratio:.2f} ratio)")

# read raw data
# recording the filepaths, timestamps, and sources of the document
print('\nReading and processing documents...')
docs = []
not_found = []
timestamps = []
sources = [] 
keep_paths = []

# Statistics for reporting
len_stats = defaultdict(list)
kept_counts = Counter()
filtered_counts = Counter()

# Process all files with source-specific handling
for src in files_by_source:
    min_len = min_doc_lengths.get(src, 10)  # Default to 10 if not specified
    print(f"Processing {src} files (min length: {min_len})...")
    
    for path, tt in files_by_source[src]:
        if not os.path.isfile(path):
            not_found.append(path)
            continue
            
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                doc = f.read()
                
            # Files are already preprocessed - just split by whitespace
            words = doc.split()
            len_stats[src].append(len(words))
  
            # Keep all documents for now - will filter by vocabulary-based length later
            docs.append(doc)
            timestamps.append(tt)
            sources.append(src)
            keep_paths.append(path)
            kept_counts[src] += 1
        except Exception as e:
            print(f"Error processing {path}: {e}")

# Print initial processing statistics (before vocabulary filtering)
print("\nInitial document processing statistics (before vocabulary filtering):")
for src in sorted(len_stats.keys()):
    lengths = len_stats[src]
    if lengths:
        print(f"  {src}:")
        print(f"    - Documents: {kept_counts[src]} documents")
        print(f"    - Length stats: min={min(lengths)}, mean={np.mean(lengths):.1f}, median={np.median(lengths):.1f}, max={max(lengths)}")

print(f"\nTotal documents before vocabulary filtering: {len(docs)}")
print(f"Documents not found: {len(not_found)}")

# Write as raw text

print('writing to text file...')
out_filename = os.path.join(out_folder, 'merged_docs_processed.txt')

with open(out_filename, 'w') as f:
    for line in docs:
        f.write(line + '\n')
# Read stopwords
# /shared/share_hbr-ilr_nlp/DETM_advanced/DETM/scripts/stops.txt
with open("/shared/share_hbr-ilr_nlp/DETM_advanced/DETM/scripts/stops.txt", 'r') as f:
    stops = f.read().split('\n')

# Create count vectorizer 
print(f'counting document frequency of words using min_df={min_df}, max_df={max_df}')
cvectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
cvz = cvectorizer.fit_transform(docs).sign()
# Get vocabulary
print('building the vocabulary...')

sum_counts = cvz.sum(axis=0)
v_size = sum_counts.shape[1]
sum_counts_np = np.zeros(v_size, dtype=int)
for v in range(v_size):
    sum_counts_np[v] = sum_counts[0,v]
word2id = dict([(w, cvectorizer.vocabulary_.get(w)) for w in cvectorizer.vocabulary_])
id2word = dict([(cvectorizer.vocabulary_.get(w), w) for w in cvectorizer.vocabulary_])

print('  initial vocabulary size: {}'.format(v_size))
# Sort elements in vocabulary
idx_sort = np.argsort(sum_counts_np)
vocab_aux = [id2word[idx_sort[cc]] for cc in range(v_size)]


# Filter out stopwords (if any)
vocab_aux = [w for w in vocab_aux if w not in stops]
print('  vocabulary size after removing stopwords from list: {}'.format(len(vocab_aux)))
# Create dictionary and inverse dictionary
vocab = vocab_aux
del vocab_aux
word2id = dict([(w, j) for j, w in enumerate(vocab)])
id2word = dict([(j, w) for j, w in enumerate(vocab)])



# Create mapping of timestamps
all_times = sorted(set(timestamps))
time2id = dict([(t, i) for i, t in enumerate(all_times)])
id2time = dict([(i, t) for i, t in enumerate(all_times)])
time_list = [id2time[i] for i in range(len(all_times))]

# Now filter documents by vocabulary-based length (after vocabulary processing)
print('\nFiltering documents by vocabulary-based minimum length...')
vocab_filtered_docs = []
vocab_filtered_timestamps = []
vocab_filtered_sources = []
vocab_filtered_paths = []

final_kept_counts = Counter()
final_filtered_counts = Counter()
final_len_stats = defaultdict(list)

for i, (doc, timestamp, source, path) in enumerate(zip(docs, timestamps, sources, keep_paths)):
    # Count words that are in the final vocabulary using CountVectorizer's tokenization
    cv_tokens = cvectorizer.build_analyzer()(doc)  # Use CountVectorizer's tokenizer
    vocab_words = [w for w in cv_tokens if w in word2id]
    vocab_length = len(vocab_words)
    
    # Apply source-specific minimum length on vocabulary-filtered content
    min_len = min_doc_lengths.get(source, 50)  # Use 50 as default since we updated all to 50
    if vocab_length >= min_len:
        vocab_filtered_docs.append(doc)
        vocab_filtered_timestamps.append(timestamp)
        vocab_filtered_sources.append(source)
        vocab_filtered_paths.append(path)
        final_kept_counts[source] += 1
        final_len_stats[source].append(vocab_length)  # Only add stats for kept documents
    else:
        final_filtered_counts[source] += 1

# Replace original lists with filtered ones
docs = vocab_filtered_docs
timestamps = vocab_filtered_timestamps  
sources = vocab_filtered_sources
keep_paths = vocab_filtered_paths

# Print final statistics
print("\nFinal document statistics (after vocabulary filtering):")
for src in sorted(final_len_stats.keys()):
    lengths = final_len_stats[src]
    if lengths:
        print(f"  {src}:")
        print(f"    - Documents: {final_kept_counts[src]} kept, {final_filtered_counts[src]} filtered")
        print(f"    - Vocab-based length stats: min={min(lengths)}, mean={np.mean(lengths):.1f}, median={np.median(lengths):.1f}, max={max(lengths)}")

print(f"\nTotal documents after vocabulary filtering: {len(docs)}")

# Create unified analyzer for consistent tokenization
analyzer = cvectorizer.build_analyzer()
print('Using CountVectorizer analyzer for consistent tokenization across pipeline')

# Split in train/test/valid (after vocabulary-based filtering)
print('tokenizing documents and splitting into train/test/valid…')
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import pandas as pd  

# ---generate meta----
meta = pd.DataFrame({
    "chunk_idx": np.arange(len(docs)),          
    "fname"    : keep_paths,                 
    "year"     : [int(t) for t in timestamps],  # label-1
    "source"   : sources                        # label-2
})

# --- remove "_chunk\d+.txt", get origianl file ID ---
meta["doc_id"] = meta["fname"].str.replace(r"_chunk\d+\.txt$", "", regex=True)   ### <--

# ========= 2. 基于 doc_id 做分层 ==========
files = meta.groupby("doc_id").agg(
            year   = ("year",   "first"),
            source = ("source", "first"))
X_files = files.index.values.reshape(-1, 1)
y_files = np.vstack([files['year'],
                     pd.Categorical(files['source']).codes]).T
msss = MultilabelStratifiedShuffleSplit(test_size=0.15, random_state=123)
train_files, test_files = next(msss.split(X_files, y_files))

msss2 = MultilabelStratifiedShuffleSplit(test_size=0.05/0.85, random_state=123)
sub_train, valid_sub = next(msss2.split(X_files[train_files], y_files[train_files]))

valid_files = train_files[valid_sub]
train_files = train_files[sub_train]

# ========= map back to chunk indexing ==========
def files_to_chunks(file_idx_arr):
    doc_ids = files.index[file_idx_arr]
    return meta.loc[meta['doc_id'].isin(doc_ids), 'chunk_idx'].to_numpy(dtype=np.int64)

train_idx = files_to_chunks(train_files)
valid_idx = files_to_chunks(valid_files)
test_idx  = files_to_chunks(test_files)

trSize, vaSize, tsSize = map(len, [train_idx, valid_idx, test_idx])

print(f'  number of documents (train): {trSize}')
print(f'  number of documents (test) : {tsSize}')
print(f'  number of documents (valid): {vaSize}')

# recording mapping to both timestamps and sources
# Use unified analyzer for consistency
vocab = list(set([w
                  for i in train_idx
                  for w in analyzer(docs[i])
                  if w in word2id]))
word2id = {w: j for j, w in enumerate(vocab)}
id2word = {j: w for j, w in enumerate(vocab)}
print('  vocabulary after removing words not in train: {}'.format(len(vocab)))

#Train Skip-Gram Word2Vec Embedding
print('Training skip-gram word2vec embeddings with Gensim...')
# Use unified analyzer for consistency
tokenized_docs = [analyzer(doc) for doc in docs]
w2v_model = Word2Vec(
    sentences=tokenized_docs,
    vector_size=300,      
    window=10,
    sg=1,                 # sg=1 means skip-gram
    min_count=10,
    workers=108,
    epochs=10
)
embedding_matrix = np.zeros((len(vocab), 300))
for word, idx in word2id.items():
    if word in w2v_model.wv:
        embedding_matrix[idx] = w2v_model.wv[word]
    else:
        embedding_matrix[idx] = np.random.normal(scale=0.6, size=(300,))


path_save = os.path.join(out_folder, f"min_df_{min_df}")
os.makedirs(path_save, exist_ok=True)
np.save(os.path.join(path_save, 'merged_embedding.npy'), embedding_matrix)
print(f'Embedding matrix saved to {path_save}/merged_embedding.npy')

# Use unified analyzer for consistency
docs_tr = [[word2id[w] for w in analyzer(docs[i]) if w in word2id] for i in train_idx]
timestamps_tr = [time2id[timestamps[i]] for i in train_idx]
sources_tr = [sources[i] for i in train_idx] 

docs_ts = [[word2id[w] for w in analyzer(docs[i]) if w in word2id] for i in test_idx]
timestamps_ts = [time2id[timestamps[i]] for i in test_idx]
sources_ts = [sources[i] for i in test_idx]

docs_va = [[word2id[w] for w in analyzer(docs[i]) if w in word2id] for i in valid_idx]
timestamps_va = [time2id[timestamps[i]] for i in valid_idx]
sources_va = [sources[i] for i in valid_idx]



print('  number of documents (train): {} [this should be equal to {} and {}]'.format(
      len(docs_tr), trSize, len(timestamps_tr)))
print('  number of documents (test):  {} [this should be equal to {} and {}]'.format(
      len(docs_ts), tsSize, len(timestamps_ts)))
print('  number of documents (valid): {} [this should be equal to {} and {}]'.format(
      len(docs_va), vaSize, len(timestamps_va)))



# Remove empty documents
print('removing empty documents...')

def remove_empty(in_docs, in_timestamps, in_sources):
    out_docs, out_ts, out_src = [], [], []
    for d, t, s in zip(in_docs, in_timestamps, in_sources):
        if d:
            out_docs.append(d); out_ts.append(t); out_src.append(s)
    return out_docs, out_ts, out_src

def remove_by_threshold(in_docs, in_timestamps, in_sources, thr):
    out_docs, out_ts, out_src = [], [], []
    for d, t, s in zip(in_docs, in_timestamps, in_sources):
        if len(d) > thr:
            out_docs.append(d); out_ts.append(t); out_src.append(s)
    return out_docs, out_ts, out_src


docs_tr, timestamps_tr, sources_tr = remove_empty(docs_tr, timestamps_tr, sources_tr)
docs_ts, timestamps_ts, sources_ts = remove_empty(docs_ts, timestamps_ts, sources_ts)
docs_va, timestamps_va, sources_va = remove_empty(docs_va, timestamps_va, sources_va)

# Remove test documents with length=1
docs_ts, timestamps_ts, sources_ts = remove_by_threshold(docs_ts, timestamps_ts, sources_ts, 1)

# Split test set in 2 halves
print('splitting test documents in 2 halves...')
docs_ts_h1 = [[w for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for doc in docs_ts]
docs_ts_h2 = [[w for i,w in enumerate(doc) if i>len(doc)/2.0-1] for doc in docs_ts]
sources_ts_h1 = sources_ts  
sources_ts_h2 = sources_ts 

# Getting lists of words and doc_indices
print('creating lists of words...')

def create_list_words(in_docs):
    return [x for y in in_docs for x in y]

words_tr = create_list_words(docs_tr)
words_ts = create_list_words(docs_ts)
words_ts_h1 = create_list_words(docs_ts_h1)
words_ts_h2 = create_list_words(docs_ts_h2)
words_va = create_list_words(docs_va)

print('  len(words_tr): ', len(words_tr))
print('  len(words_ts): ', len(words_ts))
print('  len(words_ts_h1): ', len(words_ts_h1))
print('  len(words_ts_h2): ', len(words_ts_h2))
print('  len(words_va): ', len(words_va))

# Get doc indices
print('getting doc indices...')

def create_doc_indices(in_docs):
    aux = [[j for i in range(len(doc))] for j, doc in enumerate(in_docs)]
    return [int(x) for y in aux for x in y]

doc_indices_tr = create_doc_indices(docs_tr)
doc_indices_ts = create_doc_indices(docs_ts)
doc_indices_ts_h1 = create_doc_indices(docs_ts_h1)
doc_indices_ts_h2 = create_doc_indices(docs_ts_h2)
doc_indices_va = create_doc_indices(docs_va)

print('  len(np.unique(doc_indices_tr)): {} [this should be {}]'.format(len(np.unique(doc_indices_tr)), len(docs_tr)))
print('  len(np.unique(doc_indices_ts)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts)), len(docs_ts)))
print('  len(np.unique(doc_indices_ts_h1)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts_h1)), len(docs_ts_h1)))
print('  len(np.unique(doc_indices_ts_h2)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts_h2)), len(docs_ts_h2)))
print('  len(np.unique(doc_indices_va)): {} [this should be {}]'.format(len(np.unique(doc_indices_va)), len(docs_va)))

# Number of documents in each set
n_docs_tr = len(docs_tr)
n_docs_ts = len(docs_ts)
n_docs_ts_h1 = len(docs_ts_h1)
n_docs_ts_h2 = len(docs_ts_h2)
n_docs_va = len(docs_va)

# Remove unused variables
del docs_tr
del docs_ts
del docs_ts_h1
del docs_ts_h2
del docs_va

# Create bow representation

print('creating bow representation...')

def create_bow(doc_indices, words, n_docs, vocab_size):
    return sparse.coo_matrix(([1]*len(doc_indices),(doc_indices, words)), shape=(n_docs, vocab_size)).tocsr()

bow_tr = create_bow(doc_indices_tr, words_tr, n_docs_tr, len(vocab))
bow_ts = create_bow(doc_indices_ts, words_ts, n_docs_ts, len(vocab))
bow_ts_h1 = create_bow(doc_indices_ts_h1, words_ts_h1, n_docs_ts_h1, len(vocab))
bow_ts_h2 = create_bow(doc_indices_ts_h2, words_ts_h2, n_docs_ts_h2, len(vocab))
bow_va = create_bow(doc_indices_va, words_va, n_docs_va, len(vocab))

del words_tr
del words_ts
del words_ts_h1
del words_ts_h2
del words_va
del doc_indices_tr
del doc_indices_ts
del doc_indices_ts_h1
del doc_indices_ts_h2
del doc_indices_va

# Write files for LDA C++ code
def write_lda_file(filename, timestamps_in, time_list_in, bow_in):
    idxSort = np.argsort(timestamps_in)
    
    with open(filename, "w") as f:
        for row in idxSort:
            x = bow_in.getrow(row)
            n_elems = x.count_nonzero()
            f.write(str(n_elems))
            if(n_elems != len(x.indices) or n_elems != len(x.data)):
                raise ValueError("[ERR] THIS SHOULD NOT HAPPEN")
            for ii, dd in zip(x.indices, x.data):
                f.write(' ' + str(ii) + ':' + str(dd))
            f.write('\n')
            
    with open(filename.replace("-mult", "-seq"), "w") as f:
        f.write(str(len(time_list_in)) + '\n')
        for idx_t, _ in enumerate(time_list_in):
            n_elem = len([t for t in timestamps_in if t==idx_t])
            f.write(str(n_elem) + '\n')
            

path_save = out_folder +'/min_df_'+str(min_df)+'/'
if not os.path.isdir(path_save):
    os.system('mkdir -p ' + path_save)

# Write files for LDA C++ code
print('saving LDA files for C++ code...')
write_lda_file(path_save + 'dtm_tr-mult.dat', timestamps_tr, time_list, bow_tr)
write_lda_file(path_save + 'dtm_ts-mult.dat', timestamps_ts, time_list, bow_ts)
write_lda_file(path_save + 'dtm_ts_h1-mult.dat', timestamps_ts, time_list, bow_ts_h1)
write_lda_file(path_save + 'dtm_ts_h2-mult.dat', timestamps_ts, time_list, bow_ts_h2)
write_lda_file(path_save + 'dtm_va-mult.dat', timestamps_va, time_list, bow_va)

# Also write the vocabulary and timestamps and sources
with open(path_save + 'vocab.txt', "w") as f:
    for v in vocab:
        f.write(v + '\n')

with open(path_save + 'timestamps.txt', "w") as f:
    for t in time_list:
        f.write(t + '\n')

with open(path_save + 'vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
del vocab

with open(path_save + 'timestamps.pkl', 'wb') as f:
    pickle.dump(time_list, f)


with open(path_save + 'sources.pkl', 'wb') as f:
    pickle.dump(sources, f)

# Save sources alone
savemat(path_save + 'bow_tr_sources.mat', {'sources': sources_tr}, do_compression=True)
savemat(path_save + 'bow_ts_sources.mat', {'sources': sources_ts}, do_compression=True)
savemat(path_save + 'bow_va_sources.mat', {'sources': sources_va}, do_compression=True)


# Save timestamps alone
savemat(path_save + 'bow_tr_timestamps.mat', {'timestamps': timestamps_tr}, do_compression=True)
savemat(path_save + 'bow_ts_timestamps.mat', {'timestamps': timestamps_ts}, do_compression=True)
savemat(path_save + 'bow_va_timestamps.mat', {'timestamps': timestamps_va}, do_compression=True)

# Split bow intro token/value pairs
print('splitting bow intro token/value pairs and saving to disk...')

def split_bow(bow_in, n_docs):
    indices = [[w for w in bow_in[doc,:].indices] for doc in range(n_docs)]
    counts = [[c for c in bow_in[doc,:].data] for doc in range(n_docs)]
    return indices, counts

bow_tr_tokens, bow_tr_counts = split_bow(bow_tr, n_docs_tr)
savemat(path_save + 'bow_tr_tokens.mat', {'tokens': bow_tr_tokens}, do_compression=True)
savemat(path_save + 'bow_tr_counts.mat', {'counts': bow_tr_counts}, do_compression=True)
del bow_tr
del bow_tr_tokens
del bow_tr_counts

bow_ts_tokens, bow_ts_counts = split_bow(bow_ts, n_docs_ts)
savemat(path_save + 'bow_ts_tokens.mat', {'tokens': bow_ts_tokens}, do_compression=True)
savemat(path_save + 'bow_ts_counts.mat', {'counts': bow_ts_counts}, do_compression=True)
del bow_ts
del bow_ts_tokens
del bow_ts_counts

bow_ts_h1_tokens, bow_ts_h1_counts = split_bow(bow_ts_h1, n_docs_ts_h1)
savemat(path_save + 'bow_ts_h1_tokens.mat', {'tokens': bow_ts_h1_tokens}, do_compression=True)
savemat(path_save + 'bow_ts_h1_counts.mat', {'counts': bow_ts_h1_counts}, do_compression=True)
del bow_ts_h1
del bow_ts_h1_tokens
del bow_ts_h1_counts

bow_ts_h2_tokens, bow_ts_h2_counts = split_bow(bow_ts_h2, n_docs_ts_h2)
savemat(path_save + 'bow_ts_h2_tokens.mat', {'tokens': bow_ts_h2_tokens}, do_compression=True)
savemat(path_save + 'bow_ts_h2_counts.mat', {'counts': bow_ts_h2_counts}, do_compression=True)
del bow_ts_h2
del bow_ts_h2_tokens
del bow_ts_h2_counts

bow_va_tokens, bow_va_counts = split_bow(bow_va, n_docs_va)
savemat(path_save + 'bow_va_tokens.mat', {'tokens': bow_va_tokens}, do_compression=True)
savemat(path_save + 'bow_va_counts.mat', {'counts': bow_va_counts}, do_compression=True)
del bow_va
del bow_va_tokens
del bow_va_counts

print('Data ready !!')
print('*************')