#!/usr/bin/env python

"""
Print out the size for each document.
"""
import os, re, pathlib, csv, matplotlib.pyplot as plt
import numpy as np
ROOT = "COHA_by_year"      # corpus root
YEAR_MIN = 1920            # keep >= 1920
KEEP_GENRE = {"MAG", "NEWS"}
OUT_TSV = "coha_mag_news_lengths.tsv"
def clean(text: str) -> str:
    """basic ASCII clean + lower-case + squeeze whitespace"""
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"\s+", " ", text)
    return text.lower()

def count_tokens(text: str) -> int:
    """count alphabetic tokens only"""
    return sum(1 for w in text.split() if w.isalpha())

lengths, rows = [], []

for year_folder in os.listdir(ROOT):
    if not year_folder.isdigit() or int(year_folder) < YEAR_MIN:
        continue
    year_dir = os.path.join(ROOT, year_folder)
    for fname in os.listdir(year_dir):
        if not fname.endswith(".txt"):
            continue
        parts = fname.split("_")                       # e.g. 1935_MAG_MAG_564933.txt
        if len(parts) < 3:
            continue
        genre = parts[1]
        if genre not in KEEP_GENRE:
            continue
        fpath = os.path.join(year_dir, fname)
        raw = pathlib.Path(fpath).read_text(errors="ignore")
        tokens = count_tokens(clean(raw))
        lengths.append(tokens)
        rows.append((fpath, int(year_folder), genre, tokens))

# --- save tsv ---
with open(OUT_TSV, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["filepath", "year", "genre", "tokens"])
    writer.writerows(rows)
print(f"Saved {len(rows)} records to {OUT_TSV}")

# --- summary statistics ---
lengths_sorted = sorted(lengths)
pct = lambda p: lengths_sorted[int(p * (len(lengths) - 1))]
print("Token count summary:")
print(f"min={lengths_sorted[0]}")
print(f"25%={pct(0.25)}, 50%={pct(0.50)}, 75%={pct(0.75)}")
print(f"max={lengths_sorted[-1]}")

# --- histogram ---
plt.figure(figsize=(6,4))
plt.hist(lengths, bins=100)          # do NOT specify color
plt.xlabel("tokens per article")
plt.ylabel("frequency")
plt.title("COHA MAG+NEWS (>=1920) token distribution")
plt.tight_layout()
plt.show()


# ---------- paths ----------
HBR_ROOT = "HBR"    # <-- revise to your folder
ILR_ROOT = "ILR"
# ---------------------------

def count_tokens(path: str) -> int:
    """Return token count by whitespace split (no cleaning)."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        return len(f.read().split())

def scan_corpus(root: str, out_tsv: str):
    records, lengths = [], []
    for dirpath, _, files in os.walk(root):
        for fname in files:
            if fname.endswith(".txt"):
                fpath = os.path.join(dirpath, fname)
                n_tok = count_tokens(fpath)
                lengths.append(n_tok)
                records.append((fpath, n_tok))

    # save tsv
    with open(out_tsv, "w", newline="") as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerow(["filepath", "tokens"])
        wr.writerows(records)
    print(f"[{root}]  saved {len(records)} rows to {out_tsv}")

    # summary stats
    arr = np.array(lengths)
    q = np.percentile(arr, [0, 25, 50, 75, 100])
    print(f"[{root}]  token summary:")
    print(f"min {q[0]:.0f}, 25% {q[1]:.0f}, median {q[2]:.0f}, "
          f"75% {q[3]:.0f}, max {q[4]:.0f}\n")

if __name__ == "__main__":
    scan_corpus(HBR_ROOT, "hbr_lengths.tsv")
    scan_corpus(ILR_ROOT, "ilr_lengths.tsv")