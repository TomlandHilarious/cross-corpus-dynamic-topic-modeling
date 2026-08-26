#!/usr/bin/env python

# merge_and_chunk.py
"""
Clean + stem + custom-stopword removal, then chunk into 500‑token windows.
 
* **Change 2025‑04‑24‑b**: **do not drop** the tail chunk. If a document is
  longer than 500 tokens, split every 500; the final chunk can be <500 tokens
  (but >0).  Short docs (≤500) are still kept intact.

Outputs →  OUTPUT_ROOT/<year>/<SRC>_<orig>_chunkN.txt
"""

import os
import re
import string
from pathlib import Path
from tqdm import tqdm

# ============== CONFIG ==============
COHA_ROOT   = "/shared/share_hbr-ilr_nlp/COHA_by_year"
HBR_ROOT    = "/shared/share_hbr-ilr_nlp/HBR"
ILR_ROOT    = "/shared/share_hbr-ilr_nlp/ILR"
OUTPUT_ROOT = "Merged_1920plus_v2_phrase"
PHRASE_MAP = {
    ("woman", "work"): "woman_work",
}
YEAR_MIN = 1920
WIN_LEN  = 500     # fixed window size

STOP_PATH = "/shared/share_hbr-ilr_nlp/data_processing_scripts/stops.txt"
# ====================================

# ---------- stopwords ---------------
if Path(STOP_PATH).exists():
    STOPS = set(Path(STOP_PATH).read_text().split())
else:
    print(f"[WARN] stop list not found: {STOP_PATH}; proceeding without stopwords")
    STOPS = set()

# ---------- helpers -----------------
tbl = str.maketrans(string.punctuation + string.digits,
                    " " * (len(string.punctuation) + 10))
NON_ASCII = re.compile(r"[^\x00-\x7F]+")



def tokenize(text: str):
    """ASCII clean → lower → remove punct/digit → split → stem + stop‑filter."""
    text = NON_ASCII.sub(" ", text)
    text = text.lower().replace("\n", " ")
    text = text.replace("’", " ").replace("'", " ")
    text = text.translate(tbl)

    return [w for w in text.split() if len(w) > 1 and w not in STOPS]



def add_whitelist_phrases(tokens):
    """Add whitelist phrases to tokens."""
    extra = []
    for i in range(len(tokens) - 1):
        key = (tokens[i], tokens[i + 1])
        if key in PHRASE_MAP:
            extra.append(PHRASE_MAP[key])
    return tokens + extra

def chunk_tokens(tokens):
    """Return intact doc if ≤WIN_LEN; else yield windows of WIN_LEN (last may be shorter)."""
    n = len(tokens)
    if n <= WIN_LEN:
        yield " ".join(tokens)
    else:
        for i in range(0, n, WIN_LEN):
            seg = tokens[i:i + WIN_LEN]
            if seg:  # ensure non‑empty
                yield " ".join(seg)


def write_chunks(tokens, year: int, src: str, stem_name: str):
    out_dir = Path(OUTPUT_ROOT, str(year))
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, seg in enumerate(chunk_tokens(tokens)):
        seg_tokens = seg.split()
        seg_tokens = add_whitelist_phrases(seg_tokens)
        out_dir.joinpath(f"{src}_{stem_name}_chunk{idx}.txt").write_text(" ".join(seg_tokens))

# -------- COHA (keep only MAG / NEWS) --------
KEEP_GENRE = {"MAG", "NEWS"}

def process_coha():
    for year_dir in Path(COHA_ROOT).iterdir():
        if not year_dir.name.isdigit() or int(year_dir.name) < YEAR_MIN:
            continue
        for f in year_dir.glob("*.txt"):
            parts = f.name.split("_")
            if len(parts) < 3 or parts[1] not in KEEP_GENRE:
                continue
            tokens = tokenize(f.read_text(encoding="latin1", errors="ignore"))
            if tokens:
                write_chunks(tokens, int(year_dir.name), "COHA", f.stem)

# -------- generic for HBR / ILR ------------
YEAR_RE = re.compile(r"(19|20)\d{2}")

def process_generic(root: str, src: str):
    for f in tqdm(Path(root).rglob("*.txt"), desc=src):
        m = YEAR_RE.search(str(f))
        if not m or int(m.group()) < YEAR_MIN:
            continue
        year = int(m.group())
        tokens = tokenize(f.read_text(errors="ignore"))
        if tokens:
            write_chunks(tokens, year, src, f.stem)

if __name__ == "__main__":
    print("[1/3] COHA …")
    process_coha()

    print("[2/3] HBR …")
    process_generic(HBR_ROOT, "HBR")

    print("[3/3] ILR …")
    process_generic(ILR_ROOT, "ILR")

    print("All chunked files are in", OUTPUT_ROOT)
