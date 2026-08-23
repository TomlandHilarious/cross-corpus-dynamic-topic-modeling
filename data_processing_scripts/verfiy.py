#!/user/rl3403/.conda/envs/nlp_kogut/bin/python
ROOT = "Merged_1920plus"

"""
count_sources_per_year.py

Walk through ROOT=<Merged_1920plus>/<year>/, count how many files start with
COHA_, HBR_, ILR_ in each year sub-folder, and report years lacking a source.
"""

import os
from collections import defaultdict

ROOT = "Merged_1920plus"          # adjust if needed
SRC_TAGS = ("COHA", "HBR", "ILR")  # file name prefixes

# year -> {src: count}
year_counts = defaultdict(lambda: {s: 0 for s in SRC_TAGS})

for year in os.listdir(ROOT):
    if not year.isdigit():
        continue                              # skip stray files
    subdir = os.path.join(ROOT, year)
    if not os.path.isdir(subdir):
        continue
    for fname in os.listdir(subdir):
        for tag in SRC_TAGS:
            if fname.startswith(f"{tag}_"):
                year_counts[int(year)][tag] += 1
                break                        # avoid double-count

# --- print summary ---
print(f"{'Year':<6}  COHA  HBR  ILR")
print("-" * 22)
missing = []
for y in sorted(year_counts):
    c = year_counts[y]
    print(f"{y:<6}  {c['COHA']:<4}  {c['HBR']:<3}  {c['ILR']:<3}")
    if any(c[src] == 0 for src in SRC_TAGS):
        missing.append(y)

if not missing:
    print("\nEvery year has at least one file from COHA, HBR and ILR.")
else:
    print("\n Years missing a source:", sorted(missing))