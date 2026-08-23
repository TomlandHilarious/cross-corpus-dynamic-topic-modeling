import pandas as pd
import numpy as np
import torch
from scipy.special import rel_entr          # KL helper
from difflib import SequenceMatcher
from detm import DETM                       # make sure detm is import-able

# ---------- helper: JSD (base-2, 0-1 range) ----------
EPS = 1e-12
def jsd(p, q):
    p = p / p.sum();  q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(rel_entr(p, m))
    kl_qm = np.sum(rel_entr(q, m))
    return 0.5 * (kl_pm + kl_qm) / np.log(2)   # convert to bits

# ---------- load CSV with titles ----------
all_df = pd.read_csv("label_results/topic_labels_all.csv")
global_titles = all_df[all_df.source=="GLOBAL"].set_index("topic")["title"]
src_df = all_df[all_df.source!="GLOBAL"]

# ---------- load DETM & average beta by time ----------
ckpt = "/media/volume/sdb/projects/detm/detm_weighted/topic_50_min_df_10_delta_{0.05}_time_20250520_162831/lora_rank_16/detm_merged_K_50_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_Lr_0.0_Bsz_500_RhoSize_300_L_3_minDF_10_trainEmbeddings_1_lora_r16.pt"
device = torch.device("cpu")
model: DETM = torch.load(ckpt, map_location=device).eval()

with torch.no_grad():
    beta_global = model.get_beta(model.mu_q_alpha).mean(dim=1).cpu().numpy() # (K,V)
    src_names = ["COHA","HBR","ILR"]
    beta_src  = {s: model.get_beta(model.mu_q_alpha, src_id=i).mean(dim=1).cpu().numpy()
                 for i,s in enumerate(src_names)}

# ---------- build table ----------
rows=[]
for k,g_title in global_titles.items():
    beta_g = beta_global[k]
    for src in src_names:
        s_row = src_df[(src_df.source==src)&(src_df.topic==k)].iloc[0]
        s_title = s_row["title"]
        sim = SequenceMatcher(None,g_title.lower(),s_title.lower()).ratio()
        if   sim>=0.9: cat="Identical"
        elif sim>=0.6: cat="Similar"
        elif sim>=0.3: cat="Moderate"
        else:          cat="Very-diff"

        js = jsd(beta_g+EPS, beta_src[src][k]+EPS)  # 0 identical → 1 max

        rows.append({"Topic":k,
                     "Global title":g_title,
                     "Source":src,
                     "Local title":s_title,
                     "TitleSim":round(sim,3),
                     "SimCategory":cat,
                     "JSDtoGlobal":round(js,4)})
summary_df = pd.DataFrame(rows)
summary_df.to_csv("label_results/post_analysis.csv", index=False)
print("Saved label_results/post_analysis.csv")
