#!/user/rl3403/.conda/envs/nlp_kogut/bin/python
import scipy.io 
import matplotlib.pyplot as plt 
import data 
import pickle 
import numpy as np 
import torch
from pathlib import Path

ckpt_path = f"{Path(__file__).resolve().parent.parent}/coha/topic_50_min_df_100/detm_coha_K_50_Htheta_400_Optim_adam_Clip_10.0_ThetaAct_relu_Lr_0.001_Bsz_500_RhoSize_300_L_3_minDF_100_trainEmbeddings_1_epoch_35"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(ckpt_path, map_location=device).to(device).eval()
with torch.no_grad():
    alpha = model.mu_q_alpha              # K × T × L   (learned mean)
    beta  = model.get_beta(alpha).cpu().numpy()      # K × T × V  (softmax probabiliyt)

    # get src specific beta
    src_names = ["COHA", "HBR", "ILR"]
    name2id   = {s: i for i, s in enumerate(src_names)}
    beta_src  = {
        s: model.get_beta(alpha, src_id=name2id[s])   # (K, T, V)
        for s in src_names
    }

print('beta:', beta.shape) 

with open(f'{Path(__file__).resolve().parent.parent}/coha/min_df_100/timestamps.pkl', 'rb') as f:
    timelist = pickle.load(f)
# print('timelist: ', timelist)
T = len(timelist)
ticks = [str(x) for x in timelist]
# print('ticks: ', ticks)

## get vocab
data_file = f'{Path(__file__).resolve().parent.parent}/coha/min_df_100'
vocab, train, valid, test = data.get_data(data_file, temporal=True)
vocab_size = len(vocab)

## plot topics 
num_words = 10
times = [0, 10, 40]
num_topics = 50
for k in range(num_topics):
    for t in times:
        gamma = beta[k, t, :]
        top_words = list(gamma.argsort()[-num_words+1:][::-1])
        topic_words = [vocab[a] for a in top_words]
        print('Topic {} .. Time: {} ===> {}'.format(k, t, topic_words)) 

# print('Topic Government...')
# num_words = 10
# for t in range(0, 199, 10):
#     gamma = beta[49, t, :]
#     top_words = list(gamma.argsort()[-num_words+1:][::-1])
#     topic_words = [vocab[a] for a in top_words]
#     print('Year: {} ===> {}'.format(timelist[t], topic_words)) 
# print("="*20)
# print('Topic Religion...')
# num_words = 10
# for t in range(0, 199, 10):
#     gamma = beta[48, t, :]
#     top_words = list(gamma.argsort()[-num_words+1:][::-1])
#     topic_words = [vocab[a] for a in top_words]
#     print('Year: {} ===> {}'.format(timelist[t], topic_words)) 

print("="*20)
print('Topic 5...')
num_words = 10
for t in range(0, 199, 10):
    gamma = beta[5, t, :]
    top_words = list(gamma.argsort()[-num_words+1:][::-1])
    topic_words = [vocab[a] for a in top_words]
    print('Year: {} ===> {}'.format(timelist[t], topic_words)) 


fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 9), dpi=80, facecolor='w', edgecolor='k')
ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes.flatten()
ticks = [str(x) for x in timelist]
plt.xticks(np.arange(T)[0::10], timelist[0::10])

# words_1 = ['vietnam', 'war', 'pakistan', 'indonesia']
# tokens_1 = [vocab.index(w) for w in words_1]
# betas_1 = [beta[1, :, x] for x in tokens_1]
# for i, comp in enumerate(betas_1):
#     ax1.plot(range(T), comp, label=words_1[i], lw=2, linestyle='--', marker='o', markersize=4)
# ax1.legend(frameon=False)
# print('np.arange(T)[0::10]: ', np.arange(T)[0::10])
# ax1.set_xticks(np.arange(T)[0::10])
# ax1.set_xticklabels(timelist[0::10])
# ax1.set_title('Topic "Southeast Asia"', fontsize=12)


words_5 = ['company', 'state', 'business', 'market', 'stock']
tokens_5 = [vocab.index(w) for w in words_5]
betas_5 = [beta[5, :, x] for x in tokens_5]
for i, comp in enumerate(betas_5):
    ax2.plot(comp, label=words_5[i], lw=2, linestyle='--', marker='o', markersize=4)
ax2.legend(frameon=False)
ax2.set_xticks(np.arange(T)[0::10])
ax2.set_xticklabels(timelist[0::10], rotation=45, ha='right')
ax2.set_title('Topic 5 (Business Development?)"', fontsize=12)


words_26 = ['work', 'social', 'research', 'students', 'knowledge']
tokens_26 = [vocab.index(w) for w in words_26]
betas_26 = [beta[26, :, x] for x in tokens_26]
for i, comp in enumerate(betas_26):
    ax3.plot(comp, label=words_26[i], lw=2, linestyle='--', marker='o', markersize=4)
ax3.legend(frameon=False)
ax3.set_xticks(np.arange(T)[0::10])
ax3.set_xticklabels(timelist[0::10],rotation=45, ha='right')
ax3.set_title('Topic Education?', fontsize=12)


words_46 = ['war', 'country', 'american', 'england', 'enemy']
tokens_46 = [vocab.index(w) for w in words_46]
betas_46 = [beta[46, :, x] for x in tokens_46]
for i, comp in enumerate(betas_46):
    ax4.plot(comp, label=words_46[i], lw=2, linestyle='--', marker='o', markersize=4)
ax4.legend(frameon=False)
ax4.set_xticks(np.arange(T)[0::10])
ax4.set_xticklabels(timelist[0::10], rotation=45, ha='right')
ax4.set_title('Topic war?"', fontsize=12)


words_28 = ['men', 'equality', 'gender', 'female', 'education']
words_28 = ['education', 'gender', 'equality']
tokens_28 = [vocab.index(w) for w in words_28]
betas_28 = [beta[28, :, x] for x in tokens_28]
for i, comp in enumerate(betas_28):
    ax5.plot(comp, label=words_28[i], lw=2, linestyle='--', marker='o', markersize=4)
ax5.legend(frameon=False)
ax5.set_xticks(np.arange(T)[0::10])
ax5.set_xticklabels(timelist[0::10], rotation=45, ha='right')
ax5.set_title('Topic "Human Rights"', fontsize=12)


words_13 = ['human', 'nature', 'earth', 'god']
tokens_13 = [vocab.index(w) for w in words_13]
betas_13 = [beta[13, :, x] for x in tokens_13]
for i, comp in enumerate(betas_13):
    ax6.plot(comp, label=words_13[i], lw=2, linestyle='--', marker='o', markersize=4)
ax6.legend(frameon=False)
ax6.set_xticks(np.arange(T)[0::10])
ax6.set_xticklabels(timelist[0::10], rotation=45, ha='right')
ax6.set_title('Topic "Nature & Aesthetics"', fontsize=12)


words_48 = ['god', 'form', 'nature', 'art', 'history']
tokens_48 = [vocab.index(w) for w in words_48]
betas_48 = [beta[48, :, x] for x in tokens_48]
for i, comp in enumerate(betas_48):
    ax7.plot(comp, label=words_48[i], lw=2, linestyle='--', marker='o', markersize=4)
ax7.legend(frameon=False)
ax7.set_xticks(np.arange(T)[0::10])
ax7.set_xticklabels(timelist[0::10], rotation=45, ha='right')
ax7.set_title('Topic "Climate Change"', fontsize=12)


words_49 = ['government', 'soviet', 'war', 'states', 'chinese']
tokens_49 = [vocab.index(w) for w in words_49]
betas_49 = [beta[49, :, x] for x in tokens_49]
for i, comp in enumerate(betas_49):
    ax8.plot(comp, label=words_49[i], lw=2, linestyle='--', marker='o', markersize=4)
ax8.legend(frameon=False)
ax8.set_title('Topic war and government', fontsize=12)
ax8.set_xticks(np.arange(T)[0::10])
ax8.set_xticklabels(timelist[0::10], rotation=45, ha='right')
plt.savefig(f'{Path(__file__).resolve().parent.parent}/coha/topic_50_min_df_100/word_evolution.png')
plt.show()
