#!/user/rl3403/.conda/envs/nlp_kogut/bin/python 
"""
Evaluate all retrained DETM models and generate metrics table.
Computes TD (Topic Diversity), TC (Topic Coherence), TQ (Topic Quality = TD * TC)
for all individual and merged models.
"""
from pathlib import Path

import os
import sys
import argparse
import pickle
import numpy as np
import scipy.io
import torch
from collections import OrderedDict

# Import from main.py
from detm import DETM
from utils import get_topic_coherence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _diversity_helper(beta, num_tops):
    """
    Compute topic diversity: fraction of unique words in top-N across all topics.
    beta: (K, V) topic-word distributions
    """
    K = beta.shape[0]
    list_w = []
    for k in range(K):
        gamma = beta[k, :]
        top_words = gamma.argsort()[-num_tops:][::-1]
        list_w.extend(top_words.tolist())
    
    n_unique = len(set(list_w))
    diversity = n_unique / (K * num_tops)
    return float(diversity)


def evaluate_checkpoint(checkpoint_path, data_path, dataset_name):
    """
    Load checkpoint and compute TD, TC, TQ metrics.
    
    Returns:
        dict: {'TD': float, 'TC': float, 'TQ': float}
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {dataset_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data: {data_path}")
    print(f"{'='*80}")
    
    # Load vocab
    vocab_file = os.path.join(data_path, 'vocab.pkl')
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    
    # Load training data tokens for coherence computation
    train_tokens_file = os.path.join(data_path, 'bow_tr_tokens.mat')
    train_tokens_data = scipy.io.loadmat(train_tokens_file)
    train_tokens = [doc.squeeze().tolist() for doc in train_tokens_data['tokens'][0]]
    
    # Load training timestamps
    train_times_file = os.path.join(data_path, 'bow_tr_timestamps.mat')
    train_times_data = scipy.io.loadmat(train_times_file)
    train_times = train_times_data['timestamps'].squeeze().tolist()
    train_times_np = np.asarray(train_times)
    
    num_times = len(set(train_times))
    
    print(f"Vocab size: {vocab_size}")
    print(f"Num documents: {len(train_tokens)}")
    print(f"Num time slices: {num_times}")
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle two cases: checkpoint is dict with state_dict, or checkpoint is model object
    if isinstance(checkpoint, torch.nn.Module):
        # Checkpoint is the model object itself
        model = checkpoint.to(device)
        model.eval()
        print(f"Model config: K={model.num_topics}, T={model.num_times}, V={model.vocab_size}")
    else:
        # Checkpoint is a dictionary
        args_dict = checkpoint.get('args', {})
        num_topics = args_dict.get('num_topics', 20)
        rho_size = args_dict.get('rho_size', 300)
        emb_size = args_dict.get('emb_size', 300)
        t_hidden_size = args_dict.get('t_hidden_size', 800)
        theta_act = args_dict.get('theta_act', 'relu')
        
        print(f"Model config: K={num_topics}, T={num_times}, V={vocab_size}")
        
        # Create model
        class Args:
            pass
        args = Args()
        args.num_topics = num_topics
        args.num_times = num_times
        args.vocab_size = vocab_size
        args.rho_size = rho_size
        args.emb_size = emb_size
        args.t_hidden_size = t_hidden_size
        args.theta_act = theta_act
        args.train_embeddings = 1
        args.enc_drop = 0.0
        args.num_sources = 1
        args.source_adaptation_mode = 0
        args.lora_rank = 0
        
        model = DETM(args)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()
    
    print("Computing metrics...")
    
    with torch.no_grad():
        # Get beta: (K, T, V)
        alpha = model.mu_q_alpha
        beta = model.get_beta(alpha)
        
        # ---- Topic Diversity ----
        num_tops_div = 25
        TD_all = []
        
        for tt in range(num_times):
            beta_t = beta[:, tt, :]  # (K, V)
            td_t = _diversity_helper(beta_t.cpu().numpy(), num_tops_div)
            TD_all.append(td_t)
        
        TD = float(np.mean(TD_all))
        
        # ---- Topic Coherence ----
        num_tops_coh = 10
        TC_all = []
        
        for tt in range(num_times):
            beta_t = beta[:, tt, :]  # (K, V)
            
            # Only use docs from this time slice
            doc_idx_t = np.where(train_times_np == tt)[0]
            if len(doc_idx_t) == 0:
                continue
            docs_t = [train_tokens[i] for i in doc_idx_t]
            
            tc_t, _ = get_topic_coherence(
                beta_t.cpu().numpy(),
                docs_t,
                vocab=vocab,
                top_n=num_tops_coh
            )
            TC_all.append(tc_t)
        
        TC = float(np.nanmean(TC_all))
        
        # ---- Topic Quality ----
        TQ = TD * TC
    
    print(f"\nResults:")
    print(f"  TD (Topic Diversity): {TD:.6f}")
    print(f"  TC (Topic Coherence): {TC:.6f}")
    print(f"  TQ (Topic Quality):   {TQ:.6f}")
    
    return {'TD': TD, 'TC': TC, 'TQ': TQ}


def main():
    # Define all models to evaluate
    models = OrderedDict([
        # Individual corpus-specific vocab
        ('COHA_specific', {
            'checkpoint': f'{Path(__file__).resolve().parent.parent}/detm_individual_specific_vocab_5year_v3/coha_topic20_min_df100_delta0.01_20260505_220641/detm_coha_K_20_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_Lr_5e-05_Bsz_500_RhoSize_300_L_3_minDF_100_trainEmbeddings_1_pretrain.pt',
            'data': f'{Path(__file__).resolve().parent.parent}/data_processing_scripts/individual_corpora_specific_vocab_5year_v3/coha/min_df_100',
            'label': 'COHA - Corpus-specific'
        }),
        ('HBR_specific', {
            'checkpoint': f'{Path(__file__).resolve().parent.parent}/detm_individual_specific_vocab_5year_v3/hbr_topic20_min_df100_delta0.01_20260505_220658/detm_hbr_K_20_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_Lr_5e-05_Bsz_500_RhoSize_300_L_3_minDF_100_trainEmbeddings_1_pretrain.pt',
            'data': f'{Path(__file__).resolve().parent.parent}/data_processing_scripts/individual_corpora_specific_vocab_5year_v3/hbr/min_df_100',
            'label': 'HBR - Corpus-specific'
        }),
        ('ILR_specific', {
            'checkpoint': f'{Path(__file__).resolve().parent.parent}/detm_individual_specific_vocab_5year_v3/ilr_topic20_min_df100_delta0.01_20260505_220712/detm_ilr_K_20_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_Lr_5e-05_Bsz_500_RhoSize_300_L_3_minDF_100_trainEmbeddings_1_pretrain.pt',
            'data': f'{Path(__file__).resolve().parent.parent}/data_processing_scripts/individual_corpora_specific_vocab_5year_v3/ilr/min_df_100',
            'label': 'ILR - Corpus-specific'
        }),
        # Individual merged vocab
        ('COHA_merged', {
            'checkpoint': f'{Path(__file__).resolve().parent.parent}/detm_individual_merged_5year_v3/coha_topic20_min_df100_delta0.01_20260505_220506/detm_coha_K_20_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_Lr_5e-05_Bsz_500_RhoSize_300_L_3_minDF_100_trainEmbeddings_1_pretrain.pt',
            'data': f'{Path(__file__).resolve().parent.parent}/data_processing_scripts/individual_corpora_min100_5year_v3/coha/min_df_100',
            'label': 'COHA - Merged vocab'
        }),
        ('HBR_merged', {
            'checkpoint': f'{Path(__file__).resolve().parent.parent}/detm_individual_merged_5year_v3/hbr_topic20_min_df100_delta0.01_20260505_220538/detm_hbr_K_20_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_Lr_5e-05_Bsz_500_RhoSize_300_L_3_minDF_100_trainEmbeddings_1_pretrain.pt',
            'data': f'{Path(__file__).resolve().parent.parent}/data_processing_scripts/individual_corpora_min100_5year_v3/hbr/min_df_100',
            'label': 'HBR - Merged vocab'
        }),
        ('ILR_merged', {
            'checkpoint': f'{Path(__file__).resolve().parent.parent}/detm_individual_merged_5year_v3/ilr_topic20_min_df100_delta0.01_20260505_220605/detm_ilr_K_20_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_Lr_5e-05_Bsz_500_RhoSize_300_L_3_minDF_100_trainEmbeddings_1_pretrain.pt',
            'data': f'{Path(__file__).resolve().parent.parent}/data_processing_scripts/individual_corpora_min100_5year_v3/ilr/min_df_100',
            'label': 'ILR - Merged vocab'
        }),
        # Merged model (all corpora)
        ('Merged_all', {
            'checkpoint': f'{Path(__file__).resolve().parent.parent}/detm_merged_5year/merged_topic20_min_df100_delta0.01_20260325_001151/detm_merged_K_20_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_Lr_5e-05_Bsz_500_RhoSize_300_L_3_minDF_100_trainEmbeddings_1_pretrain.pt',
            'data': f'{Path(__file__).resolve().parent.parent}/data_processing_scripts/merged_v2_min100_5year_v2/min_df_100',
            'label': 'Merged (all corpora)'
        }),
    ])
    
    # Evaluate all models
    results = OrderedDict()
    for model_name, config in models.items():
        try:
            metrics = evaluate_checkpoint(
                config['checkpoint'],
                config['data'],
                config['label']
            )
            results[model_name] = {
                'label': config['label'],
                'metrics': metrics
            }
        except Exception as e:
            print(f"\nERROR evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {
                'label': config['label'],
                'metrics': {'TD': None, 'TC': None, 'TQ': None}
            }
    
    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'Model':<30} {'Vocabulary':<20} {'TD':>10} {'TC':>10} {'TQ':>10}")
    print("-"*100)
    
    for model_name, data in results.items():
        label_parts = data['label'].split(' - ')
        corpus = label_parts[0]
        vocab_type = label_parts[1] if len(label_parts) > 1 else 'Merged vocab'
        
        metrics = data['metrics']
        td = f"{metrics['TD']:.3f}" if metrics['TD'] is not None else "N/A"
        tc = f"{metrics['TC']:.3f}" if metrics['TC'] is not None else "N/A"
        tq = f"{metrics['TQ']:.3f}" if metrics['TQ'] is not None else "N/A"
        
        print(f"{corpus:<30} {vocab_type:<20} {td:>10} {tc:>10} {tq:>10}")
    
    # Generate LaTeX table
    print("\n" + "="*100)
    print("LATEX TABLE")
    print("="*100)
    print("\\begin{table}[htbp]")
    print("\\caption{Topic Model Evaluation Results (Updated with New Vocabulary)}")
    print("\\label{tab:detm_updated_results}")
    print("\\centering")
    print("\\footnotesize")
    print("\\resizebox{\\linewidth}{!}{%")
    print("\\begin{tabular}{llccc}")
    print("\\toprule")
    print("Model & Vocabulary & TD & TC$\\uparrow$ & TQ$\\uparrow$ \\\\")
    print("\\midrule")
    
    # Order: HBR, COHA, ILR, then Merged
    ordered_keys = ['HBR_specific', 'HBR_merged', 'COHA_specific', 'COHA_merged', 
                    'ILR_specific', 'ILR_merged', 'Merged_all']
    
    for model_name in ordered_keys:
        if model_name not in results:
            continue
        data = results[model_name]
        label_parts = data['label'].split(' - ')
        corpus = label_parts[0]
        vocab_type = label_parts[1] if len(label_parts) > 1 else 'Merged vocab'
        
        metrics = data['metrics']
        td = f"{metrics['TD']:.3f}" if metrics['TD'] is not None else "N/A"
        tc = f"{metrics['TC']:.3f}" if metrics['TC'] is not None else "N/A"
        tq = f"{metrics['TQ']:.3f}" if metrics['TQ'] is not None else "N/A"
        
        print(f"{corpus} & {vocab_type} & {td} & {tc} & {tq} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}%")
    print("}")
    print("\\end{table}")
    print("="*100)
    
    # Save results to file
    output_file = f'{Path(__file__).resolve().parent.parent}/DETM_weights/results/metrics/evaluation_results.txt'
    with open(output_file, 'w') as f:
        f.write("DETM Evaluation Results\n")
        f.write("="*100 + "\n\n")
        for model_name, data in results.items():
            f.write(f"{data['label']}:\n")
            metrics = data['metrics']
            f.write(f"  TD: {metrics['TD']:.6f}\n" if metrics['TD'] is not None else "  TD: N/A\n")
            f.write(f"  TC: {metrics['TC']:.6f}\n" if metrics['TC'] is not None else "  TC: N/A\n")
            f.write(f"  TQ: {metrics['TQ']:.6f}\n\n" if metrics['TQ'] is not None else "  TQ: N/A\n\n")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
