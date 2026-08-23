"""This file defines a dynamic etm object.
"""

import torch
import torch.nn.functional as F 
import numpy as np 
import math 
import data

from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DETM(nn.Module):
    def __init__(self, args, embeddings):
        super(DETM, self).__init__()

        ## define hyperparameters
        self.num_topics = args.num_topics
        # adding the source dimension
        self.num_sources = args.num_sources

        self.num_times = args.num_times
        self.vocab_size = args.vocab_size
        self.t_hidden_size = args.t_hidden_size
        self.eta_hidden_size = args.eta_hidden_size
        self.rho_size = args.rho_size
        self.emsize = args.emb_size
        self.enc_drop = args.enc_drop
        self.eta_nlayers = args.eta_nlayers
        self.t_drop = nn.Dropout(args.enc_drop)
        self.delta = args.delta
        self.train_embeddings = args.train_embeddings

        self.theta_act = self.get_activation(args.theta_act)

        ## define the source specific parameter

        self.gamma_src = nn.Parameter(torch.randn(self.num_sources, self.num_topics))  # K‑dim bias per source
        
        ## Source-specific topic adaptation parameters
        self.source_adaptation_mode = getattr(args, 'source_adaptation_mode', False)
        if self.source_adaptation_mode:
            # delta_alpha: residual on alpha for each source
            # Shape: [num_sources, num_topics, num_times, rho_size]
            # Initialized to zeros so source-specific topics = shared topics initially
            self.delta_alpha = nn.Parameter(torch.zeros(self.num_sources, self.num_topics, self.num_times, self.rho_size))
        else:
            self.register_parameter('delta_alpha', None)

        ## define the word embedding matrix \rho
        if args.train_embeddings:
            self.rho = nn.Linear(args.rho_size, args.vocab_size, bias=False)
        else:
            num_embeddings, emsize = embeddings.size()
            rho = nn.Embedding(num_embeddings, emsize)
            rho.weight.data = embeddings
            self.rho = rho.weight.data.clone().float().to(device)
        ## LoRa Adaptor

        self.lora_rank = getattr(args, 'lora_rank', 0)
        self.lora_alpha = getattr(args, 'lora_alpha', 1.0)
        if self.lora_rank > 0:
            r, S = self.lora_rank, self.num_sources
            V, L = self.vocab_size, self.rho_size
            #  B_s : (V, r)   A_s : (r, L)   → B_s @ A_s  ≈ (V,L)
            self.lora_B = nn.Parameter(0.01 * torch.randn(S, V, r))
            self.lora_A = nn.Parameter(0.01 * torch.randn(S, r, L))
        else:
            self.register_parameter('lora_B', None)
            self.register_parameter('lora_A', None)



        ## define the variational parameters for the topic embeddings over time (alpha) ... alpha is K x T x L
        self.mu_q_alpha = nn.Parameter(torch.randn(args.num_topics, args.num_times, args.rho_size))
        self.logsigma_q_alpha = nn.Parameter(torch.randn(args.num_topics, args.num_times, args.rho_size))
    
        ## define variational distribution for \theta_{1:D} via amortizartion... theta is K x D
        self.q_theta = nn.Sequential(
                    nn.Linear(args.vocab_size+args.num_topics, args.t_hidden_size), 
                    self.theta_act,
                    nn.Linear(args.t_hidden_size, args.t_hidden_size),
                    self.theta_act,
                )
        self.mu_q_theta = nn.Linear(args.t_hidden_size, args.num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(args.t_hidden_size, args.num_topics, bias=True)

        ## define variational distribution for \eta via amortizartion... eta is K x T
        self.q_eta_map = nn.Linear(args.vocab_size, args.eta_hidden_size)
        self.q_eta = nn.LSTM(args.eta_hidden_size, args.eta_hidden_size, args.eta_nlayers, dropout=args.eta_dropout)
        self.mu_q_eta = nn.Linear(args.eta_hidden_size+args.num_topics, args.num_topics, bias=True)
        self.logsigma_q_eta = nn.Linear(args.eta_hidden_size+args.num_topics, args.num_topics, bias=True)

    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act 

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def get_kl(self, q_mu, q_logsigma, p_mu=None, p_logsigma=None):
        """Returns KL( N(q_mu, q_logsigma) || N(p_mu, p_logsigma) ).
        """
        if p_mu is not None and p_logsigma is not None:
            sigma_q_sq = torch.exp(q_logsigma)
            sigma_p_sq = torch.exp(p_logsigma)
            kl = ( sigma_q_sq + (q_mu - p_mu)**2 ) / ( sigma_p_sq + 1e-6 )
            kl = kl - 1 + p_logsigma - q_logsigma
            kl = 0.5 * torch.sum(kl, dim=-1)
        else:
            kl = -0.5 * torch.sum(1 + q_logsigma - q_mu.pow(2) - q_logsigma.exp(), dim=-1)
        return kl

    def get_alpha(self): ## mean field
        """
        Returns shared/global alpha (K, T, L)
        This is the backbone topic trajectory learned from merged training
        """
        # Use the device of model parameters instead of global device variable
        param_device = self.mu_q_alpha.device
        alphas = torch.zeros(self.num_topics, self.num_times, self.rho_size).to(param_device)
        kl_alpha = []

        alphas[:, 0, :] = self.reparameterize(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :])

        p_mu_0 = torch.zeros(self.num_topics, self.rho_size).to(param_device)
        logsigma_p_0 = torch.zeros(self.num_topics, self.rho_size).to(param_device)
        kl_0 = self.get_kl(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :], p_mu_0, logsigma_p_0)
        kl_alpha.append(kl_0)
        for t in range(1, self.num_times):
            alphas[:, t, :] = self.reparameterize(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :]) 
            p_mu_t = alphas[:, t-1, :]
            logsigma_p_t = torch.log(self.delta * torch.ones(self.num_topics, self.rho_size).to(param_device))
            kl_t = self.get_kl(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :], p_mu_t, logsigma_p_t)
            kl_alpha.append(kl_t)
        kl_alpha = torch.stack(kl_alpha).sum()
        return alphas, kl_alpha.sum()
    
    def get_alpha_source(self, src_id, alpha_global=None):
        """
        Returns source-specific alpha by adding residual delta_alpha
        alpha_source[s] = alpha_global + delta_alpha[s]
        
        Args:
            src_id: int, source ID in [0, num_sources-1]
            alpha_global: (K, T, L) tensor, if None will call get_alpha()[0]
        
        Returns:
            alpha_source: (K, T, L) tensor
        """
        if not self.source_adaptation_mode or self.delta_alpha is None:
            # Fallback to shared alpha
            if alpha_global is None:
                alpha_global, _ = self.get_alpha()
            return alpha_global
        
        if alpha_global is None:
            alpha_global, _ = self.get_alpha()
        
        # Add source-specific residual
        alpha_source = alpha_global + self.delta_alpha[src_id]  # (K, T, L)
        return alpha_source
    
    def get_delta_alpha_regularization(self, lambda_anchor=1e-3, lambda_smooth=1e-3):
        """
        Compute regularization losses on delta_alpha:
        - L_anchor: penalize large deviations from shared backbone
        - L_smooth: penalize temporal discontinuities
        
        Returns:
            loss_anchor: scalar tensor
            loss_smooth: scalar tensor
        """
        if not self.source_adaptation_mode or self.delta_alpha is None:
            return torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
        
        # Anchor regularization: ||delta_alpha||^2
        loss_anchor = lambda_anchor * torch.sum(self.delta_alpha ** 2)
        
        # Temporal smoothness: ||delta_alpha[:,:,t,:] - delta_alpha[:,:,t-1,:]||^2
        loss_smooth = torch.tensor(0.0).to(device)
        if self.num_times > 1:
            delta_diff = self.delta_alpha[:, :, 1:, :] - self.delta_alpha[:, :, :-1, :]
            loss_smooth = lambda_smooth * torch.sum(delta_diff ** 2)
        
        return loss_anchor, loss_smooth

    def get_eta(self, rnn_inp): ## structured amortized inference
        inp = self.q_eta_map(rnn_inp).unsqueeze(1)
        hidden = self.init_hidden()
        output, _ = self.q_eta(inp, hidden)
        output = output.squeeze(1)

        etas = torch.zeros(self.num_times, self.num_topics).to(device)
        kl_eta = []

        inp_0 = torch.cat([output[0], torch.zeros(self.num_topics,).to(device)], dim=0)
        mu_0 = self.mu_q_eta(inp_0)
        logsigma_0 = self.logsigma_q_eta(inp_0)
        etas[0] = self.reparameterize(mu_0, logsigma_0)

        p_mu_0 = torch.zeros(self.num_topics,).to(device)
        logsigma_p_0 = torch.zeros(self.num_topics,).to(device)
        kl_0 = self.get_kl(mu_0, logsigma_0, p_mu_0, logsigma_p_0)
        kl_eta.append(kl_0)
        for t in range(1, self.num_times):
            inp_t = torch.cat([output[t], etas[t-1]], dim=0)
            mu_t = self.mu_q_eta(inp_t)
            logsigma_t = self.logsigma_q_eta(inp_t)
            etas[t] = self.reparameterize(mu_t, logsigma_t)

            p_mu_t = etas[t-1]
            logsigma_p_t = torch.log(self.delta * torch.ones(self.num_topics,).to(device))
            kl_t = self.get_kl(mu_t, logsigma_t, p_mu_t, logsigma_p_t)
            kl_eta.append(kl_t)
        kl_eta = torch.stack(kl_eta).sum()
        return etas, kl_eta

    def get_theta(self, eta, bows, times, src_ids): ## amortized inference
        """Returns the topic proportions. Amortised inference of θ with **source bias**.
        src_ids : LongTensor (batch,) values in [0, S-1]
        """
        # 1. build prior mean  μ_p = η_t + γ_s
        times = times.to(device=eta.device, dtype=torch.long)
        eta_td = eta[times]
        gamma_td = self.gamma_src[src_ids]    # (B,K)

        prior_mu = eta_td + gamma_td  
        # 2. standard encoder network
        inp = torch.cat([bows, prior_mu], dim=1)
        q_theta = self.q_theta(inp)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        # 3. KL( q || N(prior_mu, I) )
        kl_theta = self.get_kl(mu_theta, logsigma_theta, prior_mu, torch.zeros(self.num_topics).to(device))
        return theta, kl_theta

    def get_beta(self, alpha, src_id=None):
        """Returns the topic matrix beta [K, T, V]
        
        Args:
            alpha: (K, T, L) topic trajectory tensor
            src_id: optional source ID for LoRA adaptation (legacy, not used in new adaptation)
        
        Returns:
            beta: (K, T, V) topic-word distributions
        """
        if isinstance(self.rho, nn.Linear):
            W_base = self.rho.weight # (V, L)
        else:                   # freeze rho
            W_base = self.rho # (V, L)
        # lora adapt (if on) - legacy mechanism, kept for backward compatibility
        if src_id is not None and self.lora_rank > 0:
            B = self.lora_B[src_id]
            A = self.lora_A[src_id]
            delta = self.lora_alpha * (B @ A)
            W_eff = W_base + delta
        else:
            W_eff = W_base
        
        tmp = alpha.view(-1, self.rho_size)  
        logit = tmp @ W_eff.t()   ## (K·T, V)

        logit = logit.view(alpha.size(0), alpha.size(1), -1)  # (K, T, V)
        beta = F.softmax(logit, dim=-1)
        return beta
    
    def get_beta_source(self, src_id, alpha_global=None):
        """Returns source-specific beta using source-specific alpha
        
        Args:
            src_id: int, source ID
            alpha_global: optional (K, T, L) shared alpha, if None will compute it
        
        Returns:
            beta_source: (K, T, V) source-specific topic-word distributions
        """
        alpha_source = self.get_alpha_source(src_id, alpha_global)
        beta_source = self.get_beta(alpha_source)
        return beta_source 

    def get_nll(self, theta, beta, bows):
        theta = theta.unsqueeze(1)
        loglik = torch.bmm(theta, beta).squeeze(1)
        loglik = loglik
        loglik = torch.log(loglik+1e-6)
        nll = -loglik * bows
        nll = nll.sum(-1)
        return nll  

    def forward(self, bows, normalized_bows, times, rnn_inp, num_docs, src_ids):
        bsz = normalized_bows.size(0)
        coeff = num_docs / bsz 
        alpha, kl_alpha = self.get_alpha()
        eta, kl_eta = self.get_eta(rnn_inp)
        theta, kl_theta = self.get_theta(eta, normalized_bows, times, src_ids)
        
        # Compute beta based on adaptation mode
        unique_src = src_ids.unique()                       # ≤ S
        
        if self.source_adaptation_mode and self.delta_alpha is not None:
            # Source-specific beta using delta_alpha residuals
            beta_pool  = {
                int(s): self.get_beta_source(int(s), alpha)    # (K, T, V)
                for s in unique_src
            }
        else:
            # Legacy: LoRA-adapted beta or shared beta
            beta_pool  = {
                int(s): self.get_beta(alpha, src_id=int(s))    # (K, T, V)
                for s in unique_src
            }
        
        beta_list = []
        for doc_src, doc_t in zip(src_ids, times):
            # beta_pool[src] : (K, T, V)
            s = int(doc_src.item())
            t = int(doc_t.item())
            beta_st = beta_pool[s][:, t, :] #(K, V)
            beta_list.append(beta_st)
        beta = torch.stack(beta_list, dim=0) 
        
        nll_vec = self.get_nll(theta, beta, bows)
        return nll_vec, kl_alpha, kl_eta, kl_theta

    def init_hidden(self):
        """Initializes the first hidden state of the RNN used as inference network for \eta.
        """
        weight = next(self.parameters())
        nlayers = self.eta_nlayers
        nhid = self.eta_hidden_size
        return (weight.new_zeros(nlayers, 1, nhid), weight.new_zeros(nlayers, 1, nhid))
    
    @torch.no_grad()
    def infer_all_theta(self,
                        tokens,  # list of np arrays, len = D
                        counts,  # ist of np arrays, len = D
                        times,   # np array shape (D,)
                        rnn_inp, 
                        src_ids=None,      # train_rnn_inp
                        batch_size=1000,
                        bow_norm=True,
                        length_weight=True
                        ):
        """return theta_mat : torch.Tensor, shape = (D, self.num_topics)
        run self.get_eta(rnn_inp)
        Usage: theta_mat, doc_len = model.infer_all_theta(
                                   train_tokens, train_counts,
                                   train_times, train_rnn_inp)
        """
        device = next(self.parameters()).device
        D = len(tokens)
        K = self.num_topics
        # 1) eta_t is of shape (T, K)
        eta, _ = self.get_eta(rnn_inp.to(device))

        theta_chunks = []
        lengths = []
        # 2) iterate through each batch 
        # and return theta_mat: shape = (D, self.num_topics)
        for idx_batch in torch.split(torch.arange(D), batch_size):
            bows, times_b = data.get_batch(
                tokens, counts, idx_batch,
                self.vocab_size, self.emsize,
                temporal=True, times=times)
            bows = bows.to(device)
            sums = bows.sum(1).unsqueeze(1)    # (B,1)

            if bow_norm:
                bows_in = bows / sums
            else:
                bows_in = bows
            if src_ids is None:
                src_b = None                       #  get_theta will fall back to 0-bias
            else:
                src_b = src_ids[idx_batch].to(device)

            theta_b, _ = self.get_theta(eta, bows_in, times_b.to(device), src_b) 
            theta_chunks.append(theta_b.cpu())  # (B,K)
            lengths.append(sums.squeeze(1).cpu()) # (B,)
        theta_mat = torch.cat(theta_chunks, dim=0)  # (D,K)
        doc_len = torch.cat(lengths, dim=0)   # (D,)

        if length_weight:
            return theta_mat, doc_len
        else:
            return theta_mat

            
            



