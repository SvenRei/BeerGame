import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
import numpy as np
from .comm_utils import get_vocab_tensor
import torch.nn.functional as F

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.local_obs, self.global_states, self.hidden_states = [], [], []
        self.comm_in, self.actions, self.log_probs = [], [], []
        self.comm_actions = [] 
        self.rewards, self.is_terminals = [], []

    def push(self, obs, g_state, hidden, comm_in, action, log_prob, comm_action, reward, terminal):
        self.local_obs.append(obs)           # Expected [4, obs_dim]
        self.global_states.append(g_state)   # Expected [1, global_dim]
        self.hidden_states.append(hidden)    # Expected [4, hidden_dim]
        self.comm_in.append(comm_in)         # Expected [4, 1]
        self.actions.append(action)          # Expected [4, 1]
        self.log_probs.append(log_prob)      # Expected [4, 1]
        self.comm_actions.append(comm_action)# Expected [4, 1]
        self.rewards.append(reward)          # Expected [4, 1]
        self.is_terminals.append(terminal)   # Expected [4, 1]

class MAPPOActor(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=False)
        self.action_mean = nn.Linear(hidden_dim, 1)
        self.action_log_std = nn.Parameter(torch.full((1, 1), -1.0)) 
        
        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        nn.init.constant_(self.action_mean.bias, -0.75)

    def forward(self, obs, hidden):
        # ENGINEER FIX 1: Scale raw observations
        scaled_obs = obs / 100.0
        
        x = self.fc(scaled_obs).unsqueeze(0)
        hidden = hidden.unsqueeze(0)
        x, hidden = self.gru(x, hidden)
        x = x.squeeze(0)
        hidden = hidden.squeeze(0)
        
        mean = torch.sigmoid(self.action_mean(x))
        
        # ENGINEER FIX 2: Clamp log_std to prevent Variance Collapse (NaNs)
        clamped_log_std = torch.clamp(self.action_log_std, min=-3.0, max=0.5)
        std = clamped_log_std.exp().expand_as(mean)
        
        return Normal(mean, std), hidden

    def evaluate_actions(self, obs_seq, hidden_0, action_seq):
        """
        ENGINEER FIX: Recurrent Unrolling for BPTT.
        Evaluates an entire sequence (T, Num_Agents, Dim) to maintain gradient memory.
        """
        # ENGINEER FIX 1 (BPTT Path): Scale raw observations
        scaled_obs_seq = obs_seq / 100.0
        
        x = self.fc(scaled_obs_seq)
        x, _ = self.gru(x, hidden_0) # Gradients flow through time T here
        
        mean = torch.sigmoid(self.action_mean(x))
        
        # ENGINEER FIX 2 (BPTT Path): Clamp log_std to prevent Variance Collapse (NaNs)
        clamped_log_std = torch.clamp(self.action_log_std, min=-3.0, max=0.5)
        std = clamped_log_std.exp().expand_as(mean)
        
        dist = Normal(mean, std)
        action_log_probs = dist.log_prob(action_seq)
        
        return action_log_probs, dist.entropy().mean()

class CommMAPPOActor(nn.Module):
    def __init__(self, obs_dim, hidden_dim, vocab_size): 
        super().__init__()
        
        self.vocab_size = vocab_size
        comm_dim = 1 
        
        self.fc = nn.Sequential(
            nn.Linear(obs_dim + comm_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=False)
        self.action_mean = nn.Linear(hidden_dim, 1)
        
        # We still use a parameter, but we will clamp it in the forward pass
        self.action_log_std = nn.Parameter(torch.full((1, 1), -1.0)) 
        
        self.comm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size) 
        )

        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        nn.init.constant_(self.action_mean.bias, -0.75)

    def forward(self, obs, comm_in, hidden, tau=1.0):
        # ENGINEER FIX 1: Scale raw observations (e.g. inventory 2000 -> 20.0)
        # This prevents the GRU from saturating and exploding gradients.
        scaled_obs = obs / 100.0 
        
        x = torch.cat([scaled_obs, comm_in], dim=-1)
        
        x = self.fc(x).unsqueeze(0)
        hidden = hidden.unsqueeze(0)
        x, hidden = self.gru(x, hidden)
        
        x_flat = x.squeeze(0)
        hidden = hidden.squeeze(0)
        
        mean = torch.sigmoid(self.action_mean(x_flat))
        
        # ENGINEER FIX 2: Clamp the log_std. 
        # min=-3.0 prevents std from reaching 0. max=0.5 prevents infinite exploration.
        clamped_log_std = torch.clamp(self.action_log_std, min=-3.0, max=0.5)
        std = clamped_log_std.exp().expand_as(mean)
        
        msg_logits = self.comm_head(x_flat)
        dist_comm = Categorical(logits=msg_logits)
        msg_probs = F.gumbel_softmax(msg_logits, tau=tau, hard=True)
        
        return Normal(mean, std), dist_comm, msg_probs, hidden

    def evaluate_actions(self, obs_seq, comm_in_seq, hidden_0, action_seq, comm_action_seq):
        # ENGINEER FIX 1 (BPTT Path): Scale raw observations
        scaled_obs_seq = obs_seq / 100.0
        
        x = torch.cat([scaled_obs_seq, comm_in_seq], dim=-1)
        
        x = self.fc(x)
        x, _ = self.gru(x, hidden_0) 
        
        mean = torch.sigmoid(self.action_mean(x))
        
        # ENGINEER FIX 2 (BPTT Path): Clamp the log_std
        clamped_log_std = torch.clamp(self.action_log_std, min=-3.0, max=0.5)
        std = clamped_log_std.exp().expand_as(mean)
        dist_action = Normal(mean, std)
        
        msg_logits = self.comm_head(x)
        dist_comm = Categorical(logits=msg_logits)
        
        action_log_probs = dist_action.log_prob(action_seq)
        comm_log_probs = dist_comm.log_prob(comm_action_seq.squeeze(-1))
        
        entropy = dist_action.entropy().mean() + dist_comm.entropy().mean()
        
        return action_log_probs, comm_log_probs, dist_comm.probs, entropy

class MAPPOCommMAC(nn.Module):
    def __init__(self, actor_network, vocab_size=3, num_agents=4):
        super().__init__()
        self.actor = actor_network
        self.num_agents = num_agents
        self.vocab_size = vocab_size
        
        self.register_buffer("adj_mask", torch.tensor([
            [0.0, 1.0, 0.0, 0.0], 
            [1.0, 0.0, 1.0, 0.0], 
            [0.0, 1.0, 0.0, 1.0], 
            [0.0, 0.0, 1.0, 0.0]  
        ]))
        self.msg_buffer = None

    def init_buffer(self, batch_size, device):
        self.msg_buffer = torch.zeros(batch_size, self.num_agents, 1, device=device)

    def forward(self, obs, hiddens, tau=1.0, test_mode=False):
        B = obs.size(0)
        vocab_tensor = get_vocab_tensor(self.vocab_size, obs.device)
        
        masked_msgs = torch.matmul(self.adj_mask, self.msg_buffer)
        
        obs_flat = obs.view(B * self.num_agents, -1)
        msg_flat = masked_msgs.view(B * self.num_agents, -1)
        hiddens_flat = hiddens.view(B * self.num_agents, -1)
        
        dist_action, dist_comm, msg_probs, next_hiddens = self.actor(obs_flat, msg_flat, hiddens_flat, tau=tau)
        vocab_tensor = get_vocab_tensor(self.vocab_size, obs.device)
        comm_out = (msg_probs * vocab_tensor).sum(dim=-1, keepdim=True)
        
        if test_mode:
            comm_actions = dist_comm.probs.argmax(dim=-1)
        else:
            comm_actions = dist_comm.sample()
            
        comm_out = vocab_tensor[comm_actions].unsqueeze(-1) 
        self.msg_buffer = comm_out.view(B, self.num_agents, 1).clone()
        safe_logs = self.msg_buffer.detach().cpu().numpy()
        
        return dist_action, dist_comm, comm_actions, next_hiddens.view(B, self.num_agents, -1), masked_msgs.view(B, self.num_agents, 1), safe_logs

class MAPPOCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        # ENGINEER FIX 3: Scale the global state for the Critic.
        # If the Actor saturates on unscaled inputs, the Critic's MSE loss will explode too.
        scaled_state = state / 100.0
        return self.fc(scaled_state)

class MAPPOTrainer:
    def __init__(self, actor, critic, cfg, total_episodes, device, algo):
        self.actor, self.critic, self.device, self.algo = actor, critic, device, algo
        self.gamma = cfg.get("gamma", 0.99)
        self.entropy_coef = cfg.get("entropy_coef", 0.05)
        self.comm_penalty_coef = cfg.get("comm_penalty_coef", 0.0001) 
        self.k_epochs = cfg.get("k_epochs", 4)
        
        self.warm_up_episodes = cfg.get("warm_up_episodes", 1000)
        self.total_episodes = total_episodes 
        
        # ENGINEER FIX: Annealing schedule parameters
        self.tau_start = 1.0
        self.tau_min = 0.1
        self.tau_decay_episodes = cfg.get("tau_decay_episodes", total_episodes * 0.5)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.get("lr_actor", 3e-4))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.get("lr_critic", 1e-3))
        self.max_grad_norm = 0.2 

    def get_current_tau(self, current_ep):
        """Calculates the annealed Gumbel-Softmax temperature."""
        if current_ep >= self.tau_decay_episodes:
            return self.tau_min
        return self.tau_start - (self.tau_start - self.tau_min) * (current_ep / self.tau_decay_episodes)

    def update(self, buffer, current_ep):
        num_agents = 4
        
        # 1. STACK & DEFINE
        obs_seq = torch.stack(buffer.local_obs).detach()
        T = obs_seq.size(0)
        
        actions_seq = torch.stack(buffer.actions).detach()
        old_log_probs_seq = torch.stack(buffer.log_probs).detach()
        rewards_seq = torch.stack(buffer.rewards).detach()
        terminals_seq = torch.stack(buffer.is_terminals).detach()
        
        # FIX: Add .unsqueeze(0) here to create the [num_layers, batch, hidden] shape
        hidden_0 = torch.stack(buffer.hidden_states)[0].unsqueeze(0).detach()
        
        # Global states logic (Unified)
        g_states_raw = torch.stack(buffer.global_states).detach() # [T, 1, global_dim]
        g_states_seq = g_states_raw.repeat(1, num_agents, 1)      # [T, 4, global_dim]

        # 2. VECTORIZED RETURNS CALCULATION
        returns = torch.zeros_like(rewards_seq)
        next_return = 0.0
        for t in reversed(range(T)):
            next_return = rewards_seq[t] + self.gamma * next_return * (1.0 - terminals_seq[t])
            returns[t] = next_return
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 3. PRE-CALCULATE ADVANTAGES
        if self.algo == "ippo":
            values_flat = self.critic(obs_seq.reshape(T * num_agents, -1))
        else:
            values_flat = self.critic(g_states_seq.reshape(T * num_agents, -1))
        
        values_seq = values_flat.view(T, num_agents, 1)
        advantages_seq = returns - values_seq.detach()

        # 4. EPOCH LOOP (Uses pre-calculated sequences, no re-stacking)
        for _ in range(self.k_epochs):
            if self.algo == "comm_mappo":
                comm_in_seq = torch.stack(buffer.comm_in).detach()
                comm_actions_seq = torch.stack(buffer.comm_actions).detach()
                
                # Unroll comm-mappo
                act_log_probs, comm_log_probs, comm_dist_probs, entropy = self.actor.actor.evaluate_actions(
                    obs_seq, comm_in_seq, hidden_0, actions_seq, comm_actions_seq
                )
                current_log_probs_seq = act_log_probs + comm_log_probs.unsqueeze(-1)
                comm_penalty = 0.0 # (Or your comm penalty logic)
            else:
                # Basic unrolling
                act_log_probs, entropy = self.actor.evaluate_actions(obs_seq, hidden_0, actions_seq)
                current_log_probs_seq = act_log_probs
                comm_penalty = 0.0

            ratios = torch.exp(current_log_probs_seq - old_log_probs_seq)
            surr1 = ratios * advantages_seq
            surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages_seq
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            # Critic Loss
            if self.algo == "ippo":
                values_flat = self.critic(obs_seq.reshape(T * num_agents, -1))
            else:
                values_flat = self.critic(g_states_seq.reshape(T * num_agents, -1))
            
            critic_loss = torch.nn.MSELoss()(values_flat, returns.reshape(T * num_agents, 1))

            # Optimizer steps
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
        
        buffer.clear()
        return actor_loss.item(), critic_loss.item()