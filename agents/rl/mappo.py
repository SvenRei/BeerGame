import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
import numpy as np

# --- 1. ROLLOUT BUFFER ---
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.local_obs, self.global_states, self.hidden_states = [], [], []
        self.comm_in, self.actions, self.log_probs = [], [], []
        self.comm_actions = [] # --- FIX: Store communication actions ---
        self.rewards, self.is_terminals = [], []

# --- 2. MAPPO ACTOR (Standard) ---
class MAPPOActor(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=False)
        self.action_mean = nn.Linear(hidden_dim, 1)
        
        self.action_log_std = nn.Parameter(torch.full((1, 1), -1.0)) 
        
        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        nn.init.constant_(self.action_mean.bias, -0.75)

    def forward(self, obs, hidden):
        x = self.fc(obs).unsqueeze(0)
        x, hidden = self.gru(x, hidden)
        x = x.squeeze(0)
        
        mean = torch.sigmoid(self.action_mean(x))
        std = self.action_log_std.exp().expand_as(mean)
        return Normal(mean, std), hidden

# --- 3. COMM-MAPPO ACTOR (Decentralized Communication) ---
class CommMAPPOActor(nn.Module):
    def __init__(self, obs_dim, hidden_dim, comm_dim=1):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(obs_dim + comm_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=False)
        self.action_mean = nn.Linear(hidden_dim, 1)
        self.action_log_std = nn.Parameter(torch.full((1, 1), -1.0)) 
        
        # --- FIX: Outputs 3 logits for Categorical [Decrease, Silence, Increase] ---
        self.comm_head = nn.Linear(hidden_dim, 3)

        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        nn.init.constant_(self.action_mean.bias, -0.75)

    def forward(self, obs, comm_in, hidden):
        x = torch.cat([obs, comm_in], dim=-1).unsqueeze(0)
        x = self.fc(x)
        x, hidden = self.gru(x, hidden)
        x_flat = x.squeeze(0)
        
        mean = torch.sigmoid(self.action_mean(x_flat))
        std = self.action_log_std.exp().expand_as(mean)
        
        # --- FIX: Return a Categorical Distribution for the RL update ---
        logits = self.comm_head(x_flat)
        dist_comm = Categorical(logits=logits)
        
        return Normal(mean, std), dist_comm, hidden

# --- 4. MAPPO CRITIC (Centralized) ---
class MAPPOCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, state):
        return self.fc(state)

# --- 5. MAPPO TRAINER ---
class MAPPOTrainer:
    # --- FIX: Added total_episodes to signature ---
    def __init__(self, actor, critic, cfg, total_episodes, device, algo):
        self.actor, self.critic, self.device, self.algo = actor, critic, device, algo
        self.gamma = cfg.get("gamma", 0.99)
        self.entropy_coef = cfg.get("entropy_coef", 0.05)
        
        # --- FIX: Static Penalty Fallback ---
        self.comm_penalty_coef = cfg.get("comm_penalty_coef", 0.0001) 
        
        self.warm_up_episodes = cfg.get("warm_up_episodes", 1000)
        self.total_episodes = total_episodes 
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.get("lr_actor", 3e-4))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.get("lr_critic", 1e-3))
        self.max_grad_norm = 0.2 

    def update(self, buffer, current_ep):
        obs = torch.cat(buffer.local_obs).detach()
        hiddens = torch.cat(buffer.hidden_states).detach().transpose(0, 1)
        actions = torch.cat(buffer.actions).detach()
        old_log_probs = torch.cat(buffer.log_probs).detach()
        
        num_agents = 4
        returns = [0.0] * len(buffer.rewards)
        for i in reversed(range(len(buffer.rewards))):
            r = buffer.rewards[i]
            term = buffer.is_terminals[i]
            next_discounted = returns[i + num_agents] if i + num_agents < len(buffer.rewards) else 0.0
            returns[i] = r + (self.gamma * next_discounted * (1 - term))
            
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device).unsqueeze(1)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # --- FIX: Removed dynamic_penalty_coef and progress block ---
        
        for _ in range(4):
            if self.algo == "ippo":
                values = self.critic(obs)
            else:
                values = self.critic(torch.cat(buffer.global_states).detach())
            
            advantages = returns - values.detach()
            
            # --- FIX: Evaluate the dual-action space for COMM_MAPPO ---
            if self.algo == "comm_mappo":
                dist, dist_comm, _ = self.actor(obs, torch.cat(buffer.comm_in).detach(), hiddens)
                comm_actions = torch.cat(buffer.comm_actions).detach()
                
                # Combine physical log_prob and communication log_prob
                current_log_probs = dist.log_prob(actions) + dist_comm.log_prob(comm_actions).unsqueeze(-1)
                ratios = torch.exp(current_log_probs - old_log_probs)
                
                # Penalize the probability of speaking (Index 1 is Silence)
                prob_speak = 1.0 - dist_comm.probs[:, 1]
                comm_penalty = torch.mean(prob_speak)
                
                entropy = dist.entropy().mean() + dist_comm.entropy().mean()
            else:
                dist, _ = self.actor(obs, hiddens)
                ratios = torch.exp(dist.log_prob(actions) - old_log_probs)
                comm_penalty = 0.0
                entropy = dist.entropy().mean()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            
            # --- FIX: Applied Static Goldilocks Penalty ---
            if self.algo == "comm_mappo" and current_ep > self.warm_up_episodes:
                actor_loss = actor_loss + (self.comm_penalty_coef * comm_penalty)

            critic_loss = nn.MSELoss()(values, returns)

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