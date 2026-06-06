import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
import numpy as np
from comm_utils import get_vocab_tensor

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.local_obs, self.global_states, self.hidden_states = [], [], []
        self.comm_in, self.actions, self.log_probs = [], [], []
        self.comm_actions = [] 
        self.rewards, self.is_terminals = [], []

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
        x = self.fc(obs).unsqueeze(0)
        hidden = hidden.unsqueeze(0)
        x, hidden = self.gru(x, hidden)
        x = x.squeeze(0)
        hidden = hidden.squeeze(0)
        
        mean = torch.sigmoid(self.action_mean(x))
        std = self.action_log_std.exp().expand_as(mean)
        return Normal(mean, std), hidden

class CommMAPPOActor(nn.Module):
    def __init__(self, obs_dim, hidden_dim, vocab_size=3, comm_dim=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.fc = nn.Sequential(
            nn.Linear(obs_dim + comm_dim, hidden_dim), nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=False)
        self.action_mean = nn.Linear(hidden_dim, 1)
        self.action_log_std = nn.Parameter(torch.full((1, 1), -1.0)) 
        self.comm_head = nn.Linear(hidden_dim, vocab_size)

        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        nn.init.constant_(self.action_mean.bias, -0.75)

    def forward(self, obs, comm_in, hidden, tau=1.0):
        x = torch.cat([obs, comm_in], dim=-1)
        x = self.fc(x).unsqueeze(0)
        hidden = hidden.unsqueeze(0)
        x, hidden = self.gru(x, hidden)
        
        x_flat = x.squeeze(0)
        hidden = hidden.squeeze(0)
        
        mean = torch.sigmoid(self.action_mean(x_flat))
        std = self.action_log_std.exp().expand_as(mean)
        
        # Use Gumbel-Softmax for differentiable communication
        msg_logits = self.comm_head(x_flat)
        # We need the underlying categorical for the PPO ratio calculation
        dist_comm = Categorical(logits=msg_logits)
        # We sample via Gumbel for the differentiable forward pass
        msg_probs = F.gumbel_softmax(msg_logits, tau=tau, hard=True)
        
        return Normal(mean, std), dist_comm, msg_probs, hidden

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

    def get_vocab_tensor(self, device):
        if self.vocab_size == 1: return torch.tensor([0.0], device=device)
        elif self.vocab_size == 3: return torch.tensor([-1.0, 0.0, 1.0], device=device)
        elif self.vocab_size == 5: return torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)
        else: raise ValueError("Vocab size must be 1, 3, or 5")

    def init_buffer(self, batch_size, device):
        self.msg_buffer = torch.zeros(batch_size, self.num_agents, 1, device=device)

    def forward(self, obs, hiddens, tau=1.0, test_mode=False):
        B = obs.size(0)
        vocab_tensor = self.get_vocab_tensor(obs.device)
        
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
        return self.fc(state)

class MAPPOTrainer:
    def __init__(self, actor, critic, cfg, total_episodes, device, algo):
        self.actor, self.critic, self.device, self.algo = actor, critic, device, algo
        self.gamma = cfg.get("gamma", 0.99)
        self.entropy_coef = cfg.get("entropy_coef", 0.05)
        self.comm_penalty_coef = cfg.get("comm_penalty_coef", 0.0001) 
        self.k_epochs = cfg.get("k_epochs", 4)
        
        self.warm_up_episodes = cfg.get("warm_up_episodes", 1000)
        self.total_episodes = total_episodes 
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.get("lr_actor", 3e-4))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.get("lr_critic", 1e-3))
        self.max_grad_norm = 0.2 

    def update(self, buffer, current_ep):
        obs = torch.cat(buffer.local_obs).detach()
        # FIX: Removed the invalid .transpose(0, 1) so it safely enters the GRU loop
        hiddens = torch.cat(buffer.hidden_states).detach()
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
        
        for _ in range(self.k_epochs):
            if self.algo == "ippo":
                values = self.critic(obs)
            else:
                values = self.critic(torch.cat(buffer.global_states).detach())
            
            advantages = returns - values.detach()
            
            if self.algo == "comm_mappo":
                base_actor = self.actor.actor
                dist, dist_comm, _ = base_actor(obs, torch.cat(buffer.comm_in).detach(), hiddens)
                comm_actions = torch.cat(buffer.comm_actions).detach()
                
                current_log_probs = dist.log_prob(actions) + dist_comm.log_prob(comm_actions).unsqueeze(-1)
                ratios = torch.exp(current_log_probs - old_log_probs)
                
                out_features = base_actor.comm_head.out_features
                if out_features > 1:
                    silence_idx = out_features // 2
                    prob_speak = 1.0 - dist_comm.probs[:, silence_idx]
                    comm_penalty = torch.mean(prob_speak)
                else:
                    comm_penalty = 0.0
                    
                entropy = dist.entropy().mean() + dist_comm.entropy().mean()
            else:
                dist, _ = self.actor(obs, hiddens)
                ratios = torch.exp(dist.log_prob(actions) - old_log_probs)
                comm_penalty = 0.0
                entropy = dist.entropy().mean()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            
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