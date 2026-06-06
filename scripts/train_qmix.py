import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import hydra
import random
from omegaconf import DictConfig
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.beer_game_env import BeerGameParallelEnv
from agents.rl.qmix import QMixLocalAgent, QMixer

# ENGINEER FIX: ReplayBuffer must store hidden and next_hidden for the GRUCell
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, obs, acts, reward, next_state, next_obs, done, hidden, next_hidden):
        self.buffer.append((state, obs, acts, reward, next_state, next_obs, done, hidden, next_hidden))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    run = wandb.init(project="BeerGame_Research", config=dict(cfg), name="qmix_baseline")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg.env.demand_type = "poisson"
    env = BeerGameParallelEnv(cfg.env)
    
    run_dir = os.path.join("weights_qmix", f"run_{run.name}_{run.id}")
    os.makedirs(run_dir, exist_ok=True)
    
    local_dim = env.observation_space("retailer").shape[0]
    
    # ENGINEER FIX: Use the true, unclipped CTDE global state
    dummy_global = env.get_global_state()
    state_dim = len(dummy_global)
    
    n_actions = cfg.agent.n_actions 
    hidden_dim = cfg.agent.hidden_dim
    
    mac = {a: QMixLocalAgent(local_dim, hidden_dim, n_actions).to(device) for a in env.agents}
    mixer = QMixer(len(env.agents), state_dim, cfg.agent.mixing_embed_dim, cfg.agent.hypernet_embed).to(device)
    
    target_mac = {a: QMixLocalAgent(local_dim, hidden_dim, n_actions).to(device) for a in env.agents}
    target_mixer = QMixer(len(env.agents), state_dim, cfg.agent.mixing_embed_dim, cfg.agent.hypernet_embed).to(device)
    
    for a in env.agents: target_mac[a].load_state_dict(mac[a].state_dict())
    target_mixer.load_state_dict(mixer.state_dict())
    
    all_params = list(mixer.parameters())
    for a in env.agents: all_params += list(mac[a].parameters())
    optimizer = optim.Adam(all_params, lr=cfg.agent.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    
    buffer = ReplayBuffer(cfg.agent.buffer_size)
    
    patience = cfg.agent.get("patience", 2000)
    since_imp = 0
    warm_up = cfg.agent.get("warm_up_episodes", 1000)
    eps_decay_eps = cfg.agent.get("epsilon_decay_episodes", 5000)
    
    cost_history = deque(maxlen=50)
    best_avg_cost = float('inf')
    global_step = 0
    epsilon = cfg.agent.epsilon_start

    print(f"--- Starting QMIX Baseline Training Marathon ---")

    for ep in range(cfg.total_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        hidden = {a: torch.zeros(1, hidden_dim).to(device) for a in env.agents}
        ep_cost = 0.0
        
        while True:
            acts, env_acts = {}, {}
            sorted_agents = sorted(env.agents)
            
            # ENGINEER FIX: Fetch true global state
            state = env.get_global_state()
            
            next_hiddens_dict = {}
            for a in env.agents:
                o_t = torch.tensor(obs[a], dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    # Evaluate Q-values
                    q_vals, next_h = mac[a](o_t, hidden[a].detach()) # FIX: Detach to prevent memory leak
                
                next_hiddens_dict[a] = next_h
                
                if random.random() < epsilon:
                    action_idx = random.randint(0, n_actions - 1)
                else:
                    action_idx = q_vals.argmax(dim=1).item()
                
                acts[a] = action_idx
                # FAIR BENCHMARK: Map discrete index to continuous space
                env_acts[a] = [action_idx / (n_actions - 1)]
            
            next_obs, rewards, terms, truncs, infos = env.step(env_acts)
            
            raw_cost = sum(infos[a]["local_cost"] for a in env.agents)
            ep_cost += raw_cost
            global_reward = -np.log1p(raw_cost)
            
            next_state = env.get_global_state()
            done = any(terms.values()) or any(truncs.values())
            
            # ENGINEER FIX: Store hiddens as grouped arrays to match the CommQMix batching style
            h_array = np.stack([hidden[a].cpu().numpy().squeeze() for a in sorted_agents])
            next_h_array = np.stack([next_hiddens_dict[a].cpu().numpy().squeeze() for a in sorted_agents])
            
            buffer.push(state, obs, acts, global_reward, next_state, next_obs, done, h_array, next_h_array)
            
            obs = next_obs
            # Update hiddens for next step, detached!
            for a in env.agents:
                hidden[a] = next_hiddens_dict[a].detach()
                
            global_step += 1
            
            if len(buffer) > cfg.agent.batch_size:
                batch = buffer.sample(cfg.agent.batch_size)
                
                b_states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32).to(device)
                b_rewards = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32).unsqueeze(1).to(device)
                b_next_states = torch.tensor(np.array([b[4] for b in batch]), dtype=torch.float32).to(device)
                b_dones = torch.tensor(np.array([b[6] for b in batch]), dtype=torch.float32).unsqueeze(1).to(device)
                
                b_h = torch.tensor(np.array([b[7] for b in batch]), dtype=torch.float32).to(device)
                b_next_h = torch.tensor(np.array([b[8] for b in batch]), dtype=torch.float32).to(device)
                
                q_evals, target_q_evals = [], []
                
                for idx, a in enumerate(sorted_agents):
                    b_o = torch.tensor(np.array([b[1][a] for b in batch]), dtype=torch.float32).to(device)
                    b_next_o = torch.tensor(np.array([b[5][a] for b in batch]), dtype=torch.float32).to(device)
                    b_a = torch.tensor(np.array([b[2][a] for b in batch]), dtype=torch.long).unsqueeze(1).to(device)
                    
                    # ENGINEER FIX: Pass the stored hidden states, NOT zeros
                    b_h_agent = b_h[:, idx, :] 
                    b_next_h_agent = b_next_h[:, idx, :]
                    
                    q_val, _ = mac[a](b_o, b_h_agent)
                    q_evals.append(q_val.gather(1, b_a).unsqueeze(1))
                    
                    with torch.no_grad():
                        online_next_q, _ = mac[a](b_next_o, b_next_h_agent)
                        best_next_actions = online_next_q.argmax(dim=1, keepdim=True)
                        
                        target_q, _ = target_mac[a](b_next_o, b_next_h_agent)
                        target_q_evals.append(target_q.gather(1, best_next_actions).unsqueeze(1))
                
                q_evals = torch.cat(q_evals, dim=1)
                target_q_evals = torch.cat(target_q_evals, dim=1)
                
                q_tot = mixer(q_evals, b_states)
                with torch.no_grad():
                    target_q_tot = target_mixer(target_q_evals, b_next_states)
                # HARDENED TARGETS: Ensure shape [batch_size, 1]
                # If target_q_tot is [B, 1, 1], squeeze(2) makes it [B, 1]
                q_tot_flat = q_tot.reshape(cfg.agent.batch_size, 1)
                target_q_tot_flat = target_q_tot.reshape(cfg.agent.batch_size, 1)
                
                # Targets: rewards [B, 1] + gamma * [B, 1]
                targets = b_rewards + cfg.agent.gamma * (1 - b_dones) * target_q_tot_flat
                
                # MSELoss needs [batch_size, 1] vs [batch_size, 1]
                loss = nn.MSELoss()(q_tot_flat, targets.detach())
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 5.0)
                optimizer.step()
                
            if global_step % cfg.agent.target_update_freq == 0:
                for a in env.agents: target_mac[a].load_state_dict(mac[a].state_dict())
                target_mixer.load_state_dict(mixer.state_dict())
                
            if done: break
            
        scheduler.step()
        epsilon = max(cfg.agent.epsilon_end, cfg.agent.epsilon_start - ep / cfg.agent.epsilon_decay_episodes)
        
        cost_history.append(ep_cost)
        avg_cost = sum(cost_history) / len(cost_history)
        
        wandb.log({
            "Cost": ep_cost, 
            "Avg_Cost_50": avg_cost, 
            "Epsilon": epsilon,
            "LR": scheduler.get_last_lr()[0]
        })
        
        if ep == warm_up: best_avg_cost, since_imp = float('inf'), 0
            
        if avg_cost < best_avg_cost and len(cost_history) == 50: 
            best_avg_cost = avg_cost
            since_imp = 0
            if ep >= warm_up:
                for a in env.agents:
                    save_path = os.path.join(run_dir, f"qmix_agent_{a}_best.pth")
                    torch.save(mac[a].state_dict(), save_path)
        else:
            if ep >= warm_up: since_imp += 1
                
        exploration_lock = max(warm_up, eps_decay_eps)
        if ep > exploration_lock and since_imp >= patience: break
        if ep % 10 == 0: 
            best_display = best_avg_cost if best_avg_cost != float('inf') else 0.0
            print(f"Ep {ep} | Cost: {ep_cost:.2f} | 50-Ep Avg: {avg_cost:.2f} | Best Avg: {best_display:.2f} | Eps: {epsilon:.2f}")
            
    wandb.finish()

if __name__ == "__main__": main()