import sys, os, torch, random, wandb, numpy as np
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
import hydra
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.beer_game_env import BeerGameParallelEnv
from agents.rl.qmix import QMixLocalAgent, QMixer

# P1 FIX: Episode Replay Buffer
class EpisodeReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, states, obs, actions, rewards, dones):
        self.buffer.append({
            "states": np.array(states, dtype=np.float32),
            "obs": np.array(obs, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32)
        })
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return {k: torch.tensor(np.stack([b[k] for b in batch])) for k in batch[0].keys()}
    
    def __len__(self):
        return len(self.buffer)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    run = wandb.init(project="BeerGame_Research", config=dict(cfg), name="qmix_baseline")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg.env.demand_type = cfg.env.get("demand_type", "step")
    env = BeerGameParallelEnv(cfg.env)
    
    obs, _ = env.reset(seed=1000)
    dummy_global = env.get_global_state()
    state_dim = len(dummy_global)
    
    run_dir = os.path.join("weights_qmix", f"run_{run.name}_{run.id}")
    os.makedirs(run_dir, exist_ok=True)
    
    local_dim = env.observation_space("retailer").shape[0]
    n_actions = cfg.agent.n_actions 
    if n_actions < 2: raise ValueError(f"n_actions must be >= 2, got {n_actions}")
    
    hidden_dim = cfg.agent.hidden_dim
    
    mac = {a: QMixLocalAgent(local_dim, hidden_dim, n_actions).to(device) for a in env.possible_agents}
    mixer = QMixer(len(env.possible_agents), state_dim, cfg.agent.mixing_embed_dim, cfg.agent.hypernet_embed).to(device)
    target_mac = {a: QMixLocalAgent(local_dim, hidden_dim, n_actions).to(device) for a in env.possible_agents}
    target_mixer = QMixer(len(env.possible_agents), state_dim, cfg.agent.mixing_embed_dim, cfg.agent.hypernet_embed).to(device)
    
    for a in env.possible_agents: target_mac[a].load_state_dict(mac[a].state_dict())
    target_mixer.load_state_dict(mixer.state_dict())
    
    all_params = list(mixer.parameters())
    for a in env.possible_agents: all_params += list(mac[a].parameters())
    optimizer = optim.Adam(all_params, lr=cfg.agent.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    
    buffer = EpisodeReplayBuffer(cfg.agent.buffer_size)
    
    patience, since_imp = cfg.agent.get("patience", 2000), 0
    warm_up = cfg.agent.get("warm_up_episodes", 1000)
    eps_decay_eps = cfg.agent.get("epsilon_decay_episodes", 5000)
    cost_history = deque(maxlen=50)
    best_avg_cost, global_step, epsilon = float('inf'), 0, cfg.agent.epsilon_start

    print(f"--- Starting QMIX Sequence Replay Marathon ---")

    for ep in range(cfg.total_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        hidden = {a: torch.zeros(1, hidden_dim).to(device) for a in env.possible_agents}
        
        ep_states, ep_obs, ep_actions, ep_rewards, ep_dones = [], [], [], [], []
        ep_cost = 0.0
        ep_agent_costs = {a: 0.0 for a in env.possible_agents}
        
        # Store the very first T=0 state
        ep_obs.append(np.stack([obs[a] for a in env.possible_agents]))
        ep_states.append(env.get_global_state())
        
        while True:
            acts, env_acts, actions_list = {}, {}, []
            current_agents = env.possible_agents
            
            for i, a in enumerate(current_agents):
                o_t = torch.tensor(obs[a], dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_vals, next_h = mac[a](o_t, hidden[a].detach()) 
                hidden[a] = next_h
                
                if random.random() < epsilon: action_idx = random.randint(0, n_actions - 1)
                else: action_idx = q_vals.argmax(dim=1).item()
                
                acts[a] = action_idx
                actions_list.append([action_idx])
                env_acts[a] = [action_idx / max(1, n_actions - 1)]
                
            if env.current_step % 10 == 0:
                wandb.log({f"Order_Qty/{a}": float(np.round(env_acts[a][0] * env.max_order)) for a in current_agents}, commit=False)

            next_obs, rewards, terms, truncs, infos = env.step(env_acts)
            
            raw_cost = 0.0
            for a in current_agents:
                local_cost = infos[a]["local_cost"]
                raw_cost += local_cost
                ep_agent_costs[a] += local_cost
                
            ep_cost += raw_cost
            global_reward = -raw_cost / 100.0
            done = any(terms.values()) or any(truncs.values())
            
            ep_actions.append(actions_list)
            ep_rewards.append([global_reward])
            ep_dones.append([float(done)])
            ep_obs.append(np.stack([next_obs[a] for a in current_agents]))
            ep_states.append(env.get_global_state())
            
            obs = next_obs
            global_step += 1
            
            # ---------------- BATCH TRAINING ----------------
            if len(buffer) > cfg.agent.batch_size:
                batch = buffer.sample(cfg.agent.batch_size)
                
                b_states = batch["states"].to(device) # [B, T+1, state_dim]
                b_obs = batch["obs"].to(device)       # [B, T+1, N, obs_dim]
                b_actions = batch["actions"].to(device) # [B, T, N, 1]
                b_rewards = batch["rewards"].to(device) # [B, T, 1]
                b_dones = batch["dones"].to(device)     # [B, T, 1]
                
                B, T_plus_1, N, _ = b_obs.shape
                T = T_plus_1 - 1
                
                q_evals_agents, target_q_evals_agents = [], []
                
                # Unroll through time for each agent independently
                for i, a in enumerate(current_agents):
                    h_train = torch.zeros(B, hidden_dim).to(device)
                    target_h_train = torch.zeros(B, hidden_dim).to(device)
                    
                    q_agent, target_q_agent = [], []
                    
                    for t in range(T_plus_1):
                        q, h_train = mac[a](b_obs[:, t, i, :], h_train)
                        q_agent.append(q)
                        with torch.no_grad():
                            target_q, target_h_train = target_mac[a](b_obs[:, t, i, :], target_h_train)
                            target_q_agent.append(target_q)
                            
                    q_agent = torch.stack(q_agent, dim=1)           # [B, T+1, n_actions]
                    target_q_agent = torch.stack(target_q_agent, dim=1) 
                    
                    # Gather Q values for the actions taken
                    chosen_q = q_agent[:, :-1, :].gather(2, b_actions[:, :, i, :]) # [B, T, 1]
                    q_evals_agents.append(chosen_q)
                    
                    # Double Q-Learning Target logic
                    best_next_actions = q_agent[:, 1:, :].argmax(dim=2, keepdim=True)
                    target_q_gathered = target_q_agent[:, 1:, :].gather(2, best_next_actions) # [B, T, 1]
                    target_q_evals_agents.append(target_q_gathered)
                    
                q_evals = torch.cat(q_evals_agents, dim=2)        # [B, T, N]
                target_q_evals = torch.cat(target_q_evals_agents, dim=2) 
                
                b_states_t = b_states[:, :-1, :]   # [B, T, state_dim]
                b_states_next = b_states[:, 1:, :] # [B, T, state_dim]
                
                # Reshape for Mixing Network
                q_tot = mixer(q_evals.reshape(B*T, N, 1), b_states_t.reshape(B*T, -1))
                with torch.no_grad():
                    target_q_tot = target_mixer(target_q_evals.reshape(B*T, N, 1), b_states_next.reshape(B*T, -1))
                    
                q_tot_flat = q_tot.reshape(B*T, 1)
                targets = b_rewards.reshape(B*T, 1) + cfg.agent.gamma * (1 - b_dones.reshape(B*T, 1)) * target_q_tot.reshape(B*T, 1)
                
                loss = nn.MSELoss()(q_tot_flat, targets.detach())
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 5.0)
                optimizer.step()
                
            if global_step % cfg.agent.target_update_freq == 0:
                for a in current_agents: target_mac[a].load_state_dict(mac[a].state_dict())
                target_mixer.load_state_dict(mixer.state_dict())
                
            if done: break
            
        # Push the fully populated episode trajectory to the buffer
        buffer.push(ep_states, ep_obs, ep_actions, ep_rewards, ep_dones)
        
        scheduler.step()
        decay_step = (cfg.agent.epsilon_start - cfg.agent.epsilon_end) / cfg.agent.epsilon_decay_episodes
        epsilon = max(cfg.agent.epsilon_end, cfg.agent.epsilon_start - decay_step * ep)
        
        cost_history.append(ep_cost)
        avg_cost = sum(cost_history) / len(cost_history)
        
        log_dict = {"Cost": ep_cost, "Avg_Cost_50": avg_cost, "Epsilon": epsilon, "LR": scheduler.get_last_lr()[0]}
        for a, cost in ep_agent_costs.items(): log_dict[f"Cost/{a}"] = cost
        wandb.log(log_dict)
        
        if ep == warm_up: best_avg_cost, since_imp = float('inf'), 0
        if avg_cost < best_avg_cost and len(cost_history) == 50: 
            best_avg_cost = avg_cost
            since_imp = 0
            if ep >= warm_up:
                for a in env.possible_agents:
                    torch.save(mac[a].state_dict(), os.path.join(run_dir, f"qmix_agent_{a}_best.pth"))
        else:
            if ep >= warm_up: since_imp += 1
                
        exploration_lock = max(warm_up, eps_decay_eps)
        if ep > exploration_lock and since_imp >= patience: break
        if ep % 10 == 0: print(f"Ep {ep} | Cost: {ep_cost:.2f} | 50-Ep Avg: {avg_cost:.2f} | Best: {best_avg_cost if best_avg_cost != float('inf') else 0.0:.2f} | Eps: {epsilon:.2f}")
            
    wandb.finish()

if __name__ == "__main__": main()