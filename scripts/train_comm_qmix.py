import sys, os, torch, random, wandb, numpy as np
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
import hydra
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.beer_game_env import BeerGameParallelEnv
from agents.rl.qmix import CommQMixLocalAgent, QMixCommMAC, QMixer

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
    def __len__(self): return len(self.buffer)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    run = wandb.init(project="BeerGame_Research", config=dict(cfg), name="comm_qmix")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg.env.demand_type = cfg.env.get("demand_type", "step")
    env = BeerGameParallelEnv(cfg.env)
    
    obs, _ = env.reset(seed=1000)
    dummy_global = env.get_global_state()
    state_dim = len(dummy_global)
    
    cfg.agent.lr = wandb.config.get("lr", cfg.agent.lr)
    cfg.agent.target_update_freq = wandb.config.get("target_update_freq", cfg.agent.target_update_freq)
    cfg.agent.batch_size = wandb.config.get("batch_size", cfg.agent.batch_size)
    vocab_size = wandb.config.get("vocab_size", cfg.agent.get("vocab_size", 3))
    
    run_dir = f"weights_comm_qmix/run_{run.name}_{run.id}"
    os.makedirs(run_dir, exist_ok=True)
    
    local_dim = env.observation_space("retailer").shape[0]
    n_actions = cfg.agent.n_actions 
    if n_actions < 2: raise ValueError(f"n_actions must be >= 2")
    hidden_dim = cfg.agent.hidden_dim
    
    base_agent = CommQMixLocalAgent(local_dim, hidden_dim, n_actions, vocab_size=vocab_size)
    mac = QMixCommMAC(base_agent, num_agents=len(env.possible_agents)).to(device)
    mixer = QMixer(len(env.possible_agents), state_dim, cfg.agent.mixing_embed_dim, cfg.agent.hypernet_embed).to(device)
    
    target_base = CommQMixLocalAgent(local_dim, hidden_dim, n_actions, vocab_size=vocab_size)
    target_mac = QMixCommMAC(target_base, num_agents=len(env.possible_agents)).to(device)
    target_mixer = QMixer(len(env.possible_agents), state_dim, cfg.agent.mixing_embed_dim, cfg.agent.hypernet_embed).to(device)
    
    target_mac.load_state_dict(mac.state_dict())
    target_mixer.load_state_dict(mixer.state_dict())
    
    all_params = list(mixer.parameters()) + list(mac.parameters())
    optimizer = optim.Adam(all_params, lr=cfg.agent.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    
    buffer = EpisodeReplayBuffer(cfg.agent.buffer_size)
    
    patience, since_imp = cfg.agent.get("patience", 2000), 0
    warm_up = cfg.agent.get("warm_up_episodes", 1000)
    eps_decay_eps = cfg.agent.get("epsilon_decay_episodes", 5000)
    cost_history = deque(maxlen=50)
    best_avg_cost, global_step, epsilon = float('inf'), 0, cfg.agent.epsilon_start
    
    tau_start, tau_min = 1.0, 0.1
    tau_decay_episodes = cfg.total_episodes * 0.5

    print(f"--- Starting COMM_QMIX Sequence Replay Marathon ---")

    for ep in range(cfg.total_episodes):
        if ep >= tau_decay_episodes: tau = tau_min
        else: tau = tau_start - (tau_start - tau_min) * (ep / tau_decay_episodes)
        
        obs, _ = env.reset(seed=1000 + ep)
        mac.init_buffer(batch_size=1, device=device)
        hiddens_tensor = torch.zeros(1, len(env.possible_agents), hidden_dim).to(device)
        
        ep_states, ep_obs, ep_actions, ep_rewards, ep_dones = [], [], [], [], []
        ep_cost = 0.0
        ep_agent_costs = {a: 0.0 for a in env.possible_agents}
        episode_messages = []
        
        ep_obs.append(np.stack([obs[a] for a in env.possible_agents]))
        ep_states.append(env.get_global_state())
        
        while True:
            current_agents = env.possible_agents 
            obs_array = np.stack([obs[a] for a in current_agents])
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                q_vals, next_hiddens, _, safe_logs = mac(obs_tensor, hiddens_tensor.detach(), tau=tau)
                episode_messages.append(safe_logs)
                
            acts, env_acts, actions_list = {}, {}, []
            for i, a in enumerate(current_agents):
                if random.random() < epsilon: action_idx = random.randint(0, n_actions - 1)
                else: action_idx = q_vals[0, i].argmax(dim=-1).item()
                
                acts[a] = action_idx
                actions_list.append([action_idx])
                env_acts[a] = [action_idx / max(1, n_actions - 1)]
                
            if env.current_step % 10 == 0: wandb.log({f"Order_Qty/{a}": float(np.round(env_acts[a][0] * env.max_order)) for a in current_agents}, commit=False)

            next_obs, rewards, terms, truncs, infos = env.step(env_acts)
            
            raw_cost = sum(infos[a].get("local_cost", 0.0) for a in current_agents)
            ep_cost += raw_cost
            for a in current_agents: ep_agent_costs[a] += infos[a].get("local_cost", 0.0)
            
            global_reward = -raw_cost / 100.0 
            done = any(terms.values()) or any(truncs.values())
            
            ep_actions.append(actions_list)
            ep_rewards.append([global_reward])
            ep_dones.append([float(done)])
            ep_obs.append(np.stack([next_obs[a] for a in current_agents]))
            ep_states.append(env.get_global_state())
            
            obs = next_obs
            hiddens_tensor = next_hiddens.detach() 
            global_step += 1
            
            # ---------------- BATCH TRAINING (BPTT through Messages) ----------------
            if len(buffer) > cfg.agent.batch_size:
                batch = buffer.sample(cfg.agent.batch_size)
                
                b_states = batch["states"].to(device)
                b_obs = batch["obs"].to(device)       
                b_actions = batch["actions"].to(device) 
                b_rewards = batch["rewards"].to(device) 
                b_dones = batch["dones"].to(device)     
                
                B, T_plus_1, N, _ = b_obs.shape
                T = T_plus_1 - 1
                
                q_evals_list, target_q_evals_list = [], []
                
                h_train = torch.zeros(B, N, hidden_dim).to(device)
                target_h_train = torch.zeros(B, N, hidden_dim).to(device)
                
                msg_in = torch.zeros(B, N, 1).to(device)
                target_msg_in = torch.zeros(B, N, 1).to(device)
                
                # Unroll completely through time, letting gradients pass from t+1 back to t
                for t in range(T_plus_1):
                    q_t, h_train, msg_out, _ = mac(b_obs[:, t], h_train, tau=tau, msg_in=msg_in)
                    msg_in = msg_out # Differentiable link for the next loop!
                    q_evals_list.append(q_t)
                    
                    with torch.no_grad():
                        target_q_t, target_h_train, target_msg_out, _ = target_mac(b_obs[:, t], target_h_train, tau=tau, msg_in=target_msg_in)
                        target_msg_in = target_msg_out
                        target_q_evals_list.append(target_q_t)
                        
                q_evals = torch.stack(q_evals_list, dim=1) # [B, T+1, N, n_actions]
                target_q_evals = torch.stack(target_q_evals_list, dim=1)
                
                chosen_q = q_evals[:, :-1].gather(3, b_actions) # [B, T, N, 1]
                best_next_actions = q_evals[:, 1:].argmax(dim=3, keepdim=True)
                target_q_gathered = target_q_evals[:, 1:].gather(3, best_next_actions) 
                
                b_states_t = b_states[:, :-1, :]
                b_states_next = b_states[:, 1:, :]
                
                q_tot = mixer(chosen_q.reshape(B*T, N, 1), b_states_t.reshape(B*T, -1))
                with torch.no_grad(): target_q_tot = target_mixer(target_q_gathered.reshape(B*T, N, 1), b_states_next.reshape(B*T, -1))
                    
                q_tot_flat = q_tot.reshape(B*T, 1)
                targets = b_rewards.reshape(B*T, 1) + cfg.agent.gamma * (1 - b_dones.reshape(B*T, 1)) * target_q_tot.reshape(B*T, 1)
                
                loss = nn.MSELoss()(q_tot_flat, targets.detach())
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 5.0)
                optimizer.step()
                
            if global_step % cfg.agent.target_update_freq == 0:
                target_mac.load_state_dict(mac.state_dict())
                target_mixer.load_state_dict(mixer.state_dict())
                
            if done: break
            
        buffer.push(ep_states, ep_obs, ep_actions, ep_rewards, ep_dones)
        scheduler.step()
        
        decay_step = (cfg.agent.epsilon_start - cfg.agent.epsilon_end) / cfg.agent.epsilon_decay_episodes
        epsilon = max(cfg.agent.epsilon_end, cfg.agent.epsilon_start - decay_step * ep)
        
        cost_history.append(ep_cost)
        avg_cost = sum(cost_history) / len(cost_history)
        
        log_dict = {"Cost": ep_cost, "Avg_Cost_50": avg_cost, "Epsilon": epsilon, "Tau": tau, "LR": scheduler.get_last_lr()[0]}
        for a, cost in ep_agent_costs.items(): log_dict[f"Cost/{a}"] = cost
            
        if len(episode_messages) > 0:
            all_msgs = np.concatenate(episode_messages, axis=0).flatten().astype(int)
            log_dict["Comm/Message_Distribution"] = wandb.Histogram(all_msgs)
            log_dict["Comm/Unique_Tokens"] = len(np.unique(all_msgs))
            vocab_size = 3
            token_counts = np.bincount(all_msgs, minlength=vocab_size)
            token_pcts = (token_counts / len(all_msgs)) * 100.0
            for v in range(vocab_size): log_dict[f"Comm/Token_{v}_Pct"] = token_pcts[v]
            
        wandb.log(log_dict)
        
        if ep == warm_up: best_avg_cost, since_imp = float('inf'), 0
        if avg_cost < best_avg_cost and len(cost_history) == 50: 
            best_avg_cost = avg_cost
            since_imp = 0
            if ep >= warm_up:
                torch.save(mac.state_dict(), f"{run_dir}/comm_qmix_mac_best.pth")
                np.save(f"{run_dir}/best_messages_ep_{ep}.npy", np.concatenate(episode_messages, axis=0))
        else:
            if ep >= warm_up: since_imp += 1
                
        exploration_lock = max(warm_up, eps_decay_eps)
        if ep > exploration_lock and since_imp >= patience: break
        if ep % 10 == 0: print(f"Ep {ep} | Cost: {ep_cost:.2f} | 50-Ep Avg: {avg_cost:.2f} | Best: {best_avg_cost if best_avg_cost != float('inf') else 0.0:.2f} | Eps: {epsilon:.2f} | Tau: {tau:.2f}")
            
    wandb.finish()

if __name__ == "__main__": main()