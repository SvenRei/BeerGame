import sys, os, torch, random, wandb, numpy as np
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
import hydra
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.beer_game_env import BeerGameParallelEnv
from agents.rl.qmix import CommQMixLocalAgent, QMixCommMAC, QMixer

# ENGINEER FIX: ReplayBuffer now stores BOTH hidden and next_hidden to preserve temporal alignment
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    # Added next_hidden to the push signature
    def push(self, state, obs, acts, reward, next_state, next_obs, done, hidden, next_hidden):
        self.buffer.append((state, obs, acts, reward, next_state, next_obs, done, hidden, next_hidden))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    run = wandb.init(project="BeerGame_Research", config=dict(cfg), name="comm_qmix")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.env.demand_type = "poisson"
    env = BeerGameParallelEnv(cfg.env)
    
    cfg.agent.lr = wandb.config.get("lr", cfg.agent.lr)
    cfg.agent.target_update_freq = wandb.config.get("target_update_freq", cfg.agent.target_update_freq)
    cfg.agent.batch_size = wandb.config.get("batch_size", cfg.agent.batch_size)
    vocab_size = wandb.config.get("vocab_size", cfg.agent.get("vocab_size", 3))
    
    run_dir = f"weights_comm_qmix/run_{run.name}_{run.id}"
    os.makedirs(run_dir, exist_ok=True)
    
    local_dim = env.observation_space("retailer").shape[0]
    state_dim = local_dim * len(env.agents)
    n_actions = cfg.agent.n_actions 
    hidden_dim = cfg.agent.hidden_dim
    
    base_agent = CommQMixLocalAgent(local_dim, hidden_dim, n_actions, vocab_size=vocab_size)
    mac = QMixCommMAC(base_agent, num_agents=len(env.agents)).to(device)
    mixer = QMixer(len(env.agents), state_dim, cfg.agent.mixing_embed_dim, cfg.agent.hypernet_embed).to(device)
    
    target_base = CommQMixLocalAgent(local_dim, hidden_dim, n_actions, vocab_size=vocab_size)
    target_mac = QMixCommMAC(target_base, num_agents=len(env.agents)).to(device)
    target_mixer = QMixer(len(env.agents), state_dim, cfg.agent.mixing_embed_dim, cfg.agent.hypernet_embed).to(device)
    
    target_mac.load_state_dict(mac.state_dict())
    target_mixer.load_state_dict(mixer.state_dict())
    
    all_params = list(mixer.parameters()) + list(mac.parameters())
    optimizer = optim.Adam(all_params, lr=cfg.agent.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    
    buffer = ReplayBuffer(cfg.agent.buffer_size)
    
    patience, since_imp = cfg.agent.get("patience", 2000), 0
    warm_up = cfg.agent.get("warm_up_episodes", 1000)
    eps_decay_eps = cfg.agent.get("epsilon_decay_episodes", 5000)
    cost_history = deque(maxlen=50)
    best_avg_cost, global_step, epsilon = float('inf'), 0, cfg.agent.epsilon_start

    print(f"--- Starting COMM_QMIX Training Marathon ---")

    for ep in range(cfg.total_episodes):
        tau = max(0.05, 5.0 * (1.0 - ep / cfg.total_episodes))
        
        obs, _ = env.reset(seed=1000 + ep)
        mac.init_buffer(batch_size=1, device=device)
        hiddens_tensor = torch.zeros(1, len(env.agents), hidden_dim).to(device)
        
        ep_cost = 0.0
        episode_messages = []
        
        while True:
            obs_array = np.stack([obs[a] for a in env.agents])
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0).to(device)
            state = np.concatenate([obs[a] for a in sorted(env.agents)])
            
            with torch.no_grad():
                q_vals, next_hiddens, safe_logs = mac(obs_tensor, hiddens_tensor, tau=tau)
                episode_messages.append(safe_logs)
                
            acts, env_acts = {}, {}
            actions_list = []
            
            for i, a in enumerate(env.agents):
                if random.random() < epsilon:
                    action_idx = random.randint(0, n_actions - 1)
                else:
                    action_idx = q_vals[0, i].argmax(dim=-1).item()
                
                acts[a] = action_idx
                actions_list.append(action_idx)
                env_acts[a] = [action_idx / (n_actions - 1)]
                
            next_obs, rewards, terms, truncs, infos = env.step(env_acts)
            raw_cost = sum(infos[a]["local_cost"] for a in env.agents)
            ep_cost += raw_cost
            # ENGINEER FIX: Log-scale the reward to prevent Q-value gradient explosion
            # np.log1p(x) safely calculates log(1 + x), compressing 80,000 down to a safe -11.2 penalty
            global_reward = -np.log1p(raw_cost)

            next_state = np.concatenate([next_obs[a] for a in sorted(env.agents)])
            next_obs_array = np.stack([next_obs[a] for a in env.agents])
            
            done = any(terms.values()) or any(truncs.values())
            
            # ENGINEER FIX: Pushing BOTH current hiddens AND next hiddens
            buffer.push(state, obs_array, actions_list, global_reward, next_state, next_obs_array, done, hiddens_tensor.detach().cpu().numpy(), next_hiddens.detach().cpu().numpy())
            
            obs = next_obs
            hiddens_tensor = next_hiddens
            global_step += 1
            
            if len(buffer) > cfg.agent.batch_size:
                batch = buffer.sample(cfg.agent.batch_size)
                
                b_states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32).to(device)
                b_obs = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32).to(device)
                b_actions = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.long).unsqueeze(-1).to(device)
                b_rewards = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32).unsqueeze(1).to(device)
                b_next_states = torch.tensor(np.array([b[4] for b in batch]), dtype=torch.float32).to(device)
                b_next_obs = torch.tensor(np.array([b[5] for b in batch]), dtype=torch.float32).to(device)
                b_dones = torch.tensor(np.array([b[6] for b in batch]), dtype=torch.float32).unsqueeze(1).to(device)
                
                # ENGINEER FIX: Extract BOTH hidden states
                b_h = torch.tensor(np.array([b[7] for b in batch]), dtype=torch.float32).to(device)
                b_next_h = torch.tensor(np.array([b[8] for b in batch]), dtype=torch.float32).to(device)
                
                mac.init_buffer(cfg.agent.batch_size, device)
                target_mac.init_buffer(cfg.agent.batch_size, device)
                
                # Calculate current Q values using b_h
                q_evals, _, _ = mac(b_obs, b_h, tau=tau)
                chosen_q_evals = torch.gather(q_evals, dim=2, index=b_actions)
                
                with torch.no_grad():
                    # ENGINEER FIX: Calculate Next Q values strictly using b_next_h
                    online_next_q, _, _ = mac(b_next_obs, b_next_h, tau=tau)
                    best_next_actions = online_next_q.argmax(dim=2, keepdim=True)
                    target_q, _, _ = target_mac(b_next_obs, b_next_h, tau=tau)
                    target_q_evals = torch.gather(target_q, dim=2, index=best_next_actions)
                    
                q_tot = mixer(chosen_q_evals, b_states)
                with torch.no_grad():
                    target_q_tot = target_mixer(target_q_evals, b_next_states)
                    
                targets = b_rewards + cfg.agent.gamma * (1 - b_dones) * target_q_tot.squeeze(2)
                loss = nn.MSELoss()(q_tot.squeeze(2), targets.detach())
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 5.0)
                optimizer.step()
                
            if global_step % cfg.agent.target_update_freq == 0:
                target_mac.load_state_dict(mac.state_dict())
                target_mixer.load_state_dict(mixer.state_dict())
                
            if done: break
            
        scheduler.step()
        epsilon = max(cfg.agent.epsilon_end, cfg.agent.epsilon_start - ep / cfg.agent.epsilon_decay_episodes)
        
        cost_history.append(ep_cost)
        avg_cost = sum(cost_history) / len(cost_history)
        
        wandb.log({"Cost": ep_cost, "Avg_Cost_50": avg_cost, "Epsilon": epsilon, "Tau": tau, "LR": scheduler.get_last_lr()[0]})
        
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
        if ep % 10 == 0: 
            print(f"Ep {ep} | Cost: {ep_cost:.2f} | 50-Ep Avg: {avg_cost:.2f} | Best: {best_avg_cost if best_avg_cost != float('inf') else 0.0:.2f} | Eps: {epsilon:.2f} | Tau: {tau:.2f}")
            
    wandb.finish()

if __name__ == "__main__": main()