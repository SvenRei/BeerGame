import sys, os, torch, random, wandb, numpy as np
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
import hydra
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.beer_game_env import BeerGameParallelEnv
from agents.rl.qmix import CommQMixLocalAgent, QMixer

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, obs, acts, reward, next_state, next_obs, done):
        self.buffer.append((state, obs, acts, reward, next_state, next_obs, done))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Initialize W&B tracking
    run = wandb.init(project="BeerGame_Research", config=dict(cfg), name="comm_qmix")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.env.demand_type = "poisson"
    env = BeerGameParallelEnv(cfg.env)
    
    # --- W&B SWEEP PARAMETER INJECTION ---
    cfg.agent.lr = wandb.config.get("lr", cfg.agent.lr)
    cfg.agent.target_update_freq = wandb.config.get("target_update_freq", cfg.agent.target_update_freq)
    cfg.agent.batch_size = wandb.config.get("batch_size", cfg.agent.batch_size)
    
    # --- DYNAMIC CHECKPOINT DIRECTORY CREATION ---
    run_dir = f"weights_comm_qmix/run_{run.name}_{run.id}"
    os.makedirs(run_dir, exist_ok=True)
    
    local_dim = env.observation_space("retailer").shape[0]
    state_dim = local_dim * len(env.agents)
    n_actions = cfg.agent.n_actions 
    hidden_dim = cfg.agent.hidden_dim
    
    # CRITICAL: Define the hardcoded downstream sequential order required for DIAL
    comm_order = ["retailer", "wholesaler", "distributor", "manufacturer"]
    
    # 1. Initialize Online Networks using the new CommQMixLocalAgent
    mac = {a: CommQMixLocalAgent(local_dim, hidden_dim, n_actions).to(device) for a in env.agents}
    mixer = QMixer(len(env.agents), state_dim, cfg.agent.mixing_embed_dim, cfg.agent.hypernet_embed).to(device)
    
    # 2. Initialize Target Networks
    target_mac = {a: CommQMixLocalAgent(local_dim, hidden_dim, n_actions).to(device) for a in env.agents}
    target_mixer = QMixer(len(env.agents), state_dim, cfg.agent.mixing_embed_dim, cfg.agent.hypernet_embed).to(device)
    
    for a in env.agents: target_mac[a].load_state_dict(mac[a].state_dict())
    target_mixer.load_state_dict(mixer.state_dict())
    
    # 3. Setup Optimizer and Memory
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

    print(f"--- Starting COMM_QMIX Baseline Training Marathon ---")
    print(f"Target Save Directory: {run_dir}")

    for ep in range(cfg.total_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        hidden = {a: torch.zeros(1, hidden_dim).to(device) for a in env.agents}
        ep_cost = 0.0
        
        while True:
            acts, env_acts = {}, {}
            sorted_agents = sorted(env.agents)
            state = np.concatenate([obs[a] for a in sorted_agents])
            
            # Initialize the physical execution message chain. Retailer starts with Silence (0.0).
            current_msg = torch.zeros(1, 1).to(device)
            
            # --- ACTION & MESSAGE SELECTION (Sequential Downstream) ---
            for i, a in enumerate(comm_order):
                o_t = torch.tensor(obs[a], dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    # Pass the current observation and the incoming message into the network
                    q_vals, msg_out, next_h = mac[a](o_t, current_msg, hidden[a])
                
                if random.random() < epsilon:
                    action_idx = random.randint(0, n_actions - 1)
                else:
                    action_idx = q_vals.argmax(dim=1).item()
                
                acts[a] = action_idx
                hidden[a] = next_h.squeeze(1)
                env_acts[a] = [action_idx / (n_actions - 1)]
                
                # --- APPLY LATENT CHANNEL DROPOUT ---
                if i < len(comm_order) - 1:
                    if torch.rand(1).item() < 0.10:
                        current_msg = torch.zeros(1, 1).to(device) # Network disconnect
                    else:
                        current_msg = msg_out # Hand off generated message to next agent
            
            next_obs, rewards, terms, truncs, infos = env.step(env_acts)
            ep_cost += sum(infos[a]["local_cost"] for a in env.agents)
            global_reward = -sum(infos[a]["local_cost"] for a in env.agents) / 100.0
            next_state = np.concatenate([next_obs[a] for a in sorted_agents])
            done = any(terms.values())
            
            # We do NOT save the messages in the buffer. DIAL mathematically reconstructs 
            # them during the training pass to maintain the unbroken PyTorch gradient graph.
            buffer.push(state, obs, acts, global_reward, next_state, next_obs, done)
            
            obs = next_obs
            global_step += 1
            
            # --- TRAINING STEP ---
            if len(buffer) > cfg.agent.batch_size:
                batch = buffer.sample(cfg.agent.batch_size)
                
                b_states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32).to(device)
                b_rewards = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32).unsqueeze(1).to(device)
                b_next_states = torch.tensor(np.array([b[4] for b in batch]), dtype=torch.float32).to(device)
                b_dones = torch.tensor(np.array([b[6] for b in batch]), dtype=torch.float32).unsqueeze(1).to(device)
                
                # Use dictionaries to prevent Matrix misalignment when concatenating for the Mixer
                q_evals_dict, target_q_evals_dict = {}, {}
                
                # Initialize silence for the Retailer at the start of the batch training graph
                b_msg_online = torch.zeros(cfg.agent.batch_size, 1).to(device)
                b_msg_target = torch.zeros(cfg.agent.batch_size, 1).to(device)
                
                # CRITICAL DIAL STEP: Process agents in downstream order to weave the gradients together
                for a in comm_order:
                    b_o = torch.tensor(np.array([b[1][a] for b in batch]), dtype=torch.float32).to(device)
                    b_next_o = torch.tensor(np.array([b[5][a] for b in batch]), dtype=torch.float32).to(device)
                    b_a = torch.tensor(np.array([b[2][a] for b in batch]), dtype=torch.long).unsqueeze(1).to(device)
                    b_h = torch.zeros(cfg.agent.batch_size, hidden_dim).to(device)
                    
                    # 1. ONLINE NETWORK (Gradient Chain)
                    q_val, next_msg_online, _ = mac[a](b_o, b_msg_online, b_h)
                    q_evals_dict[a] = q_val.gather(1, b_a)
                    
                    # 2. TARGET NETWORK (Detached Evaluation)
                    with torch.no_grad():
                        # Use detached online message to select action (DDQN)
                        online_next_q, _, _ = mac[a](b_next_o, b_msg_online.detach(), b_h)
                        best_next_actions = online_next_q.argmax(dim=1, keepdim=True)
                        
                        target_q, next_msg_target, _ = target_mac[a](b_next_o, b_msg_target, b_h)
                        target_q_evals_dict[a] = target_q.gather(1, best_next_actions)
                    
                    # 3. MESSAGE HANDOFF WITH DROPOUT
                    # Apply identical 10% dropout to the training graph to make the policies robust
                    dropout_mask = (torch.rand(cfg.agent.batch_size, 1).to(device) >= 0.10).float()
                    b_msg_online = next_msg_online * dropout_mask
                    b_msg_target = next_msg_target * dropout_mask
                
                # RE-ALIGNMENT: Concatenate Q-values strictly in alphabetical order 
                # (distributor, manufacturer, retailer, wholesaler) to perfectly match the Mixer's state vector.
                q_evals = torch.cat([q_evals_dict[a] for a in sorted_agents], dim=1)
                target_q_evals = torch.cat([target_q_evals_dict[a] for a in sorted_agents], dim=1)
                
                q_tot = mixer(q_evals, b_states)
                with torch.no_grad():
                    target_q_tot = target_mixer(target_q_evals, b_next_states)
                
                targets = b_rewards + cfg.agent.gamma * (1 - b_dones) * target_q_tot.squeeze(2)
                loss = nn.MSELoss()(q_tot.squeeze(2), targets.detach())
                
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
            "Cost": ep_cost, "Avg_Cost_50": avg_cost, 
            "Epsilon": epsilon, "LR": scheduler.get_last_lr()[0]
        })
        
        if ep == warm_up:
            best_avg_cost = float('inf')
            since_imp = 0
            
        if avg_cost < best_avg_cost and len(cost_history) == 50: 
            best_avg_cost = avg_cost
            since_imp = 0
            if ep >= warm_up:
                for a in env.agents:
                    torch.save(mac[a].state_dict(), f"{run_dir}/comm_qmix_agent_{a}_best.pth")
                with open(f"{run_dir}/description.txt", "w") as f:
                    f.write(f"W&B Run Name: {run.name}\n")
                    f.write(f"Best Avg Cost: {best_avg_cost}\n")
        else:
            if ep >= warm_up: since_imp += 1
                
        exploration_lock = max(warm_up, eps_decay_eps)
        if ep > exploration_lock and since_imp >= patience:
            break
                
        if ep % 10 == 0: 
            best_display = best_avg_cost if best_avg_cost != float('inf') else 0.0
            print(f"Ep {ep} | Cost: {ep_cost:.2f} | 50-Ep Avg: {avg_cost:.2f} | Best Avg: {best_display:.2f} | Eps: {epsilon:.2f}")
            
    wandb.finish()

if __name__ == "__main__": main()