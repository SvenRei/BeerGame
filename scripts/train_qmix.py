# Import the sys module to interact with the Python runtime environment
import sys
# Import the os module to handle dynamic file paths and directory structures
import os
# Import the core PyTorch deep learning library
import torch
# Import PyTorch's neural network module for loss functions
import torch.nn as nn
# Import PyTorch's optimization algorithms (like Adam)
import torch.optim as optim
# Import NumPy for efficient numerical operations and arrays
import numpy as np
# Import Weights & Biases for live training dashboard logging
import wandb
# Import Hydra for hierarchical configuration management
import hydra
# Import the random module for epsilon-greedy exploration and buffer sampling
import random
# Import DictConfig for type hinting the Hydra configuration object
from omegaconf import DictConfig
# Import deque to create an efficient, fixed-size memory buffer
from collections import deque

# Setup the system path to locate the root directory (two levels up)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import the custom multi-agent Beer Game environment
from envs.beer_game_env import BeerGameParallelEnv
# Import the QMIX local agent and Hypernetwork architectures
from agents.rl.qmix import QMixLocalAgent, QMixer

# ==============================================================================
# REPLAY BUFFER (Off-Policy Memory)
# ==============================================================================
# Define the memory buffer class to store past experiences
class ReplayBuffer:
    # Initialize the buffer with a maximum capacity
    def __init__(self, capacity):
        # Create a deque that automatically drops the oldest memories when full
        self.buffer = deque(maxlen=capacity)
    
    # Define how to push a new experience tuple into the memory
    def push(self, state, obs, acts, reward, next_state, next_obs, done):
        # Append the full transition tuple to the deque
        self.buffer.append((state, obs, acts, reward, next_state, next_obs, done))
    
    # Define how to sample a random batch of memories for training
    def sample(self, batch_size):
        # Use random.sample to grab a unique subset of experiences
        return random.sample(self.buffer, batch_size)
    
    # Define a method to quickly check how many memories are currently stored
    def __len__(self):
        # Return the integer length of the deque
        return len(self.buffer)

# ==============================================================================
# MAIN TRAINING MARATHON
# ==============================================================================
# Use the Hydra decorator to automatically load and compose the configuration
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Initialize a new W&B run to track the QMIX baseline performance
    wandb.init(project="BeerGame_Research", config=dict(cfg), name="qmix_baseline")
    # Automatically select the GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Force the environment to use the peaceful Poisson demand for baseline training
    cfg.env.demand_type = "poisson"
    # Instantiate the Beer Game simulator with the config parameters
    env = BeerGameParallelEnv(cfg.env)
    
    # Extract the observation vector size (local state) from the Retailer
    local_dim = env.observation_space("retailer").shape[0]
    # Calculate the global state size (local state size * 4 agents)
    state_dim = local_dim * len(env.agents)
    
    # Extract structural hyperparams correctly from the composed cfg.agent block
    n_actions = cfg.agent.n_actions 
    hidden_dim = cfg.agent.hidden_dim
    
    # 1. Initialize Online Networks
    # Instantiate isolated local brains for each of the 4 agents
    mac = {a: QMixLocalAgent(local_dim, hidden_dim, n_actions).to(device) for a in env.agents}
    # Instantiate the Centralized Hypernetwork Mixer
    mixer = QMixer(len(env.agents), state_dim, cfg.agent.mixing_embed_dim, cfg.agent.hypernet_embed).to(device)
    
    # 2. Initialize Target Networks
    # Instantiate frozen copies of the local brains for stable TD-targets
    target_mac = {a: QMixLocalAgent(local_dim, hidden_dim, n_actions).to(device) for a in env.agents}
    # Instantiate a frozen copy of the Mixer
    target_mixer = QMixer(len(env.agents), state_dim, cfg.agent.mixing_embed_dim, cfg.agent.hypernet_embed).to(device)
    
    # Synchronize the Target network weights to exactly match the Online network initialization
    for a in env.agents: target_mac[a].load_state_dict(mac[a].state_dict())
    target_mixer.load_state_dict(mixer.state_dict())
    
    # 3. Setup Optimizer and Memory
    # Aggregate all parameters from the mixer and all 4 agents into one master list
    all_params = list(mixer.parameters())
    for a in env.agents: all_params += list(mac[a].parameters())
    # Initialize the Adam optimizer to update everything simultaneously
    optimizer = optim.Adam(all_params, lr=cfg.agent.lr)
    
    # Initialize the learning rate scheduler to decay the learning rate by half every 2000 steps
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    
    # Instantiate the Replay Buffer with the capacity defined in config
    buffer = ReplayBuffer(cfg.agent.buffer_size)
    
    # --- FIX: Dynamic Patience Loading ---
    patience = cfg.agent.get("patience", 2000)
    since_imp = 0
    warm_up = cfg.agent.get("warm_up_episodes", 1000)
    eps_decay_eps = cfg.agent.get("epsilon_decay_episodes", 5000)
    
    # Deque to maintain a rolling average of the last 50 episode costs
    cost_history = deque(maxlen=50)
    # Variable to lock in the best performance ever achieved
    best_avg_cost = float('inf')
    # Counter for total physical steps taken across all episodes
    global_step = 0
    # Initialize exploration rate
    epsilon = cfg.agent.epsilon_start

    # Print initialization confirmation to the terminal
    print(f"--- Starting QMIX Baseline Training Marathon ---")
    print(f"Discretization: {n_actions} bins | Buffer Size: {cfg.agent.buffer_size}")
    print(f"Warm-up: {warm_up} | Epsilon Decay: {eps_decay_eps} | Patience: {patience}")
    print(f"NOTE: Early stopping is strictly locked until episode {max(warm_up, eps_decay_eps)}.")

    # Begin the main episode loop
    for ep in range(cfg.total_episodes):
        # Reset environment with a deterministic seed
        obs, _ = env.reset(seed=1000 + ep)
        # Reset the GRU memory states to zero for all agents
        hidden = {a: torch.zeros(1, hidden_dim).to(device) for a in env.agents}
        # Reset episode cost accumulator
        ep_cost = 0.0
        
        # Begin the physical step loop
        while True:
            # Dictionaries to hold discrete network output and continuous physical output
            acts, env_acts = {}, {}
            # Sort agents strictly to ensure the global state array is ordered reliably
            sorted_agents = sorted(env.agents)
            # Concatenate local observations to form the global centralized state
            state = np.concatenate([obs[a] for a in sorted_agents])
            
            # --- ACTION SELECTION ---
            for a in env.agents:
                # Convert NumPy observation into PyTorch tensor
                o_t = torch.tensor(obs[a], dtype=torch.float32).unsqueeze(0).to(device)
                
                # Perform pure inference without tracking gradients
                with torch.no_grad():
                    # Get discrete Q-values and next memory state
                    q_vals, next_h = mac[a](o_t, hidden[a])
                
                # Epsilon-Greedy logic: Random exploration
                if random.random() < epsilon:
                    action_idx = random.randint(0, n_actions - 1)
                # Epsilon-Greedy logic: Network exploitation
                else:
                    action_idx = q_vals.argmax(dim=1).item()
                
                # Save chosen discrete action
                acts[a] = action_idx
                # Update memory for next step
                hidden[a] = next_h.squeeze(1)
                
                # Convert the discrete action bin into a physical continuous order fraction
                physical_fraction = action_idx / (n_actions - 1)
                env_acts[a] = [physical_fraction]
            
            # --- ENVIRONMENT STEP ---
            # Execute physical actions in simulator
            next_obs, rewards, terms, truncs, infos = env.step(env_acts)
            
            # Accumulate physical cost
            ep_cost += sum(infos[a]["local_cost"] for a in env.agents)
            # Define cooperative global reward required by QMIX (negative scaled sum of costs)
            global_reward = -sum(infos[a]["local_cost"] for a in env.agents) / 100.0
            # Build the next step's global state array
            next_state = np.concatenate([next_obs[a] for a in sorted_agents])
            # Check for terminal state flag
            done = any(terms.values())
            
            # Store the full transition inside the memory buffer
            buffer.push(state, obs, acts, global_reward, next_state, next_obs, done)
            
            # Advance observation pointers
            obs = next_obs
            # Increment global step counter
            global_step += 1
            
            # --- TRAINING STEP ---
            # Wait until buffer has enough samples to form a complete batch
            if len(buffer) > cfg.agent.batch_size:
                # Sample random experiences
                batch = buffer.sample(cfg.agent.batch_size)
                
                # Extract and format states, rewards, next states, and done flags
                b_states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32).to(device)
                b_rewards = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32).unsqueeze(1).to(device)
                b_next_states = torch.tensor(np.array([b[4] for b in batch]), dtype=torch.float32).to(device)
                b_dones = torch.tensor(np.array([b[6] for b in batch]), dtype=torch.float32).unsqueeze(1).to(device)
                
                # Initialize lists to aggregate Q-values
                q_evals, target_q_evals = [], []
                
                for a in sorted_agents:
                    # Extract local agent data from the batch
                    b_o = torch.tensor(np.array([b[1][a] for b in batch]), dtype=torch.float32).to(device)
                    b_next_o = torch.tensor(np.array([b[5][a] for b in batch]), dtype=torch.float32).to(device)
                    b_a = torch.tensor(np.array([b[2][a] for b in batch]), dtype=torch.long).unsqueeze(1).to(device)
                    b_h = torch.zeros(cfg.agent.batch_size, hidden_dim).to(device)
                    
                    # Compute Q-values for the specific actions taken
                    q_val, _ = mac[a](b_o, b_h)
                    q_evals.append(q_val.gather(1, b_a))
                    
                    # Compute maximum future Q-values from the target network
                    with torch.no_grad():
                        target_q, _ = target_mac[a](b_next_o, b_h)
                        target_q_evals.append(target_q.max(dim=1, keepdim=True)[0])
                
                # Concatenate individual Q-values into a single tensor
                q_evals = torch.cat(q_evals, dim=1)
                target_q_evals = torch.cat(target_q_evals, dim=1)
                
                # Pass through the Online Mixer to get estimated total value
                q_tot = mixer(q_evals, b_states)
                # Pass through Target Mixer to get objective target value
                with torch.no_grad():
                    target_q_tot = target_mixer(target_q_evals, b_next_states)
                
                # Calculate the TD-Target via the Bellman Equation
                targets = b_rewards + cfg.agent.gamma * (1 - b_dones) * target_q_tot.squeeze(2)
                # Calculate Mean Squared Error
                loss = nn.MSELoss()(q_tot.squeeze(2), targets.detach())
                
                # Clear old gradients
                optimizer.zero_grad()
                # Backpropagate error
                loss.backward()
                # Clip gradients strictly to 5.0 to prevent hypernetwork explosions
                torch.nn.utils.clip_grad_norm_(all_params, 5.0)
                # Execute weight updates
                optimizer.step()
                
            # Copy Online weights to Target network on designated frequency
            if global_step % cfg.agent.target_update_freq == 0:
                for a in env.agents: target_mac[a].load_state_dict(mac[a].state_dict())
                target_mixer.load_state_dict(mixer.state_dict())
                
            # End physical loop if episode terminates
            if done: break
            
        # --- END OF EPISODE MANAGEMENT ---
        # Update learning rate schedule
        scheduler.step()
        
        # Calculate Epsilon decay mathematically
        epsilon = max(cfg.agent.epsilon_end, cfg.agent.epsilon_start - ep / cfg.agent.epsilon_decay_episodes)
        
        # Update rolling cost tracking
        cost_history.append(ep_cost)
        avg_cost = sum(cost_history) / len(cost_history)
        
        # Push metrics to W&B dashboard
        wandb.log({
            "Cost": ep_cost, 
            "Avg_Cost_50": avg_cost, 
            "Epsilon": epsilon,
            "LR": scheduler.get_last_lr()[0]
        })
        
        # Trigger hard reset after noisy warm-up phase
        if ep == warm_up:
            print(f"--- Ep {ep}: Warm-up complete! Resetting early stopping baseline. ---")
            best_avg_cost = float('inf')
            since_imp = 0
            
        # Standard Checkpointing Logic
        if avg_cost < best_avg_cost and len(cost_history) == 50: 
            best_avg_cost = avg_cost
            since_imp = 0
            # Only save weights if we are out of the purely random warm-up phase
            if ep >= warm_up:
                for a in env.agents:
                    torch.save(mac[a].state_dict(), f"qmix_agent_{a}_best.pth")
        else:
            if ep >= warm_up:
                since_imp += 1
                
        # --- FIX: THE EARLY STOPPING LOCK ---
        # We mathematically forbid the script from early-stopping if Epsilon is still decaying.
        exploration_lock = max(warm_up, eps_decay_eps)
        if ep > exploration_lock and since_imp >= patience:
            print(f"Stopping early at Ep {ep}: No improvement in 50-Ep Avg Cost for {patience} episodes.")
            break
                
        # Terminal printout every 10 episodes
        if ep % 10 == 0: 
            best_display = best_avg_cost if best_avg_cost != float('inf') else 0.0
            print(f"Ep {ep} | Cost: {ep_cost:.2f} | 50-Ep Avg: {avg_cost:.2f} | Best Avg: {best_display:.2f} | Eps: {epsilon:.2f}")
            
    # Close W&B logger gracefully
    wandb.finish()

# Execute script directly
if __name__ == "__main__": main()