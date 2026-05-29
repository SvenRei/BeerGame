import sys
import os
import torch
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from envs.beer_game_env import BeerGameParallelEnv
from agents.rl.mappo import MAPPOActor, CommMAPPOActor

# Import the Symbolic Regression Engine
from pysr import PySRRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def gather_trajectory_data(cfg, algo, device, num_episodes=50):
    """Runs the trained model and records state-action pairs."""
    env = BeerGameParallelEnv(cfg.env)
    local_obs_dim = env.observation_space("retailer").shape[0]
    
    if algo == "comm_mappo":
        actor = CommMAPPOActor(local_obs_dim, cfg.agent.hidden_dim).to(device)
    else:
        actor = MAPPOActor(local_obs_dim, cfg.agent.hidden_dim).to(device)
        
    model_path = f"{algo}_actor_final.pth"
    if os.path.exists(model_path):
        actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        actor.eval()
        print(f"[+] Loaded {algo} for Data Extraction.")
    else:
        raise FileNotFoundError(f"Cannot extract! {model_path} missing.")

    # Data collection arrays
    X_data = []
    y_data = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=2000 + episode)
        hidden = {a: torch.zeros(1, cfg.agent.hidden_dim).to(device) for a in env.agents}
        messages = {a: torch.zeros(1, 1).to(device) for a in env.agents}
        
        while True:
            actions_dict = {}
            new_messages = {}
            
            for i, agent in enumerate(env.agents):
                local_obs_t = torch.tensor(obs[agent], dtype=torch.float32).unsqueeze(0).to(device)
                comm_in_t = messages[agent]
                
                with torch.no_grad():
                    if algo == "comm_mappo":
                        dist, comm_out, next_hidden = actor(local_obs_t, comm_in_t, hidden[agent])
                    else:
                        dist, next_hidden = actor(local_obs_t, hidden[agent])
                        comm_out = torch.zeros(1, 1).to(device)
                        
                    # Use the deterministic mean for heuristic extraction
                    action = dist.mean 
                
                actions_dict[agent] = [action.cpu().item()]
                hidden[agent] = next_hidden
                
                # --- RECORD DATA (Focusing on Retailer as our heuristic proxy) ---
                if agent == "retailer":
                    # Convert observation tensor to flat numpy array
                    state_array = local_obs_t.cpu().numpy().flatten()
                    if algo == "comm_mappo":
                        # Append the incoming message to the state features
                        state_array = np.append(state_array, comm_in_t.cpu().numpy().flatten())
                    
                    X_data.append(state_array)
                    y_data.append(action.cpu().item())
                # -----------------------------------------------------------------

                if i < len(env.agents) - 1:
                    new_messages[env.agents[i + 1]] = comm_out
                    
            messages = new_messages
            messages["retailer"] = torch.zeros(1, 1).to(device)
            
            obs, _, terms, _, _ = env.step(actions_dict)
            if any(terms.values()): break

    return np.array(X_data), np.array(y_data)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algo = cfg.agent.get("algorithm", "mappo").lower()
    
    print(f"\n--- Starting Heuristic Extraction for {algo.upper()} ---")
    
    # 1. Gather Data
    X, y = gather_trajectory_data(cfg, algo, device)
    print(f"[+] Gathered {len(X)} State-Action pairs.")
    
    # 2. Define human-readable feature names (Based on your 6-dim obs space)
    feature_names = ["Inv", "Backlog", "Inc_Orders", "Pipe_1", "Pipe_2", "Pipe_3"]
    if algo == "comm_mappo":
        feature_names.append("Comm_Signal")
        
    df_X = pd.DataFrame(X, columns=feature_names)
    
    # 3. Configure Symbolic Regression
    # We restrict operators to simple math to keep the heuristic human-readable
    print("[+] Unleashing PySR to find the optimal mathematical heuristic...")
    model = PySRRegressor(
        niterations=50,             # Number of evolutionary generations
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp"],    # Allow exponential decay/growth smoothing
        maxsize=15,                 # Prevent massive, unreadable equations
        model_selection="best",     # Pick the best accuracy/complexity tradeoff
        equation_file=f"heuristic_{algo}.csv"
    )
    
    # 4. Fit the model
    model.fit(df_X, y)
    
    print(f"\n--- Extraction Complete for {algo.upper()} ---")
    print("The discovered heuristic equation is:")
    print(model.sympy()) # Prints the mathematical formula

if __name__ == "__main__":
    main()