import sys, os, numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.beer_game_env import BeerGameParallelEnv

def sterman_heuristic(obs, max_order=100):
    # Anchor-Adjust policy: 4 units demand anchor, 0.5 gain, 12 target inventory
    # obs[0] = Inventory, obs[1] = Backlog
    net_inventory = obs[0] - obs[1]
    order = max(0, 4 + 0.5 * (12 - net_inventory))
    
    # Normalize to [0.0, 1.0] to match the environment's action space
    return min(1.0, order / max_order)

def run_sterman_baseline(num_episodes=50):
    # Match the config used in your training
    config = {"horizon": 50, "max_order": 100, "demand_type": "step"}
    env = BeerGameParallelEnv(config)
    all_costs = []
    
    print(f"--- Running Sterman Heuristic Baseline for {num_episodes} episodes ---")
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep + 1000)
        ep_cost = 0
        while True:
            # The heuristic is applied per agent based on their local observation
            acts = {a: [sterman_heuristic(obs[a], env.max_order)] for a in env.agents}
            obs, rewards, terms, _, _ = env.step(acts)
            
            # Sum the cost: rewards are negative total_system_cost
            ep_cost -= rewards["retailer"] 
            
            if any(terms.values()): break
        
        all_costs.append(ep_cost)
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1} | Cost: {ep_cost:.2f}")
            
    print(f"\nBaseline Performance: Mean Cost {np.mean(all_costs):.2f} (±{np.std(all_costs):.2f})")

if __name__ == "__main__":
    run_sterman_baseline()