import sys, os, torch, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.beer_game_env import BeerGameParallelEnv
from agents.rl.mappo import MAPPOActor, CommMAPPOActor

def sterman_heuristic(obs, max_order=100):
    # obs[0] is inventory, obs[1] is backlog
    net_inv = obs[0] - obs[1]
    # Expected demand is roughly 8 based on your env
    order = max(0, 8 + 0.5 * (24 - net_inv)) 
    # Must normalize between 0 and 1 because your env multiplies by max_order
    return min(1.0, order / max_order)

def run_scenario(algo, model_path, env_cfg, num_episodes=10):
    env = BeerGameParallelEnv(env_cfg)
    
    actor = None
    if algo != "sterman_heuristic":
        local_dim = env.observation_space("retailer").shape[0]
        actor = CommMAPPOActor(local_dim, 128) if algo == "comm_mappo" else MAPPOActor(local_dim, 128)
        actor.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        actor.eval()
    
    costs, bullwhip_ratios = [], []
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep+42) # Fixed seeds for fairness
        hidden = {a: torch.zeros(1, 1, 128) for a in env.agents}
        msg = {a: torch.zeros(1, 1) for a in env.agents}
        ep_cost = 0
        
        # Track orders to calculate Bullwhip (Variance of Upstream vs Downstream)
        retailer_orders, manufacturer_orders = [], []
        
        while True:
            acts = {}
            for i, a in enumerate(env.agents):
                if algo == "sterman_heuristic": 
                    acts[a] = [sterman_heuristic(obs[a], env.max_order)]
                else:
                    with torch.no_grad():
                        o_t = torch.tensor(obs[a], dtype=torch.float32).unsqueeze(0)
                        if algo == "comm_mappo":
                            dist, comm, next_h = actor(o_t, msg[a], hidden[a])
                            if i < len(env.agents)-1: msg[env.agents[i+1]] = comm
                        else: dist, next_h = actor(o_t, hidden[a])
                    acts[a] = [dist.sample().item()]
                    hidden[a] = next_h
            
            # Un-normalize actions to get actual order quantities for variance calculation
            retailer_orders.append(acts["retailer"][0] * env.max_order)
            manufacturer_orders.append(acts["manufacturer"][0] * env.max_order)
            
            obs, rewards, terms, _, _ = env.step(acts)
            ep_cost -= rewards["retailer"] # As per your env setup
            if any(terms.values()): break
            
        costs.append(ep_cost)
        
        # Bullwhip Effect = Variance(Upstream Orders) / Variance(Downstream Orders)
        var_downstream = np.var(retailer_orders)
        var_upstream = np.var(manufacturer_orders)
        bw = var_upstream / var_downstream if var_downstream > 0 else 1.0
        bullwhip_ratios.append(bw)
        
    return np.mean(costs), np.mean(bullwhip_ratios)

if __name__ == "__main__":
    configs = {"sterman_heuristic": None, "ippo": "ippo_best.pth", "mappo": "mappo_best.pth", "comm_mappo": "comm_mappo_best.pth"}
    
    # Utilizing your built-in environment configurations
    scenarios = {
        "1. Baseline": {"demand_type": "poisson", "jittery_lead_time": False},
        "2. Black Swan (Demand Shock)": {"demand_type": "black_swan", "jittery_lead_time": False},
        "3. Supply Shock (Jittery Transit)": {"demand_type": "poisson", "jittery_lead_time": True},
        "4. Perfect Storm (Shock + Jitter)": {"demand_type": "black_swan", "jittery_lead_time": True}
    }
    
    results = []
    print("\nRunning Zero-Shot OOD Stress Tests...\n")
    
    for scenario_name, env_cfg in scenarios.items():
        for algo, path in configs.items():
            if algo == "sterman_heuristic" or os.path.exists(path):
                mean_cost, bw_ratio = run_scenario(algo, path, env_cfg)
                results.append({
                    "Scenario": scenario_name,
                    "Topology": algo.upper(),
                    "Cost": mean_cost,
                    "Bullwhip": bw_ratio
                })

    df = pd.DataFrame(results)
    pivot_cost = df.pivot(index="Topology", columns="Scenario", values="Cost")
    pivot_bw = df.pivot(index="Topology", columns="Scenario", values="Bullwhip")
    
    print("="*80)
    print("       TABLE 4: ZERO-SHOT OOD TOTAL COSTS (STRESS TEST)")
    print("="*80)
    print(pivot_cost.round(2).to_string())
    print("\n--- LaTeX Format ---")
    print(pivot_cost.to_latex(float_format="%.2f"))

    print("\n" + "="*80)
    print("       TABLE 5: ZERO-SHOT OOD BULLWHIP RATIOS (STRESS TEST)")
    print("="*80)
    print(pivot_bw.round(2).to_string())

    # Visualization
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Scenario", y="Cost", hue="Topology", data=df, palette="viridis")
    plt.title("Zero-Shot Performance Under Extreme Supply Chain Shocks")
    plt.ylabel("Total System Cost (Log Scale)")
    plt.xlabel("Market Scenario")
    plt.yscale("log") # Sterman will explode here, log scale is mandatory
    plt.legend(title="Topology")
    plt.tight_layout()
    plt.savefig("scenario_stress_test.png", dpi=300)
    print("\nStress test visualization saved as 'scenario_stress_test.png'.")