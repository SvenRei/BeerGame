import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations

# Setup Absolute Project Root Path to prevent Hydra folder errors
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from envs.beer_game_env import BeerGameParallelEnv
from agents.rl.mappo import MAPPOActor, CommMAPPOActor

def cohens_d(x, y):
    n1, n2 = len(x), len(y)
    var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
    if var1 == 0 and var2 == 0: return 0.0 
    s = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(x) - np.mean(y)) / s if s != 0 else 0

def sterman_heuristic(obs, max_order=100):
    net_order = max(0, 4 + 0.5 * (12 - (obs[0] - obs[1])))
    return min(1.0, net_order / max_order)

def run_benchmark(algo, model_path, scenario_type, num_episodes=100):
    env = BeerGameParallelEnv({"demand_type": scenario_type, "horizon": 50, "max_order": 100})
    actor = None
    
    if algo != "sterman_heuristic":
        local_dim = env.observation_space("retailer").shape[0]
        actor = CommMAPPOActor(local_dim, 128) if algo == "comm_mappo" else MAPPOActor(local_dim, 128)
        actor.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        actor.eval()
    
    costs, episode_bullwhip_ratios, episode_comm_corrs = [], [], []
    raw_actor_floats = [] 
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=2000 + ep) 
        hidden, msg, ep_cost = {a: torch.zeros(1, 1, 128) for a in env.agents}, {a: torch.zeros(1, 1) for a in env.agents}, 0
        m_orders, r_demands, ep_ret_backlogs, ep_ret_msgs = [], [], [], []
        
        while True:
            acts = {}
            next_msg = {} 
            
            for i, a in enumerate(env.agents):
                if algo == "sterman_heuristic": 
                    acts[a] = [sterman_heuristic(obs[a], env.max_order)]
                else:
                    with torch.no_grad():
                        o_t = torch.tensor(obs[a], dtype=torch.float32).unsqueeze(0)
                        if algo == "comm_mappo":
                            dist, comm, next_h = actor(o_t, msg[a], hidden[a])
                            if i < len(env.agents)-1: next_msg[env.agents[i+1]] = comm
                            if a == "retailer":
                                ep_ret_msgs.append(comm.item())
                                ep_ret_backlogs.append(obs[a][1]) 
                        else: 
                            dist, next_h = actor(o_t, hidden[a])
                    acts[a] = [dist.mean.item()]
                    hidden[a] = next_h
                    
                    if ep == 0 and env.current_step < 3 and a == "retailer":
                        raw_actor_floats.append(acts[a][0])
            
            msg = next_msg
            msg["retailer"] = torch.zeros(1, 1) 
            
            scaled_m_order = int(np.round(np.clip(acts["manufacturer"][0], 0.0, 1.0) * env.max_order))
            m_orders.append(scaled_m_order)
            
            # --- NEW DEMAND TRACKING FOR ALL SCENARIOS ---
            if scenario_type == "step": 
                true_demand = 4 if env.current_step < 4 else 8
            elif scenario_type == "black_swan": 
                true_demand = 8 if env.current_step < 25 else 20
            elif scenario_type == "extreme_chaos":
                if env.current_step < 10: true_demand = 8
                elif env.current_step < 20: true_demand = 30
                elif env.current_step < 30: true_demand = 0
                else: true_demand = 15 # The expected mean of the random (5, 25) range
            else: 
                true_demand = 8  
                
            r_demands.append(true_demand)
            obs, rewards, terms, _, _ = env.step(acts)
            ep_cost -= rewards["retailer"]
            if any(terms.values()): break
            
        costs.append(ep_cost)
        var_demand = np.var(r_demands)
        episode_bullwhip_ratios.append(np.var(m_orders) / var_demand if var_demand > 0 else 1.0)
        
        if algo == "comm_mappo" and len(ep_ret_msgs) > 1 and np.var(ep_ret_backlogs) > 0:
            corr, _ = stats.spearmanr(ep_ret_msgs, ep_ret_backlogs)
            episode_comm_corrs.append(corr if not np.isnan(corr) else 0.0)
        else:
            episode_comm_corrs.append(0.0)
            
    return np.array(costs), np.array(episode_bullwhip_ratios), np.array(episode_comm_corrs), raw_actor_floats

if __name__ == "__main__":
    # --- ADDED 'extreme_chaos' TO THE SCENARIOS LIST ---
    scenarios = ["step", "poisson", "black_swan", "extreme_chaos"]
    configs = {
        "sterman_heuristic": None, 
        "ippo": "ippo_best.pth", 
        "mappo": "mappo_best.pth", 
        "comm_mappo": "comm_mappo_best.pth"
    }
    all_scenario_summaries = []

    print("\n=======================================================")
    print("    LAUNCHING MULTI-SCENARIO ACADEMIC BENCHMARK        ")
    print(f"    PROJECT ROOT: {PROJECT_ROOT}")
    print("=======================================================")

    for scenario in scenarios:
        print(f"\n---> Executing Test Scenario: [{scenario.upper()}]")
        results, comm_correlations = {}, []
        
        for k, v in configs.items():
            if v is None:
                print(f"  -> Running {k.upper()} (Baseline)")
                costs, bw_ratios, comm_corrs, raw_floats = run_benchmark(k, v, scenario_type=scenario)
                results[k] = (costs, bw_ratios)
            else:
                abs_path = os.path.join(PROJECT_ROOT, v)
                if os.path.exists(abs_path):
                    print(f"  -> Running {k.upper()} (Found weights: {abs_path})")
                    costs, bw_ratios, comm_corrs, raw_floats = run_benchmark(k, abs_path, scenario_type=scenario)
                    results[k] = (costs, bw_ratios)
                    if k == "comm_mappo": comm_correlations = comm_corrs
                    
                    if scenario == "step":
                        print(f"    [Diagnostic] {k.upper()} Raw Network Floats (Steps 1-3): {[f'{val:.6f}' for val in raw_floats]}")
                else:
                    print(f"  [ERROR] Skipping {k.upper()} - Cannot find file at {abs_path}")
        
        if len(results) < 2:
            print(f"  [WARNING] Only {len(results)} algorithm(s) loaded. Skipping pairwise statistical tables.")
        
        for k, v in results.items():
            all_scenario_summaries.append({
                "Scenario": scenario.upper(),
                "Algo": k.upper(),
                "Mean Cost": np.mean(v[0]),
                "Std Dev": np.std(v[0]),
                "Robustness (CV)": np.std(v[0]) / np.mean(v[0]) if np.mean(v[0]) != 0 else 0,
                "Bullwhip Ratio": np.mean(v[1])
            })
            
        if len(comm_correlations) > 0:
            print(f"\n=== COMM-MAPPO LATENT SPACE ANALYSIS ({scenario.upper()}) ===")
            print(f"  -> Mean Spearman Correlation (Retailer Msg vs. Backlog): {np.mean(comm_correlations):.4f}")

        if len(results) >= 2:
            print("\n=== ASSUMPTION TESTING: NORMALITY (Shapiro-Wilk) ===")
            for algo in results.keys():
                try:
                    if np.var(results[algo][0]) == 0:
                        print(f"  {algo.upper()}: Deterministic (Zero Variance) -> NON-NORMAL")
                    else:
                        _, p_norm = stats.shapiro(results[algo][0])
                        print(f"  {algo.upper()}: p-value = {p_norm:.4e} -> {'NON-NORMAL' if p_norm < 0.05 else 'NORMAL'}")
                except Exception:
                    print(f"  {algo.upper()}: Test failed (likely identical values)")

            print("\n=== VARIANCE ANALYSIS: SYSTEMIC VOLATILITY (Ansari-Bradley) ===")
            if "sterman_heuristic" in results and "comm_mappo" in results:
                try:
                    if np.var(results["sterman_heuristic"][0]) == 0 and np.var(results["comm_mappo"][0]) == 0:
                        print("  Sterman vs Comm-MAPPO Variance p-value: N/A (Both are perfectly deterministic)")
                    else:
                        _, p_var = stats.ansari(results["sterman_heuristic"][0], results["comm_mappo"][0])
                        print(f"  Sterman vs Comm-MAPPO Variance p-value: {p_var:.4e}")
                except Exception:
                    print("  Sterman vs Comm-MAPPO Variance p-value: N/A (Test failed due to ties)")

            print("\n=== ADJUSTED PAIRWISE SIGN-RANK (With Holm-Bonferroni Correction) ===")
            raw_p_values = []
            pairs = list(combinations(results.keys(), 2))

            for a, b in pairs:
                try:
                    diff = np.array(results[a][0]) - np.array(results[b][0])
                    if np.all(diff == 0): p_val = 1.0
                    else: p_val = stats.wilcoxon(results[a][0], results[b][0]).pvalue
                except Exception:
                    p_val = 1.0
                raw_p_values.append(p_val)

            from scipy.stats import rankdata
            n_tests = len(raw_p_values)
            sort_idx = np.argsort(raw_p_values)
            adjusted_p_vals = np.zeros(n_tests)
            for i, idx in enumerate(sort_idx):
                adjusted_p_vals[idx] = min(1.0, raw_p_values[idx] * (n_tests - i))

            for idx, (a, b) in enumerate(pairs):
                print(f"  {a.upper()} vs {b.upper()}: Raw p: {raw_p_values[idx]:.4e} | Holm-Adjusted p: {adjusted_p_vals[idx]:.4e}")

        # Visualization
        plt.figure(figsize=(10, 6))
        flat_data = pd.DataFrame([(k.upper(), v_i) for k, v in results.items() for v_i in v[0]], columns=['Topology', 'Cost'])
        sns.boxplot(x='Topology', y='Cost', hue='Topology', data=flat_data, palette="viridis", legend=False)
        if "sterman_heuristic" in results:
            plt.axhline(y=np.mean(results["sterman_heuristic"][0]), color='r', linestyle='--', label="Sterman Mean Baseline")
        plt.title(f"Statistical Cost Distributions under Scenario: {scenario.upper()}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"benchmark_{scenario}_comparison.png", dpi=300)
        plt.close()

    # --- MASTER OUTPUTS & CSV GENERATION ---
    master_df = pd.DataFrame(all_scenario_summaries).set_index(["Scenario", "Algo"])
    print("\n=======================================================")
    print("--- FINAL MANUSCRIPT MASTER DATA TABLE ---")
    print("=======================================================")
    print(master_df.round(2).to_string())
    
    master_df.to_csv("master_benchmark_results.csv")
    print("\n-> Saved 'master_benchmark_results.csv' successfully.")
    print("-> All scenario comparison plots (.png) generated successfully.")