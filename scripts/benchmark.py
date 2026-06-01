import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations

# ==============================================================================
# 1. PATH SETUP & IMPORTS
# ==============================================================================
# Ensure Python can find the custom modules (envs, agents) regardless of where the script is executed.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from envs.beer_game_env import BeerGameParallelEnv
from agents.rl.mappo import MAPPOActor, CommMAPPOActor

# ==============================================================================
# 2. ADVANCED ACADEMIC METRICS (Information Theory & Statistics)
# ==============================================================================

def calc_mutual_information(x, y, bins=10):
    """
    Calculates Mutual Information (MI) in bits.
    Measures the non-linear correlation between the latent messages (x) and the physical backlog (y).
    A score > 0 proves the network invented a semantic language.
    """
    if len(x) == 0 or len(y) == 0: return 0.0
    c_xy, _, _ = np.histogram2d(x, y, bins)
    c_xy = c_xy / np.sum(c_xy) 
    p_x, p_y = np.sum(c_xy, axis=1), np.sum(c_xy, axis=0)
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if c_xy[i, j] > 0:
                mi += c_xy[i, j] * np.log2(c_xy[i, j] / (p_x[i] * p_y[j]))
    return mi

def calc_shannon_entropy(messages):
    """
    Calculates the Shannon Entropy (in bits) of the latent vocabulary.
    Measures how evenly the agent utilizes the [-1.0, 0.0, 1.0] tokens. 
    """
    if len(messages) == 0: return 0.0
    _, counts = np.unique(messages, return_counts=True)
    probs = counts / len(messages)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

def sterman_heuristic(obs, max_order=100):
    """
    The classical baseline for human bounded rationality in supply chains.
    Calculates an order based on current inventory, backlog, and a fixed target base-stock (12).
    """
    net_order = max(0, 4 + 0.5 * (12 - (obs[0] - obs[1])))
    return min(1.0, net_order / max_order)

# ==============================================================================
# 3. CORE EVALUATION LOOP
# ==============================================================================

def run_benchmark(algo, model_path, scenario_type, num_episodes=100):
    """
    Executes a strict zero-shot evaluation of a specific algorithm against a specific OOD scenario.
    """
    env = BeerGameParallelEnv({"demand_type": scenario_type, "horizon": 50, "max_order": 100})
    actor = None
    
    if algo != "sterman_heuristic":
        local_dim = env.observation_space("retailer").shape[0]
        actor = CommMAPPOActor(local_dim, 256) if algo == "comm_mappo" else MAPPOActor(local_dim, 256)
        actor.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        actor.eval()
    
    costs, episode_bullwhip_ratios = [], []
    episode_jitters, episode_sparsities = [], []
    episode_mis, episode_shannon = [], []
    episode_fill_rates, episode_holding_costs, episode_backlog_costs = [], [], []
    raw_actor_floats = [] 
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=2000 + ep) 
        hidden = {a: torch.zeros(1, 1, 256) for a in env.agents}
        msg = {a: torch.zeros(1, 1) for a in env.agents}
        
        ep_cost, ep_holding_cost, ep_backlog_cost = 0.0, 0.0, 0.0
        
        # --- FIX: Replaced volume tracking with Type 1 Service Level tracking ---
        in_stock_steps = 0  
        
        m_orders, r_demands, ep_ret_backlogs, ep_ret_msgs = [], [], [], []
        prev_acts = None
        ep_step_jitters, ep_all_msgs = [], []
        
        while True:
            acts, next_msg = {}, {}
            
            for i, a in enumerate(env.agents):
                if algo == "sterman_heuristic": 
                    acts[a] = [sterman_heuristic(obs[a], env.max_order)]
                else:
                    with torch.no_grad():
                        o_t = torch.tensor(obs[a], dtype=torch.float32).unsqueeze(0)
                        if algo == "comm_mappo":
                            dist, dist_comm, next_h = actor(o_t, msg[a], hidden[a])
                            comm_idx = torch.argmax(dist_comm.probs, dim=-1)
                            vocab = torch.tensor([-1.0, 0.0, 1.0])
                            comm_val = vocab[comm_idx].view(1, 1)
                            
                            if i < len(env.agents)-1: next_msg[env.agents[i+1]] = comm_val
                            ep_all_msgs.append(comm_val.item())
                            
                            if a == "retailer":
                                ep_ret_msgs.append(comm_val.item())
                                ep_ret_backlogs.append(obs[a][1])
                        else: 
                            dist, next_h = actor(o_t, hidden[a])
                    
                    acts[a] = [dist.mean.item()]
                    hidden[a] = next_h
                    
                    if ep == 0 and env.current_step < 3 and a == "retailer":
                        raw_actor_floats.append(acts[a][0])
            
            if prev_acts is not None:
                step_jitter = np.mean([abs(acts[a][0] - prev_acts[a]) for a in env.agents])
                ep_step_jitters.append(step_jitter)
            prev_acts = {a: acts[a][0] for a in env.agents}
            
            msg = next_msg
            msg["retailer"] = torch.zeros(1, 1) 
            
            scaled_m_order = int(np.round(np.clip(acts["manufacturer"][0], 0.0, 1.0) * env.max_order))
            m_orders.append(scaled_m_order)
            
            if scenario_type == "step": 
                true_demand = 4 if env.current_step < 4 else 8
            elif scenario_type == "black_swan": 
                true_demand = 8 if env.current_step < 25 else 20
            elif scenario_type == "extreme_chaos":
                if env.current_step < 10: true_demand = 8
                elif env.current_step < 20: true_demand = 30
                elif env.current_step < 30: true_demand = 0
                else: true_demand = 15
            else: 
                true_demand = 8  
                
            r_demands.append(true_demand)
            obs, rewards, terms, _, infos = env.step(acts)
            
            for a in env.agents:
                local_cost = infos[a]["local_cost"]
                ep_cost += local_cost
                
                current_inv = obs[a][0]
                current_backlog = obs[a][1]
                ep_holding_cost += (current_inv * env.h)
                ep_backlog_cost += (current_backlog * env.b)
                
                # --- FIX: Type 1 Service Level (In-Stock Probability) ---
                if a == "retailer":
                    if current_backlog == 0:
                        in_stock_steps += 1
            
            if any(terms.values()): break
            
        costs.append(ep_cost)
        episode_holding_costs.append(ep_holding_cost)
        episode_backlog_costs.append(ep_backlog_cost)
        
        # --- FIX: Calculate exactly what percentage of the episode was backlog-free ---
        service_level = in_stock_steps / 50.0
        episode_fill_rates.append(service_level) 
        
        var_demand = np.var(r_demands)
        episode_bullwhip_ratios.append(np.var(m_orders) / var_demand if var_demand > 0 else 1.0)
        episode_jitters.append(np.mean(ep_step_jitters) if ep_step_jitters else 0.0)
        
        if algo == "comm_mappo":
            sparsity = np.mean(np.abs(ep_all_msgs) < 0.05) if ep_all_msgs else 0.0
            episode_sparsities.append(sparsity)
            episode_shannon.append(calc_shannon_entropy(ep_all_msgs))
            
            if len(ep_ret_msgs) > 1 and np.var(ep_ret_backlogs) > 0:
                mi = calc_mutual_information(ep_ret_msgs, ep_ret_backlogs)
                episode_mis.append(mi)
            else:
                episode_mis.append(0.0)
        else:
            episode_sparsities.append(0.0)
            episode_shannon.append(0.0)
            episode_mis.append(0.0)
            
    return (np.array(costs), np.array(episode_bullwhip_ratios), np.array(episode_jitters), 
            np.array(episode_sparsities), np.array(episode_mis), np.array(episode_shannon),
            np.array(episode_fill_rates), np.array(episode_holding_costs), np.array(episode_backlog_costs), 
            raw_actor_floats)

# ==============================================================================
# 4. MASTER BENCHMARK EXECUTION
# ==============================================================================

if __name__ == "__main__":
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
    print("    ZERO-SHOT TRANSFER: Base weights trained on POISSON")
    print("=======================================================")

    for scenario in scenarios:
        print(f"\n---> Executing Test Scenario: [{scenario.upper()}]")
        results, final_sparsities, final_mis, final_shannon = {}, [], [], []
        
        for k, v in configs.items():
            if v is None:
                print(f"  -> Running {k.upper()} (Baseline)")
                costs, bw_ratios, jitters, sparsities, mis, shannons, fill_rates, hold_c, back_c, raw_floats = run_benchmark(k, v, scenario_type=scenario)
                results[k] = {
                    "costs": costs, "bw": bw_ratios, "jitter": jitters,
                    "fill": fill_rates, "hold": hold_c, "back": back_c
                }
            else:
                abs_path = os.path.join(PROJECT_ROOT, v)
                if os.path.exists(abs_path):
                    print(f"  -> Running {k.upper()} (Found weights: {abs_path})")
                    costs, bw_ratios, jitters, sparsities, mis, shannons, fill_rates, hold_c, back_c, raw_floats = run_benchmark(k, abs_path, scenario_type=scenario)
                    results[k] = {
                        "costs": costs, "bw": bw_ratios, "jitter": jitters,
                        "fill": fill_rates, "hold": hold_c, "back": back_c
                    }
                    
                    if k == "comm_mappo": 
                        final_sparsities = sparsities
                        final_mis = mis
                        final_shannon = shannons
                    
                    if scenario == "step":
                        print(f"    [Diagnostic] {k.upper()} Raw Network Floats (Steps 1-3): {[f'{val:.6f}' for val in raw_floats]}")
                else:
                    print(f"  [ERROR] Skipping {k.upper()} - Cannot find file at {abs_path}")
        
        for k, v in results.items():
            all_scenario_summaries.append({
                "Scenario": scenario.upper(),
                "Algo": k.upper(),
                "Mean Cost": np.mean(v["costs"]),
                "Fill Rate (%)": np.mean(v["fill"]) * 100,           
                "Holding Cost": np.mean(v["hold"]),                  
                "Backlog Cost": np.mean(v["back"]),                  
                "Robustness (CV)": np.std(v["costs"]) / np.mean(v["costs"]) if np.mean(v["costs"]) != 0 else 0,
                "Bullwhip Ratio": np.mean(v["bw"]),
                "Action Volatility": np.mean(v["jitter"]),
                "Sparsity Index": np.mean(final_sparsities) if k == "comm_mappo" else 0.0,
                "Mutual Info (bits)": np.mean(final_mis) if k == "comm_mappo" else 0.0,
                "Shannon Entropy": np.mean(final_shannon) if k == "comm_mappo" else 0.0 
            })
            
        if len(final_mis) > 0:
            print(f"\n=== COMM-MAPPO LATENT SPACE ANALYSIS ({scenario.upper()}) ===")
            print(f"  -> Signal Sparsity Index (Muted ratio): {np.mean(final_sparsities):.2%}")
            print(f"  -> Mutual Information (Message vs Backlog): {np.mean(final_mis):.4f} bits")
            print(f"  -> Shannon Entropy (Vocabulary Complexity): {np.mean(final_shannon):.4f} bits")

        if len(results) >= 2:
            print("\n=== ASSUMPTION TESTING: NORMALITY (Shapiro-Wilk) ===")
            for algo in results.keys():
                try:
                    if np.var(results[algo]["costs"]) == 0:
                        print(f"  {algo.upper()}: Deterministic (Zero Variance) -> NON-NORMAL")
                    else:
                        _, p_norm = stats.shapiro(results[algo]["costs"])
                        print(f"  {algo.upper()}: p-value = {p_norm:.4e} -> {'NON-NORMAL' if p_norm < 0.05 else 'NORMAL'}")
                except Exception:
                    print(f"  {algo.upper()}: Test failed (likely identical values)")

            print("\n=== VARIANCE ANALYSIS: SYSTEMIC VOLATILITY (Ansari-Bradley) ===")
            if "sterman_heuristic" in results and "comm_mappo" in results:
                try:
                    if np.var(results["sterman_heuristic"]["costs"]) == 0 and np.var(results["comm_mappo"]["costs"]) == 0:
                        print("  Sterman vs Comm-MAPPO Variance p-value: N/A (Both are perfectly deterministic)")
                    else:
                        _, p_var = stats.ansari(results["sterman_heuristic"]["costs"], results["comm_mappo"]["costs"])
                        print(f"  Sterman vs Comm-MAPPO Variance p-value: {p_var:.4e}")
                except Exception:
                    print("  Sterman vs Comm-MAPPO Variance p-value: N/A (Test failed due to ties)")

            print("\n=== ADJUSTED PAIRWISE SIGN-RANK (With Holm-Bonferroni Correction) ===")
            raw_p_values = []
            pairs = list(combinations(results.keys(), 2))

            for a, b in pairs:
                try:
                    diff = np.array(results[a]["costs"]) - np.array(results[b]["costs"])
                    if np.all(diff == 0): p_val = 1.0
                    else: p_val = stats.wilcoxon(results[a]["costs"], results[b]["costs"]).pvalue
                except Exception:
                    p_val = 1.0
                raw_p_values.append(p_val)

            n_tests = len(raw_p_values)
            sort_idx = np.argsort(raw_p_values)
            adjusted_p_vals = np.zeros(n_tests)
            for i, idx in enumerate(sort_idx):
                adjusted_p_vals[idx] = min(1.0, raw_p_values[idx] * (n_tests - i))

            for idx, (a, b) in enumerate(pairs):
                print(f"  {a.upper()} vs {b.upper()}: Raw p: {raw_p_values[idx]:.4e} | Holm-Adjusted p: {adjusted_p_vals[idx]:.4e}")

        plt.figure(figsize=(10, 6))
        flat_data = pd.DataFrame([(k.upper(), v_i) for k, v in results.items() for v_i in v["costs"]], columns=['Topology', 'Cost'])
        sns.boxplot(x='Topology', y='Cost', hue='Topology', data=flat_data, palette="viridis", legend=False)
        if "sterman_heuristic" in results:
            plt.axhline(y=np.mean(results["sterman_heuristic"]["costs"]), color='r', linestyle='--', label="Sterman Mean Baseline")
        plt.title(f"Statistical Cost Distributions under Scenario: {scenario.upper()}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"benchmark_{scenario}_comparison.png", dpi=300)
        plt.close()

    master_df = pd.DataFrame(all_scenario_summaries).set_index(["Scenario", "Algo"])
    print("\n=======================================================")
    print("--- FINAL MANUSCRIPT MASTER DATA TABLE ---")
    print("=======================================================")
    print(master_df.round(4).to_string())
    
    master_df.to_csv("master_benchmark_results.csv")
    print("\n-> Saved 'master_benchmark_results.csv' successfully.")
    print("-> All scenario comparison plots (.png) generated successfully.")