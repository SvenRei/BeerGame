# Import the sys module to interact with the Python runtime environment and paths
import sys
# Import the os module to handle dynamic file paths and directory structures
import os
# Import the core PyTorch deep learning library
import torch
# Import NumPy for efficient numerical operations, array handling, and statistics
import numpy as np
# Import Pandas to structure the final results into a tabular DataFrame
import pandas as pd
# Import Matplotlib's pyplot module for generating the boxplot visualizations
import matplotlib.pyplot as plt
# Import Seaborn for advanced, aesthetically pleasing statistical plotting
import seaborn as sns
# Import SciPy's stats module for academic hypothesis testing (Shapiro, Ansari, Wilcoxon)
from scipy import stats
# Import combinations from itertools to generate pairwise statistical test pairs
from itertools import combinations

# ==============================================================================
# 1. PATH SETUP & IMPORTS
# ==============================================================================
# Dynamically locate the absolute path of the root project folder (two levels up)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Append the project root folder to Python's system path so it can find custom modules
sys.path.append(PROJECT_ROOT)

# Import the custom multi-agent Beer Game environment simulator class
from envs.beer_game_env import BeerGameParallelEnv
# Import the MAPPO and COMM-MAPPO Actor architectures for evaluation
from agents.rl.mappo import MAPPOActor, CommMAPPOActor
# Import the QMIX local agent architecture for the new baseline evaluation
from agents.rl.qmix import QMixLocalAgent

# ==============================================================================
# 2. ADVANCED ACADEMIC METRICS (Information Theory & Statistics)
# ==============================================================================

# Define a function to calculate Mutual Information between messages and backlog
def calc_mutual_information(x, y, bins=10):
    # Return 0.0 immediately if the arrays are empty to prevent division errors
    if len(x) == 0 or len(y) == 0: return 0.0
    # Calculate the 2D histogram (joint distribution) of the two variables
    c_xy, _, _ = np.histogram2d(x, y, bins)
    # Normalize the joint distribution to create a probability matrix
    c_xy = c_xy / np.sum(c_xy) 
    # Calculate the marginal probabilities for x and y by summing across axes
    p_x, p_y = np.sum(c_xy, axis=1), np.sum(c_xy, axis=0)
    # Initialize the mutual information accumulator
    mi = 0.0
    # Loop through the grid rows (x bins)
    for i in range(bins):
        # Loop through the grid columns (y bins)
        for j in range(bins):
            # Only calculate log if the joint probability is greater than zero
            if c_xy[i, j] > 0:
                # Add the mutual information component using the standard Shannon formula
                mi += c_xy[i, j] * np.log2(c_xy[i, j] / (p_x[i] * p_y[j]))
    # Return the final Mutual Information score in bits
    return mi

# Define a function to calculate the Shannon Entropy of the latent vocabulary
def calc_shannon_entropy(messages):
    # Return 0.0 immediately if the message array is empty
    if len(messages) == 0: return 0.0
    # Count the unique occurrences of each token in the message array
    _, counts = np.unique(messages, return_counts=True)
    # Convert the raw counts into a probability distribution
    probs = counts / len(messages)
    # Calculate Shannon Entropy using the standard -sum(p * log2(p)) formula
    entropy = -np.sum(probs * np.log2(probs))
    # Return the calculated entropy in bits
    return entropy

# Define the classical Operations Research baseline function
def sterman_heuristic(obs, max_order=100):
    # Calculate the order using a fixed target base-stock of 12 and current net inventory
    net_order = max(0, 4 + 0.5 * (12 - (obs[0] - obs[1])))
    # Clip the order to the environment's maximum limit and return it as a fraction
    return min(1.0, net_order / max_order)

# ==============================================================================
# 3. CORE EVALUATION LOOP
# ==============================================================================

# Define the main function that tests a single algorithm on a single scenario
def run_benchmark(algo, model_path, scenario_type, num_episodes=100):
    # Instantiate the test environment with the requested out-of-distribution scenario
    env = BeerGameParallelEnv({"demand_type": scenario_type, "horizon": 50, "max_order": 100})
    # Initialize the actor variable to None
    actor = None
    
    # Check if the algorithm is a neural network (not the Sterman heuristic)
    if algo != "sterman_heuristic":
        # Extract the local observation dimension size from the environment
        local_dim = env.observation_space("retailer").shape[0]
        
        # Instantiate the correct neural network architecture based on the algorithm name
        if algo == "comm_mappo":
            # Instantiate the dual-headed communicative actor
            actor = CommMAPPOActor(local_dim, 256)
            # Load the trained weights into the actor strictly onto the CPU
            actor.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            # Switch the network to evaluation mode
            actor.eval()
        elif algo in ["mappo", "ippo"]:
            # Instantiate the standard single-headed actor
            actor = MAPPOActor(local_dim, 256)
            # Load the trained weights into the actor strictly onto the CPU
            actor.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            # Switch the network to evaluation mode
            actor.eval()
        elif algo == "qmix":
            # For QMIX, we must instantiate a dictionary of 4 independent tier-specific agents
            actor = {a: QMixLocalAgent(local_dim, 256, 51) for a in env.agents}
            # Loop through all 4 supply chain agents
            for a in env.agents:
                # Construct the specific tier's saved file name dynamically
                agent_path = os.path.join(PROJECT_ROOT, f"qmix_agent_{a}_best.pth")
                # Load the specific tier's weights into its independent brain
                actor[a].load_state_dict(torch.load(agent_path, map_location="cpu", weights_only=True))
                # Switch the independent network to evaluation mode
                actor[a].eval()
    
    # Initialize a list to track total costs across all episodes
    costs = []
    # Initialize a list to track Bullwhip Effect ratios across all episodes
    episode_bullwhip_ratios = []
    # Initialize a list to track action volatility (jitter) across episodes
    episode_jitters = []
    # Initialize a list to track communication sparsity across episodes
    episode_sparsities = []
    # Initialize a list to track Mutual Information across episodes
    episode_mis = []
    # Initialize a list to track Shannon Entropy across episodes
    episode_shannon = []
    # Initialize a list to track Type 1 Service Level (Fill Rate) across episodes
    episode_fill_rates = []
    # Initialize a list to track physical holding costs
    episode_holding_costs = []
    # Initialize a list to track physical backlog costs
    episode_backlog_costs = []
    # Initialize a list to track raw network floats for early diagnostic checks
    raw_actor_floats = [] 
    
    # Begin the evaluation loop for the specified number of episodes (default 100)
    for ep in range(num_episodes):
        # Reset the environment using a fixed testing seed to ensure identical demand waves
        obs, _ = env.reset(seed=2000 + ep) 
        
        # Initialize memory states differently based on the algorithm's GRU architecture
        if algo == "qmix":
            # QMIX uses a 2D memory tensor (Batch, Hidden_Dim)
            hidden = {a: torch.zeros(1, 256) for a in env.agents}
        else:
            # MAPPO/IPPO use a 3D memory tensor (Batch, Sequence, Hidden_Dim)
            hidden = {a: torch.zeros(1, 1, 256) for a in env.agents}
            
        # Initialize the latent message inboxes to pure silence for step 0
        msg = {a: torch.zeros(1, 1) for a in env.agents}
        
        # Reset the episode's total cost accumulator
        ep_cost = 0.0
        # Reset the episode's holding cost accumulator
        ep_holding_cost = 0.0
        # Reset the episode's backlog cost accumulator
        ep_backlog_cost = 0.0
        # Reset the counter tracking how many steps the Retailer had zero backlog
        in_stock_steps = 0  
        
        # Lists to track specific variables needed for episode-level metric calculations
        m_orders, r_demands, ep_ret_backlogs, ep_ret_msgs = [], [], [], []
        # Variable to hold the previous step's actions for calculating jitter
        prev_acts = None
        # Lists for step-level jitter and all network messages
        ep_step_jitters, ep_all_msgs = [], []
        
        # Begin the 50-week physical step loop
        while True:
            # Dictionaries to hold the actions and next messages for this step
            acts, next_msg = {}, {}
            
            # Iterate through each agent in the supply chain sequentially
            for i, a in enumerate(env.agents):
                # Check if testing the classical baseline
                if algo == "sterman_heuristic": 
                    # Calculate and store the deterministic Sterman order
                    acts[a] = [sterman_heuristic(obs[a], env.max_order)]
                else:
                    # Disable gradient tracking for neural network evaluation
                    with torch.no_grad():
                        # Convert the NumPy observation into a PyTorch tensor
                        o_t = torch.tensor(obs[a], dtype=torch.float32).unsqueeze(0)
                        
                        # Branch logic for the Communicative MAPPO network
                        if algo == "comm_mappo":
                            # Pass observation, message, and memory through network
                            dist, dist_comm, next_h = actor(o_t, msg[a], hidden[a])
                            # Deterministically select the most probable communication token
                            comm_idx = torch.argmax(dist_comm.probs, dim=-1)
                            # Define the fixed 3-word vocabulary
                            vocab = torch.tensor([-1.0, 0.0, 1.0])
                            # Map the chosen index to the actual floating-point token
                            comm_val = vocab[comm_idx].view(1, 1)
                            
                            # If not the Manufacturer, pass the message to the upstream partner
                            if i < len(env.agents)-1: next_msg[env.agents[i+1]] = comm_val
                            # Log the generated message for overall entropy calculations
                            ep_all_msgs.append(comm_val.item())
                            
                            # Log specific Retailer data for Mutual Information testing
                            if a == "retailer":
                                # Log the Retailer's broadcasted message
                                ep_ret_msgs.append(comm_val.item())
                                # Log the Retailer's actual physical backlog
                                ep_ret_backlogs.append(obs[a][1])
                                
                        # Branch logic for the QMIX network
                        elif algo == "qmix":
                            # Pass the observation and memory through the agent's independent network
                            q_vals, next_h = actor[a](o_t, hidden[a])
                            # Perform Greedy Evaluation by taking the action index with the highest Q-value
                            action_idx = q_vals.argmax(dim=1).item()
                            # Convert the discrete 21-bin index back into a continuous fraction (0.0 to 1.0)
                            acts[a] = [action_idx / 20.0]
                            # Store the updated memory, stripping the extra sequence dimension
                            hidden[a] = next_h.squeeze(1)
                            
                        # Branch logic for standard IPPO/MAPPO networks
                        else: 
                            # Pass observation and memory through the single-headed network
                            dist, next_h = actor(o_t, hidden[a])
                            
                    # For continuous PPO methods, extract the deterministic mean of the Gaussian output
                    if algo in ["mappo", "ippo", "comm_mappo"]:
                        acts[a] = [dist.mean.item()]
                        # Store the updated 3D memory state
                        hidden[a] = next_h
                    
                    # Capture the raw network output of the Retailer during the first 3 steps for diagnostics
                    if ep == 0 and env.current_step < 3 and a == "retailer" and algo != "qmix":
                        # Append the continuous float output
                        raw_actor_floats.append(acts[a][0])
            
            # Calculate action jitter (volatility) if there is a previous action to compare against
            if prev_acts is not None:
                # Calculate the absolute change in orders for all agents
                step_jitter = np.mean([abs(acts[a][0] - prev_acts[a]) for a in env.agents])
                # Append to the episode tracker
                ep_step_jitters.append(step_jitter)
            # Store current actions for the next step's jitter calculation
            prev_acts = {a: acts[a][0] for a in env.agents}
            
            # Advance the latent communication pipeline
            msg = next_msg
            # Silence the Retailer's inbox as consumers do not broadcast latent signals
            msg["retailer"] = torch.zeros(1, 1) 
            
            # Extract the Manufacturer's action, clip it, scale it, and round it to a physical unit
            scaled_m_order = int(np.round(np.clip(acts["manufacturer"][0], 0.0, 1.0) * env.max_order))
            # Append Manufacturer's order to track the top of the Bullwhip
            m_orders.append(scaled_m_order)
            
            # Calculate the true underlying consumer demand based on the active scenario
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
                
            # Append Retailer's external demand to track the bottom of the Bullwhip
            r_demands.append(true_demand)
            # Execute the continuous physical actions in the environment simulator
            obs, rewards, terms, _, infos = env.step(acts)
            
            # Iterate through all agents to log their operational costs
            for a in env.agents:
                # Extract the true unscaled financial cost from the info dictionary
                local_cost = infos[a]["local_cost"]
                # Accumulate the episode total cost
                ep_cost += local_cost
                
                # Extract the raw inventory and backlog counts
                current_inv = obs[a][0]
                current_backlog = obs[a][1]
                # Calculate and accumulate the specific holding cost
                ep_holding_cost += (current_inv * env.h)
                # Calculate and accumulate the specific backlog cost
                ep_backlog_cost += (current_backlog * env.b)
                
                # Evaluate the Retailer's Service Level
                if a == "retailer":
                    # If the Retailer has 0 backlog, they successfully met all customer demand
                    if current_backlog == 0:
                        in_stock_steps += 1
            
            # Break the physical loop if the horizon (50 weeks) is reached
            if any(terms.values()): break
            
        # Append the final accumulated metrics to the multi-episode lists
        costs.append(ep_cost)
        episode_holding_costs.append(ep_holding_cost)
        episode_backlog_costs.append(ep_backlog_cost)
        
        # Calculate the Type 1 Service Level percentage for the episode
        service_level = in_stock_steps / 50.0
        # Store the calculated Fill Rate
        episode_fill_rates.append(service_level) 
        
        # Calculate the mathematical variance of the external demand
        var_demand = np.var(r_demands)
        # Calculate the classic Bullwhip Effect Ratio (Var(Upstream Orders) / Var(Downstream Demand))
        episode_bullwhip_ratios.append(np.var(m_orders) / var_demand if var_demand > 0 else 1.0)
        # Average the action jitter for the episode
        episode_jitters.append(np.mean(ep_step_jitters) if ep_step_jitters else 0.0)
        
        # Calculate semantic metrics only if evaluating the communicative network
        if algo == "comm_mappo":
            # Calculate how often the network chose to remain silent (token near 0)
            sparsity = np.mean(np.abs(ep_all_msgs) < 0.05) if ep_all_msgs else 0.0
            episode_sparsities.append(sparsity)
            # Calculate the Shannon Entropy of the utilized vocabulary
            episode_shannon.append(calc_shannon_entropy(ep_all_msgs))
            
            # Prevent Mutual Information calculation crash if variance is zero
            if len(ep_ret_msgs) > 1 and np.var(ep_ret_backlogs) > 0:
                # Calculate MI between Retailer's message and physical state
                mi = calc_mutual_information(ep_ret_msgs, ep_ret_backlogs)
                episode_mis.append(mi)
            else:
                # Default MI to zero if assumptions fail
                episode_mis.append(0.0)
        # Zero out semantic metrics for non-communicative baselines to keep DataFrame clean
        else:
            episode_sparsities.append(0.0)
            episode_shannon.append(0.0)
            episode_mis.append(0.0)
            
    # Return all collected arrays for statistical processing
    return (np.array(costs), np.array(episode_bullwhip_ratios), np.array(episode_jitters), 
            np.array(episode_sparsities), np.array(episode_mis), np.array(episode_shannon),
            np.array(episode_fill_rates), np.array(episode_holding_costs), np.array(episode_backlog_costs), 
            raw_actor_floats)

# ==============================================================================
# 4. MASTER BENCHMARK EXECUTION
# ==============================================================================

# Ensure the script executes only when run directly
if __name__ == "__main__":
    # Define the 4 distinct evaluation scenarios
    scenarios = ["step", "poisson", "black_swan", "extreme_chaos"]
    
    # Map algorithm keys to their specific saved weights files
    configs = {
        "sterman_heuristic": None, 
        "ippo": "ippo_best.pth", 
        "mappo": "mappo_best.pth", 
        "qmix": "qmix_agent", # QMIX uses a prefix since it dynamically loads 4 files
        "comm_mappo": "comm_mappo_best.pth"
    }
    # Initialize a master list to hold rows for the final CSV
    all_scenario_summaries = []

    # Print a heavy academic header to the terminal
    print("\n=======================================================")
    print("    LAUNCHING MULTI-SCENARIO ACADEMIC BENCHMARK        ")
    print(f"    PROJECT ROOT: {PROJECT_ROOT}")
    print("    ZERO-SHOT TRANSFER: Base weights trained on POISSON")
    print("=======================================================")

    # Iterate through all environmental scenarios
    for scenario in scenarios:
        # Print the current test scenario boundary
        print(f"\n---> Executing Test Scenario: [{scenario.upper()}]")
        # Initialize dictionary to hold raw arrays for statistical tests
        results = {}
        # Initialize lists for logging semantic metrics
        final_sparsities, final_mis, final_shannon = [], [], []
        
        # Iterate through every algorithm defined in the configs map
        for k, v in configs.items():
            # Check if executing the math-only classical baseline
            if v is None:
                # Print baseline execution
                print(f"  -> Running {k.upper()} (Baseline)")
                # Run evaluation and unpack all metrics
                costs, bw_ratios, jitters, sparsities, mis, shannons, fill_rates, hold_c, back_c, raw_floats = run_benchmark(k, v, scenario_type=scenario)
                # Store the unpacked arrays in the results dictionary for plotting
                results[k] = {
                    "costs": costs, "bw": bw_ratios, "jitter": jitters,
                    "fill": fill_rates, "hold": hold_c, "back": back_c
                }
            else:
                # Ensure QMIX runs even though it doesn't point to a single file
                if k == "qmix":
                    # Note: We rely on the internal logic in run_benchmark to find the 4 QMIX files
                    print(f"  -> Running {k.upper()} (Loading 4 independent tier networks)")
                    # Run evaluation using the algorithm key
                    costs, bw_ratios, jitters, sparsities, mis, shannons, fill_rates, hold_c, back_c, raw_floats = run_benchmark(k, v, scenario_type=scenario)
                    # Store results for plotting
                    results[k] = {
                        "costs": costs, "bw": bw_ratios, "jitter": jitters,
                        "fill": fill_rates, "hold": hold_c, "back": back_c
                    }
                else:
                    # Construct absolute path for single-file PPO methods
                    abs_path = os.path.join(PROJECT_ROOT, v)
                    # Safety check to ensure the `.pth` file actually exists
                    if os.path.exists(abs_path):
                        # Print confirmation
                        print(f"  -> Running {k.upper()} (Found weights: {abs_path})")
                        # Run evaluation using the single file path
                        costs, bw_ratios, jitters, sparsities, mis, shannons, fill_rates, hold_c, back_c, raw_floats = run_benchmark(k, abs_path, scenario_type=scenario)
                        # Store results for plotting
                        results[k] = {
                            "costs": costs, "bw": bw_ratios, "jitter": jitters,
                            "fill": fill_rates, "hold": hold_c, "back": back_c
                        }
                        
                        # Isolate the semantic metrics generated strictly by the communicative agent
                        if k == "comm_mappo": 
                            final_sparsities = sparsities
                            final_mis = mis
                            final_shannon = shannons
                        
                        # Print raw output diagnostics specifically during the step-demand test
                        if scenario == "step" and k != "qmix":
                            print(f"    [Diagnostic] {k.upper()} Raw Network Floats (Steps 1-3): {[f'{val:.6f}' for val in raw_floats]}")
                    else:
                        # Log error if weights are missing and skip algorithm to prevent crashing
                        print(f"  [ERROR] Skipping {k.upper()} - Cannot find file at {abs_path}")
        
        # Compile summary statistics for the DataFrame
        for k, v in results.items():
            # Append a highly-detailed dictionary row representing this scenario+algorithm combination
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
            
        # Print Information Theory diagnostics if communication was active
        if len(final_mis) > 0:
            print(f"\n=== COMM-MAPPO LATENT SPACE ANALYSIS ({scenario.upper()}) ===")
            print(f"  -> Signal Sparsity Index (Muted ratio): {np.mean(final_sparsities):.2%}")
            print(f"  -> Mutual Information (Message vs Backlog): {np.mean(final_mis):.4f} bits")
            print(f"  -> Shannon Entropy (Vocabulary Complexity): {np.mean(final_shannon):.4f} bits")

        # Conduct statistical testing only if multiple algorithms were executed
        if len(results) >= 2:
            print("\n=== ASSUMPTION TESTING: NORMALITY (Shapiro-Wilk) ===")
            # Iterate through all tested algorithms
            for algo in results.keys():
                try:
                    # Check if the policy converged to a deterministic, zero-variance state
                    if np.var(results[algo]["costs"]) == 0:
                        print(f"  {algo.upper()}: Deterministic (Zero Variance) -> NON-NORMAL")
                    else:
                        # Perform the Shapiro-Wilk test for normal distribution
                        _, p_norm = stats.shapiro(results[algo]["costs"])
                        print(f"  {algo.upper()}: p-value = {p_norm:.4e} -> {'NON-NORMAL' if p_norm < 0.05 else 'NORMAL'}")
                except Exception:
                    # Catch cases where the math fails due to identical continuous arrays
                    print(f"  {algo.upper()}: Test failed (likely identical values)")

            print("\n=== VARIANCE ANALYSIS: SYSTEMIC VOLATILITY (Ansari-Bradley) ===")
            # Compare the physical volatility of the baseline against the communicative network
            if "sterman_heuristic" in results and "comm_mappo" in results:
                try:
                    # Reject test if both are perfectly deterministic
                    if np.var(results["sterman_heuristic"]["costs"]) == 0 and np.var(results["comm_mappo"]["costs"]) == 0:
                        print("  Sterman vs Comm-MAPPO Variance p-value: N/A (Both are perfectly deterministic)")
                    else:
                        # Perform Ansari-Bradley test for difference in dispersion (variance)
                        _, p_var = stats.ansari(results["sterman_heuristic"]["costs"], results["comm_mappo"]["costs"])
                        print(f"  Sterman vs Comm-MAPPO Variance p-value: {p_var:.4e}")
                except Exception:
                    # Catch math failures
                    print("  Sterman vs Comm-MAPPO Variance p-value: N/A (Test failed due to ties)")

            print("\n=== ADJUSTED PAIRWISE SIGN-RANK (With Holm-Bonferroni Correction) ===")
            # Initialize array to hold unadjusted p-values
            raw_p_values = []
            # Generate all possible pairing combinations of tested algorithms
            pairs = list(combinations(results.keys(), 2))

            # Iterate through pairs to perform Wilcoxon Signed-Rank Tests
            for a, b in pairs:
                try:
                    # Calculate pair difference
                    diff = np.array(results[a]["costs"]) - np.array(results[b]["costs"])
                    # Default p to 1.0 if the algorithms produced perfectly identical scores
                    if np.all(diff == 0): p_val = 1.0
                    # Perform Wilcoxon test on non-identical distributions
                    else: p_val = stats.wilcoxon(results[a]["costs"], results[b]["costs"]).pvalue
                except Exception:
                    # Default to 1.0 on critical math failures
                    p_val = 1.0
                # Append raw score
                raw_p_values.append(p_val)

            # Extract number of hypotheses tested
            n_tests = len(raw_p_values)
            # Sort p-values from smallest to largest to apply Holm correction
            sort_idx = np.argsort(raw_p_values)
            # Initialize empty array for corrected values
            adjusted_p_vals = np.zeros(n_tests)
            # Apply Holm-Bonferroni step-down mathematical correction
            for i, idx in enumerate(sort_idx):
                adjusted_p_vals[idx] = min(1.0, raw_p_values[idx] * (n_tests - i))

            # Print the rigorously corrected p-values
            for idx, (a, b) in enumerate(pairs):
                print(f"  {a.upper()} vs {b.upper()}: Raw p: {raw_p_values[idx]:.4e} | Holm-Adjusted p: {adjusted_p_vals[idx]:.4e}")

        # --- SEABORN BOXPLOT GENERATION ---
        plt.figure(figsize=(10, 6))
        # Flatten the arrays into a Pandas DataFrame compatible with Seaborn
        flat_data = pd.DataFrame([(k.upper(), v_i) for k, v in results.items() for v_i in v["costs"]], columns=['Topology', 'Cost'])
        # Generate the multi-algorithm boxplot colored by the Viridis palette
        sns.boxplot(x='Topology', y='Cost', hue='Topology', data=flat_data, palette="viridis", legend=False)
        
        # Overlay the classic Sterman mean as a hard red line if it exists
        if "sterman_heuristic" in results:
            plt.axhline(y=np.mean(results["sterman_heuristic"]["costs"]), color='r', linestyle='--', label="Sterman Mean Baseline")
            
        # Annotate and format the plot for publication
        plt.title(f"Statistical Cost Distributions under Scenario: {scenario.upper()}")
        plt.legend()
        plt.tight_layout()
        # Save the high-resolution PNG
        plt.savefig(f"benchmark_{scenario}_comparison.png", dpi=300)
        # Clear the memory buffer
        plt.close()

    # Convert the entire evaluation log into a master Pandas DataFrame indexed by Scenario and Algorithm
    master_df = pd.DataFrame(all_scenario_summaries).set_index(["Scenario", "Algo"])
    
    # Print the master table to the terminal
    print("\n=======================================================")
    print("--- FINAL MANUSCRIPT MASTER DATA TABLE ---")
    print("=======================================================")
    print(master_df.round(4).to_string())
    
    # Export the exact raw numeric data to CSV for copy-pasting into LaTeX or Excel
    master_df.to_csv("master_benchmark_results.csv")
    
    # Print final success confirmations
    print("\n-> Saved 'master_benchmark_results.csv' successfully.")
    print("-> All scenario comparison plots (.png) generated successfully.")