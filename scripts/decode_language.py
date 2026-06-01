# Import the sys module to manipulate the Python runtime environment and system path
import sys
# Import the os module to handle file paths and directory structures dynamically
import os
# Import PyTorch, the core deep learning framework used for our neural networks
import torch
# Import NumPy for efficient numerical operations and array data handling
import numpy as np
# Import Matplotlib's pyplot module for generating publication-ready visualizations
import matplotlib.pyplot as plt

# ==============================================================================
# 1. PATH SETUP & ARCHITECTURE IMPORTS
# ==============================================================================

# Dynamically locate the absolute path of the root project folder (two levels up from this script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Append the project root folder to Python's system path so it can find custom modules
sys.path.append(PROJECT_ROOT)

# Import the custom multi-agent Beer Game environment simulator class
from envs.beer_game_env import BeerGameParallelEnv
# Import the specifically designed dual-headed actor network for latent communication
from agents.rl.mappo import CommMAPPOActor

# Define the main function that will execute the 4-agent decoding process
def decode_latent_language(model_path, scenario="black_swan"):
    # Print a terminal header line for visual separation
    print(f"\n=======================================================")
    # Print the title of the diagnostic test dynamically including the scenario name
    print(f"    DECODING COMM-MAPPO INFORMATION CASCADE ({scenario.upper()})   ")
    # Print a closing terminal header line
    print(f"=======================================================")
    
    # Initialize the environment dynamically based on the requested scenario parameter
    env = BeerGameParallelEnv({"demand_type": scenario, "horizon": 50, "max_order": 100})
    # Extract the size of the observation vector (how many state variables an agent sees)
    local_dim = env.observation_space("retailer").shape[0]
    
    # Set the computation device strictly to CPU for deterministic, lightweight evaluation
    device = torch.device("cpu")
    # Instantiate the neural network architecture (must match the 256 hidden dimension used during training)
    actor = CommMAPPOActor(local_dim, hidden_dim=256).to(device)
    
    # Safety check: Verify the trained weights file actually exists at the provided file path
    if not os.path.exists(model_path):
        # Print an error message if the .pth file is missing
        print(f"[ERROR] Cannot find weights at {model_path}")
        # Terminate the script early to prevent Python from crashing
        return
        
    # Load the trained neural weights from the .pth file into the PyTorch actor architecture
    # --- FIX: Relaxing weights_only to load complex agent checkpoints ---
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Check if the checkpoint is a dict (state_dict) or a wrapper
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        actor.load_state_dict(checkpoint["model_state_dict"])
    else:
        actor.load_state_dict(checkpoint)
    # Switch the network to evaluation mode (disables exploration noise, dropout, and batch normalization updates)
    actor.eval()

    # ==============================================================================
    # 2. DATA TRACKING INITIALIZATION (FOR ALL 4 AGENTS)
    # ==============================================================================

    # Create an empty list to track the global time step (the shared X-axis)
    steps = []
    # Create an empty list to track the exogenous consumer demand for context
    true_demands = []
    
    # Extract the names of the agents directly from the environment ("retailer", "wholesaler", etc.)
    agent_names = env.agents
    
    # Create a dictionary to hold a separate inventory tracking list for every individual agent
    net_inventories = {a: [] for a in agent_names}
    # --- NEW: Create a dictionary to hold the physical order amounts for every agent ---
    orders = {a: [] for a in agent_names}
    # Create a dictionary to hold a separate latent message tracking list for every individual agent
    messages = {a: [] for a in agent_names}
    
    # Reset the environment to step 0 using a fixed seed (2042) to ensure perfect reproducibility
    obs, _ = env.reset(seed=2042) 
    # Initialize the Recurrent Neural Network (GRU) memory state to zeroes (Size: 256) for all agents
    hidden = {a: torch.zeros(1, 1, 256) for a in agent_names}
    # Initialize the latent communication pipeline with absolute silence (0.0) for step 0
    msg = {a: torch.zeros(1, 1) for a in agent_names}
    
    # ==============================================================================
    # 3. DETERMINISTIC ROLLOUT LOOP
    # ==============================================================================

    # Iterate exactly 50 times to simulate the standard 50-week supply chain horizon
    for step in range(50):
        # Create a dictionary to hold the physical order actions for this specific time step
        acts = {}
        # Create a dictionary to temporarily hold the latent messages intended for the next time step
        next_msg = {}
        
        # --- DYNAMIC SCENARIO DEMAND LOGIC ---
        # Check if the requested scenario is the sudden Black Swan shock
        if scenario == "black_swan":
            # Demand sits peacefully at 8, then violently spikes to 20 at week 25
            demand = 8 if step < 25 else 20
        # Check if the requested scenario is the Extreme Chaos wave
        elif scenario == "extreme_chaos":
            # Baseline demand of 8 for the first 10 weeks
            if step < 10: demand = 8
            # Massive surge to 30 units for the next 10 weeks
            elif step < 20: demand = 30
            # Complete market crash to 0 units for the next 10 weeks
            elif step < 30: demand = 0
            # Recovery stabilization at 15 units for the remainder of the simulation
            else: demand = 15
        # Default fallback demand regime
        else:
            # Baseline deterministic demand of 8 units
            demand = 8
            
        # Append the calculated current demand to our global tracking list
        true_demands.append(demand)
        
        # Loop through every individual agent in the supply chain sequentially
        for i, a in enumerate(agent_names):
            # Temporarily disable PyTorch gradient tracking to save memory and ensure pure inference
            with torch.no_grad():
                # Convert the raw NumPy observation array into a properly shaped PyTorch tensor
                o_t = torch.tensor(obs[a], dtype=torch.float32).unsqueeze(0)
                
                # Pass the observation, incoming message, and previous memory through the neural network
                dist, dist_comm, next_h = actor(o_t, msg[a], hidden[a])
                
                # Extract the deterministic physical action (the exact center/mean of the Gaussian distribution)
                acts[a] = [dist.mean.item()]
                # Update the agent's internal GRU memory state for the next time step
                hidden[a] = next_h
                
                # Identify the index of the most probable word in the categorical communication distribution
                comm_idx = torch.argmax(dist_comm.probs, dim=-1)
                # Define the rigid 3-word vocabulary allowed by our Information Bottleneck constraints
                vocab = torch.tensor([-1.0, 0.0, 1.0])
                # Map the network's chosen index to the actual floating-point token value
                comm_val = vocab[comm_idx].view(1, 1)
                
                # Check if the agent is NOT the Manufacturer (the Manufacturer has no upstream partner to message)
                if i < len(agent_names) - 1:
                    # Place the generated message into the inbox of the immediate upstream partner (i+1)
                    next_msg[agent_names[i+1]] = comm_val
                
                # Extract the raw inventory value from the agent's observation vector (index 0)
                inventory = obs[a][0]
                # Extract the raw backlog value from the agent's observation vector (index 1)
                backlog = obs[a][1]
                # Calculate True Net Inventory (Positive = Overstock holding, Negative = Starving backlog)
                net_inventory = inventory - backlog
                
                # --- NEW: Convert the network's fractional action [0,1] into the true unit order quantity ---
                scaled_order = np.clip(acts[a][0], 0.0, 1.0) * env.max_order
                
                # Append the calculated Net Inventory to this specific agent's tracking list
                net_inventories[a].append(net_inventory)
                # --- NEW: Append the scaled physical order to this specific agent's tracking list ---
                orders[a].append(scaled_order)
                # Append the exact chosen message token to this specific agent's tracking list
                messages[a].append(comm_val.item())

        # Record the current global time step (only needs to be done once per loop, not per agent)
        steps.append(step)
        
        # Advance the communication pipeline, setting the current inbox to the messages generated this step
        msg = next_msg
        # Force the Retailer's inbox to silence, as external end-consumers cannot send latent neural warnings
        msg["retailer"] = torch.zeros(1, 1) 
        
        # Execute the physical environment dynamics (shipping beer, receiving orders, calculating true costs)
        obs, _, terms, _, _ = env.step(acts)
        
        # Check if the environment signals a premature terminal state (end of horizon reached)
        if any(terms.values()): 
            # Break the rollout loop if the episode is officially over
            break

    # ==============================================================================
    # 5. PUBLICATION-READY VISUALIZATION (4x2 GRID)
    # ==============================================================================
    
    # Create a Matplotlib figure with 4 rows (Agents) and 2 columns (Inventory vs. Message), sharing the X-axis
    fig, axes = plt.subplots(4, 2, figsize=(18, 16), sharex=True)
    # Set the grand overarching title for the entire visual figure
    fig.suptitle(f"Information Cascade & Order Volatility Analysis: {scenario.replace('_', ' ').title()}", fontsize=18, fontweight='bold')
    
    # Loop through each agent and its corresponding row in the subplot grid
    for i, a in enumerate(agent_names):
        # Assign the left column axis to the physical inventory plot
        ax_inv = axes[i, 0]
        # --- NEW: Create a secondary Y-axis overlay for the Order Quantities ---
        ax_ord = ax_inv.twinx()
        # Assign the right column axis to the latent message plot
        ax_msg = axes[i, 1]
        
        # --- PHYSICAL INVENTORY & ORDER PLOT (LEFT COLUMN) ---
        # Plot the agent's Net Inventory over time as a blue line with circular markers
        line_inv = ax_inv.plot(steps, net_inventories[a], label=f"Net Inventory", color='blue', linewidth=2, marker='o', markersize=4)
        # --- NEW: Plot the physical order quantity as a dotted orange line ---
        line_ord = ax_ord.plot(steps, orders[a], label=f"Order Quantity", color='darkorange', linestyle=':', linewidth=2)
        # Draw a solid black zero-line to strictly separate overstock (above) from backlog (below)
        ax_inv.axhline(0, color='black', linewidth=1)
        
        # Initialize an empty list to hold the demand line for the legend
        line_dem = []
        # Check if the current agent is the Retailer (the only agent that experiences external demand)
        if a == "retailer":
            # Plot the external consumer demand as a dashed red line for context
            line_dem = ax_inv.plot(steps, true_demands, label="External Demand", color='red', linestyle='--', linewidth=2)
            
        # Label the primary Y-axis (Inventory) in blue
        ax_inv.set_ylabel("Units (Inventory)", color='blue', fontweight='bold')
        # Format the primary Y-axis ticks in blue
        ax_inv.tick_params(axis='y', labelcolor='blue')
        # --- NEW: Label the secondary Y-axis (Orders) in orange ---
        ax_ord.set_ylabel("Units (Order)", color='darkorange', fontweight='bold')
        # --- NEW: Format the secondary Y-axis ticks in orange ---
        ax_ord.tick_params(axis='y', labelcolor='darkorange')
        
        # Combine the lines from both axes into a single consolidated legend
        all_lines = line_inv + line_dem + line_ord
        all_labels = [l.get_label() for l in all_lines]
        # Place the consolidated legend in the upper left corner
        ax_inv.legend(all_lines, all_labels, loc="upper left")
        # Add a faint background grid to make the physical data easier to read
        ax_inv.grid(True, alpha=0.3)
        # Title the left plot for the specific agent
        ax_inv.set_title(f"{a.capitalize()} Physical State")
        
        # --- LATENT MESSAGE PLOT (RIGHT COLUMN) ---
        # Map specific semantic colors to the agent's vocabulary (Red = Token A, Grey = Silence, Green = Token B)
        colors = ['red' if m == -1.0 else 'grey' if m == 0.0 else 'green' for m in messages[a]]
        # Plot the latent messages as a scatter plot of colored dots
        ax_msg.scatter(steps, messages[a], c=colors, s=60, zorder=5)
        # Connect the message dots with a faint gray line to emphasize the temporal sequence
        ax_msg.plot(steps, messages[a], color='black', alpha=0.2, zorder=1)
        
        # Lock the Y-axis strictly to the 3 allowed vocabulary tokens to prevent visual scaling errors
        ax_msg.set_yticks([-1.0, 0.0, 1.0])
        # Apply human-readable semantic labels to the Y-axis ticks
        ax_msg.set_yticklabels(['-1.0 (Token A)', '0.0 (Silence)', '1.0 (Token B)'])
        # Add a title to the message plot identifying which agent is broadcasting
        ax_msg.set_title(f"Message Broadcast by {a.capitalize()}")
        # Add horizontal grid lines to the message plot
        ax_msg.grid(True, alpha=0.3, axis='y')
        
        # --- DYNAMIC BACKGROUND SHADING (FOR ALL PLOTS) ---
        # Check if the scenario is the Black Swan to apply the appropriate shading
        if scenario == "black_swan":
            # Shade the post-shock period (weeks 25-50) with a faint red background on the inventory plot
            ax_inv.axvspan(24.5, 50, color='red', alpha=0.05)
            # Shade the post-shock period with a faint red background on the message plot
            ax_msg.axvspan(24.5, 50, color='red', alpha=0.05)
        # Check if the scenario is Extreme Chaos to apply the dual shading
        elif scenario == "extreme_chaos":
            # Shade the demand surge period (weeks 10-20) with faint red on the inventory plot
            ax_inv.axvspan(9.5, 19.5, color='red', alpha=0.05)
            # Shade the market crash period (weeks 20-30) with faint blue on the inventory plot
            ax_inv.axvspan(19.5, 29.5, color='blue', alpha=0.05)
            # Shade the demand surge period with faint red on the message plot
            ax_msg.axvspan(9.5, 19.5, color='red', alpha=0.05)
            # Shade the market crash period with faint blue on the message plot
            ax_msg.axvspan(19.5, 29.5, color='blue', alpha=0.05)
            
        # Only add the X-axis labels to the bottom-most row of subplots to keep the grid clean
        if i == len(agent_names) - 1:
            # Label the X-axis of the bottom-left inventory plot
            ax_inv.set_xlabel("Time Step (Weeks)")
            # Label the X-axis of the bottom-right message plot
            ax_msg.set_xlabel("Time Step (Weeks)")

    # Automatically adjust padding so titles, labels, and ticks do not overlap
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    # Define the dynamic file name based on the executed scenario
    file_name = f"information_cascade_{scenario}.png"
    # Save the highly-detailed 4x2 grid as a high-resolution PNG image
    plt.savefig(file_name, dpi=300)
    # Print a success confirmation to the terminal
    print(f"-> Decoding complete! Saved full cascade graph to '{file_name}'")
    # Close the matplotlib figure to free up system memory
    plt.close()

# Ensure the script runs automatically when executed directly from the command line
if __name__ == "__main__":
    # Construct the absolute path to the champion communication agent's weight file
    model_path = os.path.join(PROJECT_ROOT, "comm_mappo_best.pth")
    # Loop through both out-of-distribution evaluation scenarios
    for s in ["black_swan", "extreme_chaos"]:
        # Execute the 4-agent decoding function for the current scenario
        decode_latent_language(model_path, scenario=s)