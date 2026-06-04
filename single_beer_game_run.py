import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Adjust this import based on your folder structure
from envs.beer_game_env import BeerGameParallelEnv

def run_diagnostic_episode():
    print("--- Starting Advanced Beer Game Diagnostic Run ---")
    
    config = {
        "horizon": 50,
        "max_order": 100,
        "holding_cost": 0.5,
        "backorder_cost": 1.0,
        "lookahead": 4,
        "demand_type": "step", 
        "reward_alpha": 0.5,
        "jittery_lead_time": False
    }
    
    env = BeerGameParallelEnv(config)
    env.reset(seed=42)
    
    telemetry_data = []

    for t in range(1, config["horizon"] + 1):
        
        incoming_goods = {}
        incoming_demand = {}
        
        # --- A. PEEK AT INCOMING PIPELINES ---
        for i, agent in enumerate(env.agents):
            incoming_goods[agent] = env.shipment_pipelines[agent].pipeline.get(t, 0)
            
            if agent == "retailer":
                incoming_demand[agent] = 4 if t <= 5 else 8
            else:
                downstream_agent = env.agents[i - 1]
                incoming_demand[agent] = env.order_pipelines[downstream_agent].pipeline.get(t, 0)
                
        # --- B. DETERMINE ACTIONS (Naive Pass-Through Policy) ---
        actions = {}
        orders_placed = {}
        for agent in env.agents:
            desired_order = incoming_demand[agent] + env.backlog[agent]
            action_val = np.clip(desired_order / config["max_order"], 0.0, 1.0)
            actions[agent] = np.array([action_val], dtype=np.float32)
            orders_placed[agent] = int(np.round(action_val * config["max_order"]))

        # --- C. EXTRACT EXACT PHYSICAL ACCOUNTING MATH ---
        for agent in env.agents:
            start_inv = env.inventory[agent]
            start_backlog = env.backlog[agent]
            
            # Phase 1 Logic: Receive Goods
            total_available = start_inv + incoming_goods[agent]
            
            # Phase 2 Logic: Fulfill Demand
            total_requested = start_backlog + incoming_demand[agent]
            outgoing_goods = min(total_available, total_requested)
            
            # Phase 3 Logic: End of week balances
            end_inv = total_available - outgoing_goods
            end_backlog = total_requested - outgoing_goods

            telemetry_data.append({
                "Step": t,
                "Agent": agent,
                "Starting_Inventory": start_inv,
                "Incoming_Goods": incoming_goods[agent],
                "Total_Available_Stock": total_available,
                "Starting_Backlog": start_backlog,
                "Incoming_Demand": incoming_demand[agent],
                "Total_Requested_Stock": total_requested,
                "Outgoing_Goods_Shipped": outgoing_goods,
                "Ending_Inventory": end_inv,
                "Ending_Backlog": end_backlog,
                "Net_Inventory": end_inv - end_backlog,
                "Unfulfilled_Ledger": env.unfulfilled_orders[agent],
                "Action_Order_Placed": orders_placed[agent]
            })

        # --- D. STEP THE ENVIRONMENT ---
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        for i in range(4):
            telemetry_data[-(4 - i)]["Step_Cost"] = infos[env.agents[i]]["local_cost"]

    # Export to CSV
    df = pd.DataFrame(telemetry_data)
    csv_path = "beer_game_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Data successfully exported to: {csv_path}")

    # Generate Comprehensive Plot
    print("Generating comprehensive trace plot...")
    generate_plot(df, env.agents)

def generate_plot(df, agents):
    fig, axs = plt.subplots(4, 1, figsize=(16, 22), sharex=True)
    fig.suptitle("Beer Game Engine: Thermodynamic Trace (Dual-Axis)", fontsize=20, fontweight='bold')
    
    colors = {"retailer": "#1f77b4", "wholesaler": "#ff7f0e", "distributor": "#2ca02c", "manufacturer": "#d62728"}
    
    for i, agent in enumerate(agents):
        ax_inv = axs[i]
        ax_flow = ax_inv.twinx() # REVIEWER FIX: Create secondary Y-axis for Flows
        
        agent_data = df[df["Agent"] == agent]
        steps = agent_data["Step"].values
        
        # 1. Plot Stock (Net Inventory) on the Primary Left Axis
        line1 = ax_inv.plot(steps, agent_data["Net_Inventory"].values, 
                            label="Net Inventory (Stock)", color=colors[agent], 
                            linewidth=3.5, marker='o', markersize=6, zorder=3)
        
        # 2. Plot Flows (Goods and Orders) on the Secondary Right Axis
        width = 0.20
        bar1 = ax_flow.bar(steps - 1.5*width, agent_data["Incoming_Goods"].values, 
                           width=width, label="Goods In (From Upstream)", color="cyan", alpha=0.75, zorder=2)
        bar2 = ax_flow.bar(steps - 0.5*width, agent_data["Outgoing_Goods_Shipped"].values, 
                           width=width, label="Goods Out (To Downstream)", color="green", alpha=0.75, zorder=2)
        bar3 = ax_flow.bar(steps + 0.5*width, agent_data["Incoming_Demand"].values, 
                           width=width, label="Demand In (From Downstream)", color="purple", alpha=0.75, zorder=2)
        bar4 = ax_flow.bar(steps + 1.5*width, agent_data["Action_Order_Placed"].values, 
                           width=width, label="Orders Out (To Upstream)", color="black", alpha=0.75, zorder=2)
        
        # Formatting and Labels
        ax_inv.set_title(f"{agent.capitalize()} Telemetry", fontweight='bold', fontsize=14)
        ax_inv.axhline(0, color='black', linewidth=1.5, linestyle='--')
        
        ax_inv.set_ylabel("Net Inventory Level (Left Axis)", color=colors[agent], fontweight='bold')
        ax_flow.set_ylabel("Flow Volume (Right Axis)", color="#333333", fontweight='bold')
        
        # Sync grid to the inventory axis
        ax_inv.grid(True, linestyle=':', alpha=0.7)
        
        # Combine legends from both axes into one unified master legend
        if i == 0:
            lines = line1 + [bar1, bar2, bar3, bar4]
            labels = [l.get_label() for l in lines]
            ax_inv.legend(lines, labels, loc="upper left", fontsize=11, 
                          bbox_to_anchor=(1.08, 1), borderaxespad=0.)

    axs[3].set_xlabel("Simulation Step (Weeks)", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.82, 1]) # Make room for the external legend
    
    plot_path = "comprehensive_beer_game_trace.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot successfully saved to: {plot_path}")
    
if __name__ == "__main__":
    run_diagnostic_episode()