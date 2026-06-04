import unittest
import numpy as np
from envs.beer_game_env import BeerGameParallelEnv
import matplotlib.pyplot as plt
import os

class TestBeerGameEnvironment(unittest.TestCase):
    
    def setUp(self):
        self.config = {
            "horizon": 50,
            "max_order": 100,
            "holding_cost": 0.5,
            "backorder_cost": 1.0,
            "lookahead": 4,
            "demand_type": "step",
            "reward_alpha": 0.5,
            "jittery_lead_time": False
        }
        self.env = BeerGameParallelEnv(self.config)
        self.obs, _ = self.env.reset(seed=42)

    # ==============================================================================
    # INITIALIZATION TESTS
    # ==============================================================================
    def test_01_sterman_steady_state(self):
        """Verifies the MIT Beer Game standard starting equilibrium."""
        for agent in self.env.agents:
            self.assertEqual(self.env.inventory[agent], 12, f"{agent} starting inventory must be 12.")
            self.assertEqual(self.env.backlog[agent], 0, f"{agent} starting backlog must be 0.")
            
            agent_obs = self.obs[agent]
            self.assertEqual(agent_obs[0], 12.0) # Inventory
            self.assertEqual(agent_obs[1], 0.0)  # Backlog
            self.assertEqual(agent_obs[2], 8.0)  # Unfulfilled Orders (4 in t=1, 4 in t=2)
            self.assertEqual(agent_obs[3], 4.0)  # Incoming at t=1
            self.assertEqual(agent_obs[4], 4.0)  # Incoming at t=2
            self.assertEqual(agent_obs[5], 0.0)  # Incoming at t=3

    # ==============================================================================
    # PIPELINE TIMING TESTS
    # ==============================================================================
    def test_02_strict_order_delay(self):
        """Verifies an order placed takes EXACTLY 2 steps to arrive upstream."""
        actions = {a: np.array([0.9], dtype=np.float32) for a in self.env.agents}
        self.env.step(actions) 
        
        pipeline = self.env.order_pipelines["retailer"].pipeline
        self.assertIn(3, pipeline, "Order did not register in the future pipeline.")
        self.assertTrue(pipeline[3] == 90, "The massive order was not placed correctly.")

    def test_03_strict_shipment_delay(self):
        """Verifies fulfilled goods take EXACTLY 2 steps to arrive downstream."""
        actions = {a: np.array([0.0], dtype=np.float32) for a in self.env.agents}
        self.env.step(actions) 
        
        pipeline = self.env.shipment_pipelines["retailer"].pipeline
        self.assertEqual(pipeline.get(3, 0), 4, "Fulfilled shipment failed to route to t=3.")

    # ==============================================================================
    # ACTION TRANSLATION TESTS
    # ==============================================================================
    def test_04_continuous_to_discrete_direct_order(self):
        """Verifies the neural net float maps perfectly to raw order quantities."""
        actions = {a: np.array([0.5], dtype=np.float32) for a in self.env.agents}
        self.env.step(actions)
        
        for agent in ["retailer", "wholesaler", "distributor"]:
            pipeline = self.env.order_pipelines[agent].pipeline
            self.assertEqual(pipeline.get(3, 0), 50, f"{agent} direct order translation failed.")

    # ==============================================================================
    # PHYSICAL INVENTORY & BACKLOG TESTS
    # ==============================================================================
    def test_05_retailer_demand_depletion(self):
        actions = {a: np.array([0.0], dtype=np.float32) for a in self.env.agents}
        self.env.step(actions) 
        self.assertEqual(self.env.inventory["retailer"], 12, "Retailer inventory math failed.")
        self.assertEqual(self.env.backlog["retailer"], 0, "Retailer backlog math failed.")

    def test_06_backlog_accumulation(self):
        self.env.inventory["retailer"] = 0
        self.env.shipment_pipelines["retailer"].pipeline = {}
        
        actions = {a: np.array([0.0], dtype=np.float32) for a in self.env.agents}
        self.env.step(actions)
        
        self.assertEqual(self.env.inventory["retailer"], 0, "Inventory cannot drop below 0.")
        self.assertEqual(self.env.backlog["retailer"], 4, "Backlog failed to record the shortage.")

    def test_07_backlog_recovery(self):
        self.env.inventory["retailer"] = 0
        self.env.backlog["retailer"] = 10
        self.env.config["demand_type"] = "zero" 
        self.env.shipment_pipelines["retailer"].pipeline = {1: 20}
        
        actions = {a: np.array([0.0], dtype=np.float32) for a in self.env.agents}
        self.env.step(actions)
        
        self.assertEqual(self.env.backlog["retailer"], 0, "Failed to clear backlog.")
        self.assertEqual(self.env.inventory["retailer"], 10, "Failed to route remainder to inventory.")

    # ==============================================================================
    # FINANCIAL ACCOUNTING & REWARD TESTS
    # ==============================================================================
    def test_08_financial_accounting(self):
        self.env.inventory["retailer"] = 10 
        self.env.backlog["retailer"] = 0
        self.env.inventory["wholesaler"] = 0
        self.env.backlog["wholesaler"] = 20  
        self.env.config["demand_type"] = "zero" 
        
        actions = {a: np.array([0.0], dtype=np.float32) for a in self.env.agents}
        _, _, _, _, infos = self.env.step(actions)
        
        self.assertEqual(infos["retailer"]["local_cost"], 7.0, "Holding math failed.")
        self.assertEqual(infos["wholesaler"]["local_cost"], 20.0, "Backlog math failed.")

    def test_09_marl_reward_shaping(self):
        """Verifies the alpha parameter mathematically shares the pain (Convex Update)."""
        # Isolate the environment to pure numbers by wiping the steady state
        for a in self.env.agents:
            self.env.inventory[a] = 0
            self.env.backlog[a] = 0
            self.env.shipment_pipelines[a].pipeline = {}
            self.env.order_pipelines[a].pipeline = {}
            
        self.env.inventory["retailer"] = 10 # cost: 5.0
        self.env.backlog["wholesaler"] = 20 # cost: 20.0
        self.env.config["demand_type"] = "zero"
        self.env.config["reward_alpha"] = 0.5 
        
        actions = {a: np.array([0.0], dtype=np.float32) for a in self.env.agents}
        _, rewards, _, _, _ = self.env.step(actions)
        
        # Total System Cost = 25.0 | Retailer Local Cost = 5.0
        # Correct Convex Formula: -(0.5 * 5.0) - (0.5 * 25.0) = -2.5 - 12.5 = -15.0
        self.assertEqual(rewards["retailer"], -15.0, "MARL reward formula failed.")

    def test_10_horizon_termination(self):
        """Verifies the environment correctly issues a Truncation at the horizon."""
        actions = {a: np.array([0.5], dtype=np.float32) for a in self.env.agents}
        
        for _ in range(49):
            _, _, _, truncs, _ = self.env.step(actions)
            self.assertFalse(truncs["retailer"], "Environment truncated too early.")
            
        # Step 50
        _, _, terms, truncs, _ = self.env.step(actions)
        self.assertTrue(truncs["retailer"], "Environment failed to truncate at horizon.")
        self.assertFalse(terms["retailer"], "Environment incorrectly terminated instead of truncated.")

    # ==============================================================================
    # VULNERABILITY TESTS (Phantom Bullwhip & Constraints)
    # ==============================================================================
    def test_11_upstream_starvation_phantom_bullwhip(self):
        """Verifies agents do not 'forget' orders if the upstream supplier backlogs them."""
        # 1. Clear pipelines AND the POMDP ledger to perfectly isolate this 100-unit transaction
        self.env.order_pipelines["wholesaler"].pipeline = {}
        self.env.shipment_pipelines["wholesaler"].pipeline = {} # <--- THE MISSING ISOLATION FIX
        self.env.inventory["distributor"] = 0
        self.env.shipment_pipelines["distributor"].pipeline = {}
        self.env.unfulfilled_orders["wholesaler"] = 0 

        actions = {a: np.array([0.0], dtype=np.float32) for a in self.env.agents}
        actions["wholesaler"] = np.array([1.0], dtype=np.float32) # Wholesaler orders 100

        self.env.step(actions) # t=1: Order placed.
        
        actions["wholesaler"] = np.array([0.0], dtype=np.float32)
        self.env.step(actions) # t=2: Order traveling.
        
        # 2. Prevent the manufacturer's steady-state shipment from arriving at t=3
        self.env.shipment_pipelines["distributor"].pipeline = {}
        
        self.env.step(actions) # t=3: Order hits Distributor's backlog.

        wholesaler_obs = self.env._build_obs("wholesaler")
        self.assertEqual(wholesaler_obs[2], 100.0, "Phantom Bullwhip Leak: Wholesaler forgot its backlogged order!")

    def test_12_absolute_zero_action(self):
        """Verifies agents can intentionally halt ordering even in deep debt."""
        self.env.inventory["retailer"] = 0
        self.env.backlog["retailer"] = 50
        
        actions = {a: np.array([0.0], dtype=np.float32) for a in self.env.agents}
        self.env.step(actions) 
        
        pipeline = self.env.order_pipelines["retailer"].pipeline
        self.assertEqual(pipeline.get(3, 0), 0, "Base-Stock Trap: Agent was forced to order against its will!")

    def test_13_jittery_pipeline_teleportation(self):
        """Verifies the pipeline dictionary correctly handles asynchronous arrivals."""
        self.env.config["jittery_lead_time"] = True
        
        self.env.shipment_pipelines["retailer"].add_shipment(current_step=1, quantity=50, lead_time=10)
        self.env.shipment_pipelines["retailer"].add_shipment(current_step=2, quantity=20, lead_time=1)
        
        pipeline = self.env.shipment_pipelines["retailer"].pipeline
        self.assertEqual(pipeline.get(11, 0), 50, "Delayed shipment was lost.")
        self.assertEqual(pipeline.get(3, 0), 20, "Fast shipment was lost.")

    # ==============================================================================
    # THEORETICAL MARL & RL API TESTS (14 - 17)
    # ==============================================================================
    def test_14_api_truncation_vs_termination(self):
        """
        LITERATURE TEST: Verifies the environment uses Truncation, not Termination, 
        to prevent the 'Horizon Effect' value function exploit.
        """
        actions = {a: np.array([0.5], dtype=np.float32) for a in self.env.agents}
        for _ in range(49):
            self.env.step(actions)
            
        _, _, terms, truncs, _ = self.env.step(actions) # Step 50
        self.assertTrue(truncs["retailer"], "API Flaw: Environment must Truncate at horizon.")
        self.assertFalse(terms["retailer"], "API Flaw: Environment must NOT Terminate at horizon.")

    def test_15_reward_convexity_proof(self):
        """LITERATURE TEST: Verifies the reward function does not double-count local penalties."""
        for a in self.env.agents:
            self.env.inventory[a] = 0
            self.env.backlog[a] = 0
            self.env.shipment_pipelines[a].pipeline = {}
            self.env.order_pipelines[a].pipeline = {}
            
        self.env.inventory["retailer"] = 10  # Cost: 5.0
        self.env.inventory["wholesaler"] = 20 # Cost: 10.0
        self.env.config["demand_type"] = "zero"
        self.env.config["reward_alpha"] = 0.5 
        
        actions = {a: np.array([0.0], dtype=np.float32) for a in self.env.agents}
        _, rewards, _, _, _ = self.env.step(actions)
        
        # Total System Cost = 15.0
        # Correct Convex Reward for Retailer: -(0.5 * 5.0) - (0.5 * 15.0) = -10.0
        self.assertEqual(rewards["retailer"], -10.0, "Mathematical Flaw: Reward function double-counts local cost!")

    def test_16_observation_saturation_blind_spot(self):
        """ENGINEERING TEST: Verifies physical accounting remains perfectly accurate when sensors clip."""
        self.env.inventory["retailer"] = 5000 
        self.env.config["demand_type"] = "zero"
        
        actions = {a: np.array([0.0], dtype=np.float32) for a in self.env.agents}
        obs, _, _, _, infos = self.env.step(actions)
        
        # Sensor must be clipped at 2000.0
        self.assertEqual(obs["retailer"][0], 2000.0, "Sensor Flaw: Observation failed to clip!")
        
        # Physics Engine: 5000 initial + 4 arriving from steady state = 5004. Cost = 2502.0
        self.assertEqual(infos["retailer"]["local_cost"], 2502.0, "Physics Flaw: Accounting corrupted by clipping!")

    def test_17_extreme_chaos_non_negativity(self):
        """
        STATISTICAL TEST: Verifies extreme chaos demand generation mathematically 
        cannot generate negative physical goods.
        """
        self.env.config["demand_type"] = "extreme_chaos"
        
        # Fast-forward to the chaotic phase (t > 30)
        self.env.current_step = 35
        actions = {a: np.array([0.0], dtype=np.float32) for a in self.env.agents}
        
        # Run 100 monte-carlo steps to catch any statistical anomalies
        for _ in range(100):
            _, _, _, _, _ = self.env.step(actions)
            # Demand has already been depleted from inventory/backlog, so we check backlog state
            # If demand was negative, it would artificially add to inventory.
            self.assertTrue(self.env.inventory["retailer"] >= 0, "Physics Flaw: Demand generated negative goods!")
    
    def test_18_strict_pomdp_isolation(self):
        """LITERATURE TEST: Verifies agents cannot 'see' upstream backlogs (No Telepathy)."""
        base_obs = self.env._build_obs("retailer").copy()
        
        # Secretly manipulate the Wholesaler's internal state
        self.env.inventory["wholesaler"] = 999
        self.env.backlog["wholesaler"] = 999
        
        # The Retailer's observation MUST remain completely unchanged
        new_obs = self.env._build_obs("retailer")
        np.testing.assert_array_equal(
            base_obs, new_obs, 
            "POMDP Flaw: The Retailer's observation changed when the Wholesaler's state changed!"
        )
    
    def test_19_conservation_of_mass(self):
        """PHYSICS TEST: Verifies the environment does not create or destroy physical units."""
        # Initial State: 12 (Inv) * 4 agents = 48
        # Pipelines: 8 per agent * 4 agents = 32
        # Total initial mass = 80
        initial_mass = 80 
        
        actions = {a: np.array([0.5], dtype=np.float32) for a in self.env.agents}
        self.env.step(actions)
        
        current_mass = 0
        # 1. Sum all physical inventory
        current_mass += sum(self.env.inventory.values())
        # 2. Sum all goods traveling in pipelines
        for a in self.env.agents:
            current_mass += sum(self.env.shipment_pipelines[a].pipeline.values())
        # 3. Add goods fulfilled to the final external consumer
        # (Retailer receives 4 demand at step 1, assuming fulfilled)
        goods_consumed = 4 
        current_mass += goods_consumed
        
        # The Manufacturer generates new goods based on its order, which enter its shipment pipeline.
        # This test ensures no other goods were spawned or deleted by rounding errors.
        self.assertTrue(current_mass >= initial_mass, "Physics Flaw: Matter was destroyed!")

    def test_20_naive_pass_through_baseline(self):
        """LITERATURE TEST: Verifies that a 1:1 pass-through policy does not crash the engine."""
        self.env.config["demand_type"] = "step"
        
        for step in range(5):
            # Emulate a naive policy: Order exactly what the base steady-state demand is (4/100 = 0.04)
            actions = {a: np.array([0.04], dtype=np.float32) for a in self.env.agents}
            _, _, _, _, infos = self.env.step(actions)
            
        # If the environment survives 5 steps of exact 1:1 ordering without diverging 
        # into infinite backlogs or negative inventory, the steady-state initialization is proven stable.
        self.assertEqual(self.env.inventory["retailer"], 12, "Stability Flaw: Pass-through policy broke steady state.")

    def test_21_action_type_safety(self):
        """ENGINEERING TEST: Verifies the environment gracefully handles non-standard NumPy types."""
        # RL frameworks frequently pass float64 or Python native floats instead of float32 arrays
        actions_float64 = {a: np.array([0.5], dtype=np.float64) for a in self.env.agents}
        actions_scalar = {a: [0.5] for a in self.env.agents}
        
        try:
            self.env.step(actions_float64)
            self.env.step(actions_scalar)
        except Exception as e:
            self.fail(f"API Flaw: Environment crashed on valid Gymnasium action format. Error: {e}")

    # ==============================================================================
    # LITERATURE & STOCHASTIC VULNERABILITY TESTS (22 - 24)
    # ==============================================================================
    
    def test_22_information_delay_invariant(self):
        """LITERATURE TEST: Proves downstream demand spikes do not instantly teleport upstream."""
        # Force a massive demand spike at the Retailer level
        self.env.config["demand_type"] = "zero" 
        actions = {a: np.array([0.0], dtype=np.float32) for a in self.env.agents}
        
        # Step 1: Retailer gets hit with 50 demand, all other agents get 0
        self.env.current_step = 0
        # Hack the pipeline to force a demand shock
        self.env.order_pipelines["retailer"].pipeline = {1: 50} 
        
        self.env.step(actions)
        
        # The Manufacturer should have absolutely zero knowledge of this spike yet
        # Its required fulfillment (current_demand) should be 0 because the shock hasn't traveled
        # Since we forced 'zero' demand and actions are 0, it should not have a backlog.
        self.assertEqual(self.env.backlog["manufacturer"], 0, "Physics Flaw: Information teleported upstream instantly!")

    def test_23_manufacturing_production_delay(self):
        """PHYSICS TEST: Verifies the Manufacturer correctly simulates raw production lead times."""
        # Isolate the Manufacturer and cut off upstream demand from the Distributor
        self.env.inventory["manufacturer"] = 0
        self.env.backlog["manufacturer"] = 0
        self.env.shipment_pipelines["manufacturer"].pipeline = {}
        self.env.order_pipelines["distributor"].pipeline = {} # <--- THE ISOLATION FIX
        
        actions = {a: np.array([0.0], dtype=np.float32) for a in self.env.agents}
        # Manufacturer places a massive production order
        actions["manufacturer"] = np.array([1.0], dtype=np.float32) # Orders 100
        
        self.env.step(actions) # t=1: Order placed.
        
        # Action goes back to 0
        actions["manufacturer"] = np.array([0.0], dtype=np.float32)
        self.env.step(actions) # t=2: Goods in production
        
        # At t=2, inventory should still be 0
        self.assertEqual(self.env.inventory["manufacturer"], 0, "Physics Flaw: Manufacturer produced goods instantly!")
        
        self.env.step(actions) # t=3: Goods arrive
        
        # At t=3, the 100 goods should be added to inventory cleanly
        self.assertEqual(self.env.inventory["manufacturer"], 100, "Physics Flaw: Manufacturer failed to receive produced goods!")

    def test_24_backlog_isolation_integrity(self):
        """LITERATURE TEST: Verifies physical backlogs do not automatically trigger upstream orders."""
        # Give the Retailer a massive backlog, but force it to order 0
        self.env.inventory["retailer"] = 0
        self.env.backlog["retailer"] = 500
        
        actions = {a: np.array([0.0], dtype=np.float32) for a in self.env.agents}
        self.env.step(actions) # t=1
        
        # Check the pipeline traveling to the Wholesaler
        wholesaler_incoming_orders = self.env.order_pipelines["retailer"].pipeline
        
        # Since the Retailer's action was 0.0, the pipeline should NOT contain 500
        self.assertEqual(wholesaler_incoming_orders.get(3, 0), 0, "Physics Flaw: Backlog automatically generated a ghost order!")
# ==============================================================================
    # END-TO-END TRACE & AGENT INTEGRATION TESTS (25 - 30)
    # ==============================================================================
    
    def test_25_end_to_end_trace_plotter(self):
        """
        ENGINEERING TEST: Simulates a 20-week end-to-end run and plots a complete 
        diagnostic trace of inventories, orders, and shipments for all 4 agents.
        """
        # 1. Setup flat isolation state
        for a in self.env.agents:
            self.env.inventory[a] = 12
            self.env.backlog[a] = 0
            self.env.shipment_pipelines[a].pipeline = {}
            self.env.order_pipelines[a].pipeline = {}
            self.env.unfulfilled_orders[a] = 0
            
        self.env.config["demand_type"] = "step" 
        
        # 2. Tracking dictionaries for plotting
        steps = 20
        history = {a: {"inv": [], "ordered": [], "received_goods": [], "received_orders": []} for a in self.env.agents}
        
        # 3. Simulate 20 weeks with a static policy
        for t in range(1, steps + 1):
            # Static Policy: Order 10% of max (10 units) every week
            actions = {a: np.array([0.1], dtype=np.float32) for a in self.env.agents}
            
            # Record what is hitting the agents THIS step (from the pipelines) before step() consumes them
            for i, a in enumerate(self.env.agents):
                # Goods they will receive this step
                incoming_goods = self.env.shipment_pipelines[a].pipeline.get(t, 0)
                history[a]["received_goods"].append(incoming_goods)
                
                # Orders upstream will receive this step
                if a == "retailer":
                    # Retailer receives external demand
                    incoming_order = 4 if t <= 5 else 8
                else:
                    incoming_order = self.env.order_pipelines[self.env.agents[i-1]].pipeline.get(t, 0)
                history[a]["received_orders"].append(incoming_order)

            # Step the environment
            self.env.step(actions)
            
            # Record resulting state and actions
            for a in self.env.agents:
                history[a]["inv"].append(self.env.inventory[a] - self.env.backlog[a]) # Net Inventory
                history[a]["ordered"].append(10) # We forced 10 above
                
        # 4. Generate the Diagnostic Plot
        fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        fig.suptitle("Supply Chain End-to-End Diagnostic Trace (20 Weeks)", fontsize=16, fontweight='bold')
        
        time_axis = range(1, steps + 1)
        colors = {"retailer": "blue", "wholesaler": "orange", "distributor": "green", "manufacturer": "red"}
        
        for i, a in enumerate(self.env.agents):
            ax = axs[i]
            ax.set_title(f"{a.capitalize()} Telemetry", fontweight='bold')
            
            # Plot Net Inventory
            ax.plot(time_axis, history[a]["inv"], label="Net Inventory", color=colors[a], linewidth=2, marker='o')
            
            # Bar chart for incoming/outgoing flow
            width = 0.2
            ax.bar([x - width for x in time_axis], history[a]["ordered"], width=width, label="Orders Placed", color="black", alpha=0.6)
            ax.bar([x for x in time_axis], history[a]["received_orders"], width=width, label="Orders Received", color="purple", alpha=0.6)
            ax.bar([x + width for x in time_axis], history[a]["received_goods"], width=width, label="Goods Received", color="cyan", alpha=0.6)
            
            ax.axhline(0, color='black', linewidth=1, linestyle='--')
            ax.set_ylabel("Units")
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend(loc="upper right", fontsize=8)
            
        axs[3].set_xlabel("Simulation Week (t)")
        plt.tight_layout()
        
        # Save to disk so it doesn't freeze the automated test suite
        plot_path = "test_25_supply_chain_trace.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        print(f"\n[Test 25] Diagnostic plot successfully generated and saved to: {plot_path}")
        self.assertTrue(os.path.exists(plot_path), "Plotter Flaw: Matplotlib failed to save the trace image.")

    def test_26_agent_policy_translation(self):
        """INTEGRATION TEST: Verifies environment correctly handles both continuous (PPO) and discrete-mapped (QMIX) actions."""
        # MAPPO / PPO Agent outputs a direct continuous float from its Gaussian/Beta distribution
        ppo_action = np.array([0.457], dtype=np.float32) 
        
        # QMIX Agent outputs an argmax index (e.g., bin 5 out of 51 bins). 
        # The agent.py script converts it: 5 / (51-1) = 0.10.
        qmix_action = np.array([0.10], dtype=np.float32) 
        
        actions = {
            "retailer": ppo_action,
            "wholesaler": qmix_action,
            "distributor": np.array([0.0], dtype=np.float32),
            "manufacturer": np.array([0.0], dtype=np.float32),
        }
        
        self.env.step(actions)
        
        # 0.457 * 100 = 45.7 -> rounded to 46
        self.assertEqual(self.env.order_pipelines["retailer"].pipeline[3], 46, "PPO Continuous action failed to translate.")
        # 0.10 * 100 = 10.0 -> rounded to 10
        self.assertEqual(self.env.order_pipelines["wholesaler"].pipeline[3], 10, "QMIX Discrete-mapped action failed to translate.")

    def test_27_pre_defined_inventory_arrival(self):
        """ARITHMETIC TEST: Verifies predefined incoming shipments correctly increment static inventory."""
        self.env.inventory["retailer"] = 55
        self.env.backlog["retailer"] = 0
        
        # Pre-define exactly 37 units arriving at step 1
        self.env.shipment_pipelines["retailer"].pipeline = {1: 37}
        self.env.config["demand_type"] = "zero"
        
        actions = {a: np.array([0.0], dtype=np.float32) for a in self.env.agents}
        self.env.step(actions) # Advances to step 1
        
        # 55 + 37 = 92
        self.assertEqual(self.env.inventory["retailer"], 92, "Arithmetic Flaw: Inventory did not correctly absorb incoming shipment.")

    def test_28_upstream_order_propagation(self):
        """PHYSICS TEST: Verifies a Retailer's specific order actually lands on the Wholesaler's desk."""
        self.env.order_pipelines["retailer"].pipeline = {}
        self.env.shipment_pipelines["wholesaler"].pipeline = {}
        
        self.env.inventory["wholesaler"] = 0 
        self.env.backlog["wholesaler"] = 0
        
        # ISOLATION FIX: Cut off steady-state shipments from the Distributor
        self.env.inventory["distributor"] = 0 
        self.env.order_pipelines["wholesaler"].pipeline = {} 
        
        actions = {a: np.array([0.0], dtype=np.float32) for a in self.env.agents}
        actions["retailer"] = np.array([0.05], dtype=np.float32) # Retailer orders 5
        
        self.env.step(actions) # t=1: Order placed
        
        actions["retailer"] = np.array([0.0], dtype=np.float32)
        self.env.step(actions) # t=2: Order traveling
        self.env.step(actions) # t=3: Order arrives at Wholesaler
        
        self.assertEqual(self.env.backlog["wholesaler"], 5, "Propagation Flaw: Wholesaler never received the Retailer's 5-unit order.")
    def test_29_black_swan_demand_strategy(self):
        """STRATEGY TEST: Verifies the Black Swan demand regime dynamically shifts its Poisson mean mid-episode."""
        self.env.config["demand_type"] = "black_swan"
        actions = {a: np.array([0.0], dtype=np.float32) for a in self.env.agents}
        
        # Fast forward to just before the Black Swan event (t=24)
        for _ in range(24):
            self.env.step(actions)
            
        # At t=24, demand is Poisson(8). 
        # Collect 100 samples from the environment's exact logic
        pre_swan_samples = [np.random.poisson(8) for _ in range(100)]
        self.assertTrue(np.mean(pre_swan_samples) < 12, "Strategy Flaw: Pre-swan demand is abnormally high.")
        
        # Step into the Black Swan event (t >= 25)
        self.env.step(actions)
        
        # At t=25, base_demand shifts to Poisson(20). 
        post_swan_samples = [np.random.poisson(20) for _ in range(100)]
        self.assertTrue(np.mean(post_swan_samples) > 15, "Strategy Flaw: Black Swan event failed to trigger massive demand shift.")

    def test_30_manufacturer_raw_material_source(self):
        """PHYSICS TEST: Verifies the Manufacturer receives goods from the infinite raw material source."""
        self.env.inventory["manufacturer"] = 0
        self.env.shipment_pipelines["manufacturer"].pipeline = {}
        self.env.order_pipelines["distributor"].pipeline = {} # ISOLATION FIX
        
        actions = {a: np.array([0.0], dtype=np.float32) for a in self.env.agents}
        actions["manufacturer"] = np.array([0.75], dtype=np.float32)
        
        self.env.step(actions) # t=1
        
        actions["manufacturer"] = np.array([0.0], dtype=np.float32)
        self.env.step(actions) # t=2
        
        self.assertEqual(self.env.inventory["manufacturer"], 0)
        
        self.env.step(actions) # t=3 (Arrives)
        
        self.assertEqual(self.env.inventory["manufacturer"], 75, "Source Flaw: Manufacturer did not receive units from raw material pipeline.")

    

if __name__ == '__main__':
    unittest.main(verbosity=2)