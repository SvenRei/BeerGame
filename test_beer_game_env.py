import unittest
import numpy as np

# Adjust this import based on your exact folder structure
from envs.beer_game_env import BeerGameParallelEnv

class TestBeerGameEnvironment(unittest.TestCase):
    
    def setUp(self):
        """Initializes a fresh, deterministic environment before every test."""
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
            self.assertEqual(agent_obs[2], 4.0)  # Arriving at t=1
            self.assertEqual(agent_obs[3], 4.0)  # Arriving at t=2
            self.assertEqual(agent_obs[4], 0.0)  # Arriving at t=3

    # ==============================================================================
    # PIPELINE TIMING TESTS
    # ==============================================================================
    def test_02_strict_order_delay(self):
        """Verifies an order placed takes EXACTLY 2 steps to arrive upstream."""
        # Force a massive order of 90 (action=0.9) to easily track it
        actions = {a: np.array([0.9], dtype=np.float32) for a in self.env.agents}
        self.env.step(actions) # Time advances to t=1
        
        # At t=1, lead time 2 means the order should exist in the pipeline at t=3
        pipeline = self.env.order_pipelines["retailer"].pipeline
        self.assertIn(3, pipeline, "Order did not register in the future pipeline.")
        self.assertTrue(pipeline[3] > 50, "The massive order was not placed correctly.")

    def test_03_strict_shipment_delay(self):
        """Verifies fulfilled goods take EXACTLY 2 steps to arrive downstream."""
        # Action 0.0 to prevent orders from confusing the test
        actions = {a: np.array([0.0], dtype=np.float32) for a in self.env.agents}
        self.env.step(actions) # Time advances to t=1
        
        # Wholesaler fulfills Retailer's steady-state order of 4 at t=1. 
        # With lead time 2, this shipment must land in Retailer's pipeline at t=3.
        pipeline = self.env.shipment_pipelines["retailer"].pipeline
        self.assertEqual(pipeline.get(3, 0), 4, "Fulfilled shipment failed to route to t=3.")

    # ==============================================================================
    # ACTION TRANSLATION TESTS
    # ==============================================================================
    def test_04_continuous_to_discrete_base_stock(self):
        """Verifies the neural net float maps perfectly to the base-stock formula."""
        actions = {a: np.array([0.5], dtype=np.float32) for a in self.env.agents}
        self.env.step(actions)
        
        for agent in ["retailer", "wholesaler", "distributor"]:
            pipeline = self.env.order_pipelines[agent].pipeline
            # Target = 50. Inv_Pos = 12 (Inv) + 8 (Shipments) + 4 (Orders) = 24.
            # Order = 50 - 24 = 26.
            self.assertEqual(pipeline.get(3, 0), 26, f"{agent} order translation failed.")

    # ==============================================================================
    # PHYSICAL INVENTORY & BACKLOG TESTS
    # ==============================================================================
    def test_05_retailer_demand_depletion(self):
        """Verifies external consumer demand accurately drains Retailer inventory."""
        actions = {a: np.array([0.0], dtype=np.float32) for a in self.env.agents}
        self.env.step(actions) # Time advances to t=1. Step demand is 4.
        
        # Inv(12) + Receive(4) - Demand(4) = 12
        self.assertEqual(self.env.inventory["retailer"], 12, "Retailer inventory math failed.")
        self.assertEqual(self.env.backlog["retailer"], 0, "Retailer backlog math failed.")

    def test_06_backlog_accumulation(self):
        """Verifies the system traps agents in a backlog when demand > inventory."""
        # Hack the environment state: Empty the retailer's inventory
        self.env.inventory["retailer"] = 0
        # Overwrite the pipeline so they receive nothing this turn
        self.env.shipment_pipelines["retailer"].pipeline = {}
        
        actions = {a: np.array([0.0], dtype=np.float32) for a in self.env.agents}
        self.env.step(actions) # Demand of 4 hits an empty warehouse.
        
        self.assertEqual(self.env.inventory["retailer"], 0, "Inventory cannot drop below 0.")
        self.assertEqual(self.env.backlog["retailer"], 4, "Backlog failed to record the shortage.")

    def test_07_backlog_recovery(self):
        """Verifies incoming shipments pay off backlog BEFORE adding to inventory."""
        # Hack state: Retailer is in deep debt (-10 backlog), 0 inventory.
        self.env.inventory["retailer"] = 0
        self.env.backlog["retailer"] = 10
        self.env.config["demand_type"] = "zero" # Turn off demand to isolate recovery math
        
        # Inject a massive shipment of 20 arriving exactly at t=1
        self.env.shipment_pipelines["retailer"].pipeline = {1: 20}
        
        actions = {a: np.array([0.0], dtype=np.float32) for a in self.env.agents}
        self.env.step(actions) # Time advances to 1, shipment arrives.
        
        # The 20 units arrive. 10 pay off the backlog. 10 go to inventory.
        self.assertEqual(self.env.backlog["retailer"], 0, "Failed to clear backlog.")
        self.assertEqual(self.env.inventory["retailer"], 10, "Failed to route remainder to inventory.")

    # ==============================================================================
    # FINANCIAL ACCOUNTING & REWARD TESTS
    # ==============================================================================
    def test_08_financial_accounting(self):
        """Verifies H=0.5 and B=1.0 costs are applied correctly."""
        self.env.inventory["retailer"] = 10 
        self.env.backlog["retailer"] = 0
        self.env.inventory["wholesaler"] = 0
        self.env.backlog["wholesaler"] = 20  
        self.env.config["demand_type"] = "zero" 
        
        actions = {a: np.array([0.0], dtype=np.float32) for a in self.env.agents}
        _, _, _, _, infos = self.env.step(actions)
        
        # Retailer: 10 + 4(Receive) = 14. Cost = 14 * 0.5 = 7.0
        # Wholesaler: 0 + 4(Receive) = 4. Backlog(20) + Demand(4) = 24. Fulfills 4. Backlog = 20. Cost = 20.0
        self.assertEqual(infos["retailer"]["local_cost"], 7.0, "Holding math failed.")
        self.assertEqual(infos["wholesaler"]["local_cost"], 20.0, "Backlog math failed.")

    def test_09_marl_reward_shaping(self):
        """Verifies the alpha parameter mathematically shares the pain."""
        self.env.inventory["retailer"] = 10
        self.env.inventory["wholesaler"] = 0
        self.env.backlog["wholesaler"] = 20
        self.env.config["demand_type"] = "zero"
        self.env.config["reward_alpha"] = 0.5 
        
        actions = {a: np.array([0.0], dtype=np.float32) for a in self.env.agents}
        _, rewards, _, _, _ = self.env.step(actions)
        
        # System Cost = 7 (Retailer) + 20 (Wholesaler) + 6 (Dist) + 6 (Mfg) = 39.0
        # Retailer Reward = -7.0 - (0.5 * 39.0) = -26.5
        self.assertEqual(rewards["retailer"], -26.5, "MARL reward formula failed.")

    # ==============================================================================
    # HORIZON TEST
    # ==============================================================================
    def test_10_horizon_termination(self):
        """Verifies the environment correctly issues done=True at the horizon."""
        actions = {a: np.array([0.5], dtype=np.float32) for a in self.env.agents}
        
        # Step through 49 weeks
        for _ in range(49):
            _, _, terms, _, _ = self.env.step(actions)
            self.assertFalse(terms["retailer"], "Environment terminated too early.")
            
        # Step 50
        _, _, terms, _, _ = self.env.step(actions)
        self.assertTrue(terms["retailer"], "Environment failed to terminate at horizon.")

if __name__ == '__main__':
    unittest.main(verbosity=2)