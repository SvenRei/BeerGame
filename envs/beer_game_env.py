import functools
import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Box
from .transit_pipeline import TransitPipeline

class BeerGameParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "beer_game_v0"}

    def __init__(self, config):
        self.agents = ["retailer", "wholesaler", "distributor", "manufacturer"]
        self.possible_agents = self.agents[:]
        
        # Configuration parameters
        self.config = config
        self.horizon = config.get("horizon", 50)
        self.max_order = config.get("max_order", 100)
        self.h = config.get("holding_cost", 0.5)
        self.b = config.get("backorder_cost", 1.0)
        self.lookahead = config.get("lookahead", 4)
        
        # Actions are normalized 0.0 to 1.0, scaled by max_order in step()
        self._action_spaces = {a: Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32) for a in self.agents}
        
        # Obs: [Inventory, Backlog, Incoming_T1, Incoming_T2, Incoming_T3, Incoming_T4]
        obs_dim = 2 + self.lookahead
        self._observation_spaces = {a: Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32) for a in self.agents}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent): 
        return self._observation_spaces[agent]
        
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent): 
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None: 
            np.random.seed(seed)
            
        self.current_step = 0
        self.inventory = {a: 12 for a in self.agents}
        self.backlog = {a: 0 for a in self.agents}
        
        self.order_pipelines = {a: TransitPipeline() for a in self.agents}
        self.shipment_pipelines = {a: TransitPipeline() for a in self.agents}
        
        # Steady-State Initialization (Sterman Standard)
        for a in self.agents:
            for t in range(1, 3):
                # 4 units of beer traveling down, 4 units of orders traveling up
                self.shipment_pipelines[a].add_shipment(0, 4, t)
                self.order_pipelines[a].add_shipment(0, 4, t) 
                
        return {a: self._build_obs(a) for a in self.agents}, {a: {} for a in self.agents}

    def get_global_state(self):
        # Used by MAPPO's Centralized Critic
        global_state = []
        for a in self.agents: 
            global_state.extend(self._build_obs(a))
        return np.array(global_state, dtype=np.float32)

    def _build_obs(self, agent):
        obs = [float(self.inventory[agent]), float(self.backlog[agent])]
        for t in range(1, self.lookahead + 1):
            obs.append(float(self.shipment_pipelines[agent].pipeline.get(self.current_step + t, 0)))
        return np.array(obs, dtype=np.float32)

    def step(self, actions):
        rewards, terminations, truncations, infos = {}, {}, {}, {}
        total_system_cost = 0.0
        orders = {}

        # 1. Place Orders (Information Flow Upstream)
        for agent in self.agents:
            raw_action = float(np.clip(actions[agent][0], 0.0, 1.0))
            orders[agent] = int(np.round(raw_action * self.max_order))
            # Standard information delay is 2 weeks
            self.order_pipelines[agent].add_shipment(self.current_step, orders[agent], lead_time=2)

        # 2. Process Material Flow Downstream (with Stress Test logic)
        demand_type = self.config.get("demand_type", "step")
        
        for i, agent in enumerate(self.agents):
            # Receive incoming beer
            self.inventory[agent] += self.shipment_pipelines[agent].receive_shipment(self.current_step)
            
            # Determine Demand
            if agent == "retailer":
                if demand_type == "step":
                    # Canonical Benchmark: Jumps from 4 to 8 at week 5 (index 4)
                    current_demand = 4 if self.current_step < 4 else 8
                    
                elif demand_type == "black_swan":
                    # The OOD Stress Test Market: Massive spike at week 25
                    base_demand = 8 if self.current_step < 25 else 20
                    current_demand = np.random.poisson(base_demand)
                    
                elif demand_type == "extreme_chaos":
                    # The Pandemic Shock: Unpredictable, structural collapse
                    if self.current_step < 10: 
                        base_demand = 8
                    elif self.current_step < 20: 
                        base_demand = 30  # Panic Buy
                    elif self.current_step < 30: 
                        base_demand = 0   # Market Freeze
                    else: 
                        base_demand = np.random.randint(5, 25) # Wild Whiplash
                    
                    # Prevent poisson function from crashing on zero
                    current_demand = np.random.poisson(base_demand) if base_demand > 0 else 0
                    
                else:
                    # Generic Baseline (Used for Training Domain Randomization)
                    current_demand = np.random.poisson(8)
            else:
                # Upstream demand comes from the downstream agent's order pipeline
                current_demand = self.order_pipelines[self.agents[i - 1]].receive_shipment(self.current_step)
            
            # Fulfill Demand
            total_req = current_demand + self.backlog[agent]
            fulfilled = min(self.inventory[agent], total_req)
            self.inventory[agent] -= fulfilled
            self.backlog[agent] = total_req - fulfilled

            # Ship to Downstream (if not retailer)
            if agent != "retailer":
                # Supply Shock Logic (Kept False for canonical benchmarking)
                if self.config.get("jittery_lead_time", False):
                    delay = np.random.randint(1, 10) 
                else:
                    delay = 2 
                
                self.shipment_pipelines[self.agents[i - 1]].add_shipment(self.current_step, fulfilled, delay)

            # Calculate Costs
            agent_cost = (self.h * self.inventory[agent]) + (self.b * self.backlog[agent])
            total_system_cost += agent_cost
            infos[agent] = {"local_cost": agent_cost}

        self.current_step += 1
        done = self.current_step >= self.horizon
        
        # Assign Cooperative Reward
        for a in self.agents:
            rewards[a] = -total_system_cost
            terminations[a] = done
            truncations[a] = False

        return {a: self._build_obs(a) for a in self.agents}, rewards, terminations, truncations, infos