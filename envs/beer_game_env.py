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
        self.max_order = config.get("max_order", 100) # Keep max_order at 100 for true learning
        self.h = config.get("holding_cost", 0.5)
        self.b = config.get("backorder_cost", 1.0)
        self.lookahead = config.get("lookahead", 4)
        
        # Actions are normalized 0.0 to 1.0
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
                self.shipment_pipelines[a].add_shipment(0, 4, t)
                self.order_pipelines[a].add_shipment(0, 4, t) 
                
        return {a: self._build_obs(a) for a in self.agents}, {a: {} for a in self.agents}

    def get_global_state(self):
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
            
            # --- FIX: Linear Action Mapping ---
            # Standard linear scaling provides a constant, predictable gradient across the entire action spectrum.
            target_base_stock = raw_action * self.max_order
            
            # Compute total items currently in transit (pipeline inventory)
            in_transit = sum(
                qty for step, qty in self.shipment_pipelines[agent].pipeline.items() 
                if step > self.current_step
            )
            
            # Calculate True Inventory Position
            inventory_position = self.inventory[agent] + in_transit - self.backlog[agent]
            
            # The order quantity is the replenishment needed to hit the target
            calculated_order = max(0, int(np.round(target_base_stock - inventory_position)))
            orders[agent] = calculated_order
            
            # The Manufacturer Production Loop
            if agent == "manufacturer":
                self.shipment_pipelines[agent].add_shipment(self.current_step, orders[agent], lead_time=2)
            else:
                self.order_pipelines[agent].add_shipment(self.current_step, orders[agent], lead_time=2)

        # 2. Process Material Flow Downstream
        demand_type = self.config.get("demand_type", "step")
        
        for i, agent in enumerate(self.agents):
            # Receive incoming beer
            self.inventory[agent] += self.shipment_pipelines[agent].receive_shipment(self.current_step)
            
            # Determine Demand
            if agent == "retailer":
                if demand_type == "step":
                    current_demand = 4 if self.current_step < 4 else 8
                elif demand_type == "black_swan":
                    base_demand = 8 if self.current_step < 25 else 20
                    current_demand = np.random.poisson(base_demand)
                elif demand_type == "extreme_chaos":
                    if self.current_step < 10: base_demand = 8
                    elif self.current_step < 20: base_demand = 30
                    elif self.current_step < 30: base_demand = 0   
                    else: base_demand = np.random.randint(5, 25)
                    current_demand = np.random.poisson(base_demand) if base_demand > 0 else 0
                else:
                    current_demand = np.random.poisson(8)
            else:
                current_demand = self.order_pipelines[self.agents[i - 1]].receive_shipment(self.current_step)
            
            # Fulfill Demand
            total_req = current_demand + self.backlog[agent]
            fulfilled = min(self.inventory[agent], total_req)
            self.inventory[agent] -= fulfilled
            self.backlog[agent] = total_req - fulfilled

            # Ship to Downstream (if not retailer)
            if agent != "retailer":
                delay = np.random.randint(1, 10) if self.config.get("jittery_lead_time", False) else 2 
                self.shipment_pipelines[self.agents[i - 1]].add_shipment(self.current_step, fulfilled, delay)

            # Calculate Operational Costs
            agent_cost = (self.h * self.inventory[agent]) + (self.b * self.backlog[agent])
            total_system_cost += agent_cost
            infos[agent] = {"local_cost": agent_cost}

        self.current_step += 1
        done = self.current_step >= self.horizon
        
        # Assign Cooperative Reward Structure
        for a in self.agents:
            rewards[a] = -total_system_cost
            terminations[a] = done
            truncations[a] = False

        return {a: self._build_obs(a) for a in self.agents}, rewards, terminations, truncations, infos