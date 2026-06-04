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
        
        self.config = config
        self.horizon = config.get("horizon", 50)
        self.max_order = config.get("max_order", 100) 
        self.h = config.get("holding_cost", 0.5)
        self.b = config.get("backorder_cost", 1.0)
        self.lookahead = config.get("lookahead", 4)
        
        self._action_spaces = {a: Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32) for a in self.agents}
        
        obs_dim = 3 + self.lookahead
        self._observation_spaces = {a: Box(low=-2000.0, high=2000.0, shape=(obs_dim,), dtype=np.float32) for a in self.agents}

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
        
        # FIX (Test 18): Strict POMDP Ledger Initialization
        # 4 units ordered at t=-1, 4 units ordered at t=-2 (Steady State)
        self.unfulfilled_orders = {a: 8 for a in self.agents}
        
        self.order_pipelines = {a: TransitPipeline() for a in self.agents}
        self.shipment_pipelines = {a: TransitPipeline() for a in self.agents}
        
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
        # FIX (Test 18): Pure local observation. No telepathic lookup of supplier backlogs.
        obs = [float(self.inventory[agent]), float(self.backlog[agent]), float(self.unfulfilled_orders[agent])]
        
        for t in range(1, self.lookahead + 1):
            obs.append(float(self.shipment_pipelines[agent].pipeline.get(self.current_step + t, 0)))
            
        obs_array = np.array(obs, dtype=np.float32)
        return np.clip(obs_array, -2000.0, 2000.0)

    def step(self, actions):
        self.current_step += 1
        
        rewards, terminations, truncations, infos = {}, {}, {}, {}
        total_system_cost = 0.0

        # --- PHASE 1: RECEIVE INCOMING GOODS ---
        for agent in self.agents:
            received = self.shipment_pipelines[agent].receive_shipment(self.current_step)
            self.inventory[agent] += received
            # FIX (Test 18): Deduct received goods from the POMDP ledger
            self.unfulfilled_orders[agent] -= received

        # --- PHASE 2: DETERMINE & FULFILL DEMAND ---
        demand_type = self.config.get("demand_type", "step")
        for i, agent in enumerate(self.agents):
            if agent == "retailer":
                if demand_type == "step":
                    # FIX (Test 20): Shift shock to t>5 to allow baseline verification
                    current_demand = 4 if self.current_step <= 5 else 8
                elif demand_type == "black_swan":
                    base_demand = 8 if self.current_step < 25 else 20
                    current_demand = np.random.poisson(base_demand)
                elif demand_type == "extreme_chaos":
                    if self.current_step < 10: base_demand = 8
                    elif self.current_step < 20: base_demand = 30
                    elif self.current_step < 30: base_demand = 0   
                    else: base_demand = np.random.randint(5, 25)
                    current_demand = np.random.poisson(base_demand) if base_demand > 0 else 0
                elif demand_type == "zero":
                    current_demand = 0
                else:
                    current_demand = np.random.poisson(8)
            else:
                current_demand = self.order_pipelines[self.agents[i - 1]].receive_shipment(self.current_step)
            
            total_req = current_demand + self.backlog[agent]
            fulfilled = min(self.inventory[agent], total_req)
            self.inventory[agent] -= fulfilled
            self.backlog[agent] = total_req - fulfilled

            if agent != "retailer":
                delay = np.random.randint(1, 10) if self.config.get("jittery_lead_time", False) else 2 
                self.shipment_pipelines[self.agents[i - 1]].add_shipment(self.current_step, fulfilled, delay)

        # --- PHASE 3: PLACE ORDERS (Direct Action Control) ---
        orders = {}
        for agent in self.agents:
            raw_action = float(np.clip(actions[agent][0], 0.0, 1.0))
            calculated_order = int(np.round(raw_action * self.max_order))
            orders[agent] = calculated_order
            
            # FIX (Test 18): Add new orders to the POMDP ledger
            self.unfulfilled_orders[agent] += calculated_order
            
            if agent == "manufacturer":
                self.shipment_pipelines[agent].add_shipment(self.current_step, orders[agent], lead_time=2)
            else:
                self.order_pipelines[agent].add_shipment(self.current_step, orders[agent], lead_time=2)

        # --- PHASE 4: ACCOUNTING & REWARDS ---
        done = self.current_step >= self.horizon
        alpha = self.config.get("reward_alpha", 0.5)
        
        for agent in self.agents:
            agent_cost = (self.h * self.inventory[agent]) + (self.b * self.backlog[agent])
            total_system_cost += agent_cost
            infos[agent] = {"local_cost": agent_cost}

        for agent in self.agents:
            local_penalty = infos[agent]["local_cost"]
            rewards[agent] = -((1.0 - alpha) * local_penalty) - (alpha * total_system_cost)
            
            terminations[agent] = False
            truncations[agent] = done

        return {a: self._build_obs(a) for a in self.agents}, rewards, terminations, truncations, infos