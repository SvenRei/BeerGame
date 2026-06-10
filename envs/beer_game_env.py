import functools
import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Box

def _is_strict_int(v):
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)

def _is_strict_num(v):
    # ENGINEER FIX: Explicitly block NumPy complex types (np.complex64, np.complex128)
    if isinstance(v, bool) or isinstance(v, complex) or np.iscomplexobj(v):
        return False
    return isinstance(v, (int, float, np.number))

class TransitPipeline:
    def __init__(self):
        self.pipeline = {}
        
    def add_shipment(self, current_step, quantity, lead_time):
        if not _is_strict_int(current_step) or current_step < 0:
            raise ValueError(f"current_step must be non-negative int, got {type(current_step)}")
        if not _is_strict_num(quantity) or not np.isfinite(quantity) or quantity < 0:
            raise ValueError(f"Quantity must be finite and non-negative, got {quantity}")
        if not float(quantity).is_integer():
            raise ValueError(f"Quantity must be a discrete whole number, got {quantity}")
        if not _is_strict_int(lead_time) or lead_time <= 0:
            raise ValueError(f"Lead time must be a strictly positive integer, got {lead_time}")
            
        if int(quantity) == 0:
            return 
            
        arr = current_step + int(lead_time)
        self.pipeline[arr] = self.pipeline.get(arr, 0) + int(quantity)
        
    def receive_shipment(self, current_step):
        if not _is_strict_int(current_step) or current_step < 0:
            raise ValueError("current_step must be a non-negative integer")
        return self.pipeline.pop(current_step, 0)

class BeerGameParallelEnv(ParallelEnv):
    metadata = {"name": "beer_game_v0"}

    def __init__(self, config):
        self.possible_agents = ["retailer", "wholesaler", "distributor", "manufacturer"]
        self.agents = self.possible_agents[:]
        
        self._config = config.copy() if config else {}
        self.horizon = self._config.get("horizon", 50)
        self.max_order = self._config.get("max_order", 100) 
        self.h = self._config.get("holding_cost", 0.5)
        self.b = self._config.get("backorder_cost", 1.0)
        self.lookahead = self._config.get("lookahead", 4)
        
        if not _is_strict_int(self.horizon) or self.horizon <= 0: raise ValueError("Horizon must be pos int")
        if not _is_strict_int(self.max_order) or self.max_order <= 0: raise ValueError("Max order must be pos int")
        if not _is_strict_int(self.lookahead) or self.lookahead < 0: raise ValueError("Lookahead must be non-negative int")
            
        if not _is_strict_num(self.h) or not _is_strict_num(self.b): raise ValueError("Costs must be numeric")
        if not np.isfinite(self.h) or not np.isfinite(self.b) or self.h < 0 or self.b < 0: raise ValueError("Costs must be finite positive")
            
        jitter = self._config.get("jittery_lead_time", False)
        if type(jitter) is not bool: raise ValueError("jittery_lead_time must be a strict boolean")
        
        demand_type = self._config.get("demand_type", None)
        if demand_type is None:
            demand_type = "step"
        valid_demands = ["step", "zero", "black_swan", "extreme_chaos", "poisson"]
        if demand_type not in valid_demands:
            raise ValueError(f"Invalid demand_type: {demand_type}")
        self._config["demand_type"] = demand_type
        
        self.np_random = np.random.default_rng()
        self._action_spaces = {a: Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32) for a in self.possible_agents}
        obs_dim = 4 + self.lookahead
        self._observation_spaces = {a: Box(low=-2000.0, high=2000.0, shape=(obs_dim,), dtype=np.float32) for a in self.possible_agents}

    @property
    def config(self):
        """Read-only property preventing runtime mutation."""
        return self._config.copy()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent): return self._observation_spaces[agent]
        
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent): return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None: self.np_random = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self.current_step = 0
        self.inventory = {a: 12 for a in self.possible_agents}
        self.backlog = {a: 0 for a in self.possible_agents}
        
        self.order_pipelines = {a: TransitPipeline() for a in self.possible_agents}
        self.shipment_pipelines = {a: TransitPipeline() for a in self.possible_agents}
        
        for a in self.possible_agents:
            self.shipment_pipelines[a].pipeline = {1: 4, 2: 4}
            self.order_pipelines[a].pipeline = {1: 4} if a == "manufacturer" else {1: 4, 2: 4}
                
        self.unfulfilled_orders = {a: sum(self.shipment_pipelines[a].pipeline.values()) + sum(self.order_pipelines[a].pipeline.values()) for a in self.possible_agents}
        
        self.stochastic_demand_cache = {}
        if self._config.get("demand_type") not in ["step", "zero"]:
            self.stochastic_demand_cache[1] = self._roll_stochastic_demand(1)
            self.stochastic_demand_cache[2] = self._roll_stochastic_demand(2)

        self.current_incoming_order = {a: 0 for a in self.possible_agents}
        # Per-period Type-2 service (fill-rate) accounting, exposed via step()'s info dict.
        self._period_demand = {a: 0 for a in self.possible_agents}
        self._period_demand_met = {a: 0 for a in self.possible_agents}
        return {a: self._build_obs(a) for a in self.agents}, {a: {} for a in self.agents}

    def get_global_state(self):
        """Returns the true unclipped physical global state (inventory, backlog, open orders, pipelines, time)."""
        state = [float(self.current_step)]
        MAX_DELAY = 15
        for a in self.possible_agents:
            state.extend([self.inventory[a], self.backlog[a], self.unfulfilled_orders[a]])
            for t in range(1, MAX_DELAY + 1):
                state.append(self.shipment_pipelines[a].pipeline.get(self.current_step + t, 0))
                state.append(self.order_pipelines[a].pipeline.get(self.current_step + t, 0))
                
        d_type = self._config.get("demand_type")
        if d_type == "step":
            next_d = 4.0 if self.current_step + 1 < 5 else 8.0
        elif d_type == "zero":
            next_d = 0.0
        else:
            if self.current_step + 1 not in self.stochastic_demand_cache:
                 raise RuntimeError(f"Demand cache miss for step {self.current_step + 1}. Observer effect detected.")
            next_d = float(self.stochastic_demand_cache[self.current_step + 1])
        state.append(next_d)
        
        return np.array(state, dtype=np.float32)

    def _roll_stochastic_demand(self, step):
        d_type = self._config.get("demand_type")
        if d_type == "black_swan": return self.np_random.poisson(8 if step < 25 else 20)
        if d_type == "extreme_chaos":
            base = 8 if step < 10 else 30 if step < 20 else 0 if step < 30 else self.np_random.integers(5, 25)
            return self.np_random.poisson(base) if base > 0 else 0
        return self.np_random.poisson(8)

    def _peek_incoming_demand(self, agent, target_step):
        if agent == "retailer":
            d_type = self._config.get("demand_type")
            if d_type == "step": return 4 if target_step < 5 else 8
            if d_type == "zero": return 0
            if target_step not in self.stochastic_demand_cache:
                raise RuntimeError(f"Demand cache miss for step {target_step}. Observer effect detected.")
            return self.stochastic_demand_cache[target_step]
        idx = self.possible_agents.index(agent)
        return self.order_pipelines[self.possible_agents[idx - 1]].pipeline.get(target_step, 0)

    def _build_obs(self, agent):
        next_inc = self._peek_incoming_demand(agent, self.current_step + 1)
        obs = [float(self.inventory[agent]), float(self.backlog[agent]), float(self.unfulfilled_orders[agent]), float(next_inc)]
        for t in range(1, self.lookahead + 1):
            obs.append(float(self.shipment_pipelines[agent].pipeline.get(self.current_step + t, 0)))
        return np.clip(np.array(obs, dtype=np.float32), -2000.0, 2000.0)

    def _validate_actions(self, actions):
        if not isinstance(actions, dict):
            raise ValueError(f"Actions must be a dict, got {type(actions)}")
        if set(actions.keys()) != set(self.agents):
            raise ValueError(f"Action keys mismatch. Expected {self.agents}")
            
        for agent in self.agents:
            act = actions[agent]
            
            if isinstance(act, str) or isinstance(act, bool) or isinstance(act, complex):
                raise ValueError("Action must be a numeric array")
            
            try:
                act_raw = np.array(act)
            except Exception:
                raise ValueError(f"Action for {agent} is invalid.")
                
            if act_raw.dtype == bool or not np.issubdtype(act_raw.dtype, np.number) or np.iscomplexobj(act_raw):
                raise ValueError(f"Action for {agent} must be a pure numeric array. Got dtype {act_raw.dtype}")
                
            act_array = act_raw.astype(float)
            
            if act_array.shape != (1,):
                raise ValueError(f"Action for {agent} must be shape (1,), got {act_array.shape}")
            if not np.isfinite(act_array[0]):
                raise ValueError(f"Action for {agent} must be finite")

    def step(self, actions):
        if not self.agents:
            raise RuntimeError("Environment stepped after done")
            
        self._validate_actions(actions)
            
        self.current_step += 1
        rewards, terminations, truncations, infos = {}, {}, {}, {}
        total_system_cost = 0.0

        # --- PHASE 1: RECEIVE INCOMING GOODS ---
        for agent in self.possible_agents:
            received = self.shipment_pipelines[agent].receive_shipment(self.current_step)
            self.inventory[agent] += received
            self.unfulfilled_orders[agent] -= received

        # --- PHASE 2: DETERMINE & FULFILL DEMAND ---
        for i, agent in enumerate(self.possible_agents):
            if agent == "retailer":
                current_demand = self._peek_incoming_demand(agent, self.current_step)
            else:
                current_demand = self.order_pipelines[self.possible_agents[i - 1]].receive_shipment(self.current_step)
                
            self.current_incoming_order[agent] = current_demand
            
            if agent == "manufacturer":
                requests = self.order_pipelines[agent].receive_shipment(self.current_step)
                if requests > 0:
                    self.shipment_pipelines[agent].add_shipment(self.current_step, requests, lead_time=2)

            backlog_prev = self.backlog[agent]
            total_req = current_demand + backlog_prev
            fulfilled = min(self.inventory[agent], total_req)
            self.inventory[agent] -= fulfilled
            self.backlog[agent] = total_req - fulfilled

            # Type-2 service (fill rate): of THIS period's new demand, how much is met
            # immediately. Existing backlog is honored first; the remainder serves new demand.
            self._period_demand[agent] = current_demand
            self._period_demand_met[agent] = max(0, min(current_demand, fulfilled - backlog_prev))

            if agent != "retailer" and fulfilled > 0:
                delay = self.np_random.integers(1, 10) if self._config.get("jittery_lead_time", False) else 2 
                self.shipment_pipelines[self.possible_agents[i - 1]].add_shipment(self.current_step, fulfilled, delay)

        # --- PHASE 3: PLACE ORDERS ---
        for agent in self.agents:
            raw_action = float(np.clip(np.array(actions[agent], dtype=float)[0], 0.0, 1.0))
            order = int(np.floor(raw_action * self.max_order + 0.5))
            
            if self.current_step < self.horizon:
                self.unfulfilled_orders[agent] += order
                if order > 0:
                    lead_time = 1 if agent == "manufacturer" else 2
                    self.order_pipelines[agent].add_shipment(self.current_step, order, lead_time=lead_time)

        # --- PRE-ROLL FUTURE DEMAND ---
        next_lookahead = self.current_step + 2
        if self._config.get("demand_type") not in ["step", "zero"]:
            if next_lookahead not in self.stochastic_demand_cache:
                self.stochastic_demand_cache[next_lookahead] = self._roll_stochastic_demand(next_lookahead)

        # --- PHASE 4: ACCOUNTING & REWARDS ---
        done = self.current_step >= self.horizon

        for agent in self.agents:
            cost = (self.h * self.inventory[agent]) + (self.b * self.backlog[agent])
            total_system_cost += cost
            infos[agent] = {"local_cost": cost,
                            "demand": self._period_demand[agent],
                            "demand_met": self._period_demand_met[agent]}

        for agent in self.agents:
            rewards[agent] = -total_system_cost
            terminations[agent] = False
            truncations[agent] = done

        obs_dict = {a: self._build_obs(a) for a in self.agents}
        if done: self.agents = []

        return obs_dict, rewards, terminations, truncations, infos