import numpy as np

class StermanPolicy:
    def __init__(self, expected_demand=8, target_inv=20, theta=0.5, beta=0.2, max_order=100):
        self.expected_demand = expected_demand
        self.target_inv = target_inv
        self.theta = theta # Aggressiveness in fixing inventory
        self.beta = beta   # Attention paid to the pipeline (Under 1.0 causes bullwhip)
        self.max_order = max_order

    def get_action(self, obs):
        inv = obs[0]
        backlog = obs[1]
        pipeline_total = sum(obs[2:])
        
        # Sterman's Anchoring & Adjustment Formula
        adjustment = self.theta * (self.target_inv - (inv - backlog)) - (self.beta * pipeline_total)
        order_qty = max(0, self.expected_demand + adjustment)
        
        scaled_action = min(order_qty / self.max_order, 1.0)
        return [scaled_action]