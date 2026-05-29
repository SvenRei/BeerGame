import numpy as np

class OrderUpToPolicy:
    def __init__(self, target_inv=25, max_order=100):
        self.target_inv = target_inv
        self.max_order = max_order

    def get_action(self, obs):
        # obs structure: [inventory, backlog, pipeline_t1, pipeline_t2, ...]
        inv = obs[0]
        backlog = obs[1]
        pipeline_total = sum(obs[2:])
        
        # Net Inventory Position
        net_position = inv - backlog + pipeline_total
        
        # Order exactly what is needed to reach target
        order_qty = max(0, self.target_inv - net_position)
        
        # Scale down to [0.0, 1.0] for the environment
        scaled_action = min(order_qty / self.max_order, 1.0)
        return [scaled_action]