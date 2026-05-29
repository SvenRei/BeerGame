import sys, os, torch, numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.beer_game_env import BeerGameParallelEnv
from agents.rl.mappo import MAPPOActor, CommMAPPOActor

def evaluate_simple(algo, model_path, num_episodes=10):
    env = BeerGameParallelEnv({"demand_type": "step"})
    local_dim = env.observation_space("retailer").shape[0]
    actor = CommMAPPOActor(local_dim, 128) if algo == "comm_mappo" else MAPPOActor(local_dim, 128)
    actor.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    actor.eval()
    
    total_costs = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        hidden, msg, ep_cost = {a: torch.zeros(1, 1, 128) for a in env.agents}, {a: torch.zeros(1, 1) for a in env.agents}, 0
        while True:
            acts = {}
            for i, a in enumerate(env.agents):
                o_t = torch.tensor(obs[a], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    if algo == "comm_mappo":
                        dist, comm, next_h = actor(o_t, msg[a], hidden[a])
                        if i < len(env.agents)-1: msg[env.agents[i+1]] = comm
                    else:
                        dist, next_h = actor(o_t, hidden[a])
                acts[a] = [dist.sample().item()]
                hidden[a] = next_h
            obs, rewards, terms, _, _ = env.step(acts)
            ep_cost -= rewards["retailer"]
            if any(terms.values()): break
        total_costs.append(ep_cost)
    print(f"Algorithm: {algo.upper()} | Avg Cost: {np.mean(total_costs):.2f}")

if __name__ == "__main__":
    evaluate_simple("comm_mappo", "comm_mappo_best.pth")


def verify_canonical_parameters(env):
    """
    Enforces strict adherence to standard Beer Game benchmarks.
    Prevents accidental evaluation on non-canonical configurations.
    """
    print("-> Verifying Canonical Environment Parameters...")
    
    # 1. Horizon Check
    assert env.horizon in [50, 100], f"CRITICAL: Horizon is {env.horizon}. Must be 50 or 100."
    
    # 2. Cost Coefficient Check
    assert env.h == 0.5, f"CRITICAL: Holding cost is {env.h}. Must be 0.5."
    assert env.b == 1.0, f"CRITICAL: Backorder cost is {env.b}. Must be 1.0."
    
    # 3. Information Constraint Check
    # Ensure lookahead doesn't accidentally give agents full global vision
    assert env.lookahead == 4, f"CRITICAL: Lookahead is {env.lookahead}. Standard is 4."
    
    # 4. Demand Type Check
    assert env.config.get("demand_type") == "step", "CRITICAL: Evaluation must use 'step' demand."
    assert env.config.get("jittery_lead_time") is False, "CRITICAL: Evaluation must use fixed lead times."

    print("-> Environment is mathematically canonical. Cleared for benchmarking.\n")