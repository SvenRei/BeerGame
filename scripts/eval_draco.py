"""
eval_draco.py -- evaluation / benchmark loader for DRACO (and DRACO+CRAFT).

DRACO's policy is CONTINUOUS (a base-stock head) and carries recurrent encoder
state + a message buffer across steps, so it does not share the discrete
index_to_fraction protocol of the other baselines. This file provides:

  * DracoPolicy -- conforms to the benchmark policy protocol
                   (reset / act(obs)->{agent: scalar} / is_comm / msg_tokens /
                   ablate), so it can be imported into scripts/benchmark.py OR
                   used by the standalone loop below.
  * a standalone paired evaluation that reports, per scenario, the MEAN cost AND
    the CVaR tail cost (the headline robustness number the mean hides), plus
    bullwhip, Type-1/Type-2 service, order jitter, and a zero-message causal
    ablation with a paired Wilcoxon test.

Eval seeds and scenarios match benchmark.py (SEED_BASE=2000, N=100), so DRACO's
numbers line up with the baseline table. Continuous messages are summarized to a
sign token for the quick MI screen; the proper message decoding is in
analyze_draco.py.

Usage:
  python scripts/eval_draco.py --ckpt weights_draco/run_draco_<id>/draco_checkpoint_best.pt
  python scripts/eval_draco.py --ckpt ... --craft        # if the ckpt is a CRAFT-encoder run
  python scripts/eval_draco.py --ckpt ... --episodes 100 --cvar 0.2
"""
import os
import sys
import argparse
import numpy as np
import torch
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.beer_game_env import BeerGameParallelEnv
from agents.rl.draco import ADJ, ContextEncoder, DRACOActor, DistributionalCritic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AGENTS = ["retailer", "wholesaler", "distributor", "manufacturer"]
SCENARIOS = ["poisson", "black_swan", "extreme_chaos"]
SEED_BASE = 2000
ENV_BASE = {"horizon": 50, "max_order": 100, "lookahead": 4}


def _safe_ratio(num, den):
    return float(num / den) if den and np.isfinite(den) and den != 0 else float("nan")


# ==============================================================================
# DRACO policy (benchmark-protocol compatible)
# ==============================================================================
class DracoPolicy:
    is_comm = True

    def __init__(self, ckpt, env, craft=False, ablate=False, deterministic=True):
        self.env = env
        self.ablate = ablate
        self.deterministic = deterministic
        self.max_order = env.max_order
        self.N = len(AGENTS)
        cfg = ckpt.get("config", {}).get("agent", {})
        self.hidden = cfg.get("hidden_dim", 128)
        self.z_dim = cfg.get("z_dim", 8)
        self.msg_dim = cfg.get("msg_dim", 4)
        local_dim = env.observation_space("retailer").shape[0]
        self.craft = craft
        self.adj = ADJ.to(DEVICE)

        if craft:
            from agents.rl.draco_craft import CRAFTEncoder
            self.encoder = CRAFTEncoder(
                local_dim, self.z_dim, self.hidden,
                n_heads=cfg.get("craft_heads", 4), n_layers=cfg.get("craft_layers", 2),
                max_len=cfg.get("craft_max_len", 64),
            ).to(DEVICE)
        else:
            self.encoder = ContextEncoder(local_dim, self.msg_dim, self.z_dim, self.hidden).to(DEVICE)
        self.encoder.load_state_dict(ckpt["encoder"]); self.encoder.eval()

        self.actors = []
        for sd in ckpt["actors"]:
            a = DRACOActor(local_dim, self.z_dim, self.msg_dim, self.hidden, self.max_order).to(DEVICE)
            a.load_state_dict(sd); a.eval()
            self.actors.append(a)
        self.msg_tokens = {a: 0 for a in AGENTS}
        self.reset()

    def reset(self):
        self.h_enc = torch.zeros(1, self.N, self.hidden, device=DEVICE)
        self.m_buf = torch.zeros(self.N, self.msg_dim, device=DEVICE)
        self.prev_a = torch.zeros(self.N, 1, device=DEVICE)
        self._obs_prefix, self._rew_prefix = [], []   # for CRAFT (action-free) online belief
        self._last_reward = 0.0
        self.msg_tokens = {a: 0 for a in AGENTS}

    @torch.no_grad()
    def act(self, obs):
        o_arr = np.stack([obs[a] for a in AGENTS])
        o_t = torch.tensor(o_arr, dtype=torch.float32, device=DEVICE)               # [N,od]
        m_tilde = self.adj @ self.m_buf                                             # incoming (delayed)
        if self.ablate:
            m_tilde = torch.zeros_like(m_tilde)

        if self.craft:
            r_t = torch.full((self.N, 1), self._last_reward / 100.0, device=DEVICE)
            self._obs_prefix.append(o_t)
            self._rew_prefix.append(r_t)
            obs_seq = torch.stack(self._obs_prefix)                                  # [L,N,od]
            rew_seq = torch.stack(self._rew_prefix)                                  # [L,N,1]
            mu, _, _ = self.encoder.evaluate(obs_seq, rew_seq)
            z = mu[-1]                                                               # [N,z]
        else:
            z, self.h_enc = self.encoder.step(o_t, self.prev_a, m_tilde, self.h_enc)

        S = torch.zeros(self.N, 1, device=DEVICE)
        m_out = torch.zeros(self.N, self.msg_dim, device=DEVICE)
        for i in range(self.N):
            s_mu, s_std, mm, ms = self.actors[i](o_t[i:i+1], z[i:i+1], m_tilde[i:i+1])
            S[i] = s_mu if self.deterministic else torch.distributions.Normal(s_mu, s_std).sample()
            m_out[i] = mm if self.deterministic else torch.distributions.Normal(mm, ms).sample()

        order, _ = DRACOActor.order_from_S(S, o_t, self.max_order)
        frac = (order / self.max_order).clamp(0.0, 1.0)
        self.msg_tokens = {a: int(torch.sign(m_out[i, 0]).item()) for i, a in enumerate(AGENTS)}
        self.m_buf = m_out
        self.prev_a = frac
        return {a: float(frac[i, 0].item()) for i, a in enumerate(AGENTS)}

    def note_reward(self, r):
        self._last_reward = r        # CRAFT belief is action-free but reward-aware


# ==============================================================================
# Episode rollout + metrics (formulas identical to benchmark.run_episode)
# ==============================================================================
def run_episode(policy, env, seed):
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    policy.reset()
    orders = {a: [] for a in AGENTS}
    demand = {a: [] for a in AGENTS}
    inv = {a: [] for a in AGENTS}
    back = {a: [] for a in AGENTS}
    ecost = {a: 0.0 for a in AGENTS}
    fill_d = {a: 0.0 for a in AGENTS}
    fill_m = {a: 0.0 for a in AGENTS}
    tot_cost = 0.0
    while True:
        acts = policy.act(obs)
        for a in AGENTS:
            orders[a].append(int(np.floor(np.clip(acts[a], 0, 1) * env.max_order + 0.5)))
        obs, _r, _t, truncs, infos = env.step({a: [acts[a]] for a in AGENTS})
        step_cost = 0.0
        for a in AGENTS:
            ecost[a] += infos[a]["local_cost"]; tot_cost += infos[a]["local_cost"]
            step_cost += infos[a]["local_cost"]
            inv[a].append(float(obs[a][0])); back[a].append(float(obs[a][1]))
            demand[a].append(float(env.current_incoming_order[a]))
            fill_d[a] += infos[a].get("demand", 0.0); fill_m[a] += infos[a].get("demand_met", 0.0)
        if hasattr(policy, "note_reward"):
            policy.note_reward(-step_cost)
        if any(truncs.values()):
            break
    all_inv = np.concatenate([inv[a] for a in AGENTS])
    all_back = np.concatenate([back[a] for a in AGENTS])
    return {
        "cost": tot_cost,
        "bw_overall": _safe_ratio(np.var(orders["manufacturer"]), np.var(demand["retailer"])),
        "avg_inv": float(all_inv.mean()),
        "avg_back": float(all_back.mean()),
        "ret_service_alpha": float(np.mean(np.array(back["retailer"]) == 0)),
        "fill_beta": _safe_ratio(fill_m["retailer"], fill_d["retailer"]),
        "jitter": float(np.mean([np.mean(np.abs(np.diff(orders[a]))) if len(orders[a]) > 1 else 0.0 for a in AGENTS])),
    }


def cvar(costs, alpha):
    """CVaR_alpha of COST = mean of the worst (highest-cost) alpha fraction."""
    c = np.sort(np.asarray(costs))[::-1]
    k = max(1, int(np.ceil(alpha * len(c))))
    return float(c[:k].mean())


def evaluate(policy, env, episodes):
    return [run_episode(policy, env, SEED_BASE + e) for e in range(episodes)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--craft", action="store_true", help="checkpoint uses the CRAFT (transformer) encoder")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--cvar", type=float, default=0.2, help="tail level for CVaR cost")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    print(f"\nDRACO{'+CRAFT' if args.craft else ''} eval  |  ckpt={os.path.basename(args.ckpt)}  "
          f"|  episodes={args.episodes}  |  CVaR tail={args.cvar:.0%}\n")
    header = f"{'scenario':<15}{'mean cost':>12}{'CVaR cost':>12}{'bullwhip':>11}{'srv-alpha':>11}{'fill-beta':>11}{'jitter':>9}   comm value"
    print(header); print("-" * len(header))

    for scenario in SCENARIOS:
        env = BeerGameParallelEnv({**ENV_BASE, "demand_type": scenario})
        pol = DracoPolicy(ckpt, env, craft=args.craft, ablate=False)
        eps = evaluate(pol, env, args.episodes)
        costs = np.array([e["cost"] for e in eps])

        pol_abl = DracoPolicy(ckpt, env, craft=args.craft, ablate=True)
        abl = evaluate(pol_abl, env, args.episodes)
        c_abl = np.array([e["cost"] for e in abl])
        comm_value = float(np.mean((c_abl - costs) / np.where(costs == 0, np.nan, costs)) * 100.0)
        try:
            p = float(stats.wilcoxon(c_abl, costs).pvalue) if not np.all(c_abl - costs == 0) else 1.0
        except Exception:
            p = float("nan")

        print(f"{scenario:<15}{costs.mean():>12.1f}{cvar(costs, args.cvar):>12.1f}"
              f"{np.nanmean([e['bw_overall'] for e in eps]):>11.2f}"
              f"{np.mean([e['ret_service_alpha'] for e in eps]):>11.2f}"
              f"{np.nanmean([e['fill_beta'] for e in eps]):>11.2f}"
              f"{np.mean([e['jitter'] for e in eps]):>9.1f}"
              f"   {comm_value:+.1f}%  (p={p:.1e})")
    print("\ncomm value = % cost change when neighbour messages are zeroed (paired, same seeds);"
          "\n             positive => communication is actively helping.\n")


if __name__ == "__main__":
    main()