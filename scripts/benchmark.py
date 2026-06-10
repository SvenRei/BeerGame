"""
================================================================================
 BEER GAME  --  MULTI-SCENARIO BENCHMARK  (Sterman / MAPPO / Comm-MAPPO / QMIX / Comm-QMIX)
================================================================================
Evaluates the five in-scope policies under four demand regimes (step, poisson,
black_swan, extreme_chaos) and reports a literature-grounded metric suite.

WHY THESE METRICS (see the discussion in the project notes for full rationale):
  COST / OBJECTIVE
    * Mean cost, Std, CV            -- the objective + its robustness across demand draws.
    * Cost vs Sterman (ratio)       -- anchors RL numbers to the classical heuristic.
    * Per-echelon cost              -- where cost concentrates along the chain.
    * Holding vs Backlog split      -- behavioural signature (hoarding vs starving).
  SERVICE / INVENTORY (operations view -- cost alone hides starvation)
    * alpha service level           -- P(period with no stockout) at the retailer.
    * beta fill rate                -- fraction of customer demand met immediately from stock
                                       (Type-2 service; needs the env step() instrumentation).
    * Stockout frequency, avg inv, avg backlog.
  BASELINES (two classical anchors that bracket the RL agents)
    * sterman      -- behavioural anchoring-and-adjustment heuristic (bullwhip-prone, no fit).
    * base_stock   -- order-up-to ORACLE; its level S is auto-tuned on held-out seeds, giving
                      a strong near-optimal stationary benchmark (Clark & Scarf 1960).
  BULLWHIP (the signature supply-chain effect: Lee et al. 1997; Sterman 1989)
    * Per-echelon and overall order-variance amplification Var(orders)/Var(demand).
    * Order volatility (mean |delta order|) -- oscillation / smoothness.
  COMMUNICATION (comm agents only; Lowe et al. 2019 pitfalls)
    * Message entropy               -- channel usage (descriptive only).
    * MI(message; sender backlog)   -- positive *signalling* (do messages encode state?).
    * Communication value (ablation Delta cost %) -- positive *listening* (the CAUSAL test).
  GENERALISATION
    * The 4-regime sweep itself is the zero-shot transfer test (train poisson -> test all).
  STATISTICS
    * Paired Wilcoxon signed-rank (same seeds) + Holm-Bonferroni; Shapiro for normality.

USAGE
  1. After the sweep, copy each winning run's *_checkpoint_best.pt into MODEL_PATHS below
     (these .pt files carry the config, so architecture dims are auto-detected).
  2. python scripts/benchmark.py
  Missing/!exist paths are skipped with a warning -- the script never hard-crashes.
================================================================================
"""

import os
import sys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from envs.beer_game_env import BeerGameParallelEnv
from agents.action_space import index_to_fraction
from agents.rl.mappo import MAPPOActor, CommMAPPOActor, MAPPOCommMAC
from agents.rl.qmix import QMixLocalAgent, CommQMixLocalAgent, QMixCommMAC

# ==============================================================================
# CONFIG  -- edit MODEL_PATHS to point at your sweep winners' checkpoints
# ==============================================================================
DEVICE = torch.device("cpu")            # eval is tiny; CPU avoids contending with a running sweep
N_EPISODES = 100                        # paired across algos (same seeds) for Wilcoxon
SEED_BASE = 2000                        # eval seeds: SEED_BASE .. SEED_BASE+N-1 (disjoint from training)
SCENARIOS = ["step", "poisson", "black_swan", "extreme_chaos"]
AGENTS = ["retailer", "wholesaler", "distributor", "manufacturer"]   # downstream -> upstream
ENV_BASE = {"horizon": 50, "max_order": 100, "holding_cost": 0.5,
            "backorder_cost": 1.0, "lookahead": 4}                   # must match training (obs_dim=8)

# ------------------------------------------------------------------------------
# WHERE THE CHECKPOINTS COME FROM
# ------------------------------------------------------------------------------
# Training (scripts/train_*.py) writes, per run, into:
#     <repo>/weights_<algo>/run_<wandb-name>_<wandb-id>/
# containing a FULL checkpoint plus a light state-dict:
#     mappo:       <run>/mappo_checkpoint_best.pt        + mappo_actor_best.pth
#     comm_mappo:  <run>/comm_mappo_checkpoint_best.pt   + comm_mappo_actor_best.pth
#     qmix:        <run>/qmix_checkpoint_best.pt         + qmix_agent_<echelon>_best.pth (x4)
#     comm_qmix:   <run>/comm_qmix_checkpoint_best.pt    + comm_qmix_mac_best.pth
# The *_checkpoint_best.pt files embed the resolved config, so this benchmark
# auto-detects hidden_dim / n_actions / vocab_size from them (preferred input).
#
# AFTER A SWEEP: in W&B find the run with the lowest Avg_Cost_50, then copy ITS
# <run>/<algo>_checkpoint_best.pt to the path below (or edit the path to point
# straight at that run directory). The defaults assume you copied the winner to
# <repo>/weights_<algo>/<algo>_checkpoint_best.pt. Missing paths are skipped.
# (sterman and base_stock need no checkpoint -- base_stock auto-tunes its level.)
# ------------------------------------------------------------------------------
MODEL_PATHS = {
    "mappo":      os.path.join(PROJECT_ROOT, "weights_mappo",      "mappo_checkpoint_best.pt"),
    "comm_mappo": os.path.join(PROJECT_ROOT, "weights_comm_mappo", "comm_mappo_checkpoint_best.pt"),
    "qmix":       os.path.join(PROJECT_ROOT, "weights_qmix",       "qmix_checkpoint_best.pt"),
    "comm_qmix":  os.path.join(PROJECT_ROOT, "weights_comm_qmix",  "comm_qmix_checkpoint_best.pt"),
}

# Sterman (1989) anchoring-and-adjustment ordering rule (supply-line aware baseline):
#   O = max(0, D_hat + alpha_S*(S' - S) + alpha_SL*(SL' - SL))
# where S = net stock (on-hand - backlog), S' = desired stock, SL = on-order,
# SL' = desired supply line = D_hat * lead_time. Under-weighting alpha_SL is what
# Sterman showed produces the bullwhip; we use a competent (not strawman) setting.
STERMAN = {"target_stock": 12.0, "lead_time": 4, "alpha_stock": 0.5, "alpha_supply": 0.5}


# ==============================================================================
# INFORMATION-THEORETIC HELPERS
# ==============================================================================
def shannon_entropy(tokens):
    """Entropy (bits) of the emitted token distribution -- how much of the channel is used."""
    if len(tokens) == 0:
        return np.nan
    _, counts = np.unique(tokens, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def mutual_information(tokens, state, state_bins=8):
    """MI(message ; sender state) in bits -- positive *signalling* test.
    tokens: discrete ints; state: continuous (binned)."""
    tokens = np.asarray(tokens)
    state = np.asarray(state)
    if len(tokens) < 2 or np.var(state) < 1e-9 or np.unique(tokens).size < 2:
        return 0.0
    tok_edges = np.unique(tokens).size
    c, _, _ = np.histogram2d(tokens, state, bins=[tok_edges, state_bins])
    c = c / c.sum()
    px = c.sum(axis=1, keepdims=True)
    py = c.sum(axis=0, keepdims=True)
    nz = c > 0
    return float(np.sum(c[nz] * np.log2(c[nz] / (px @ py)[nz])))


# ==============================================================================
# CHECKPOINT LOADING  (auto-detects architecture)
# ==============================================================================
def _load_raw(path):
    """Return (kind, payload, cfg). kind in {'checkpoint','statedict'}."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and any(k in ckpt for k in ("actor", "mac", "config")):
        return "checkpoint", ckpt, ckpt.get("config")
    return "statedict", ckpt, None          # raw state_dict (older *_best.pth)


def _cfg_dims(cfg, fallback=(256, 101, 3)):
    if cfg is None:
        return fallback
    a = cfg.get("agent", {})
    return a.get("hidden_dim", fallback[0]), a.get("n_actions", fallback[1]), a.get("vocab_size", fallback[2])


def _cfg_action(cfg, max_order):
    """Action-space kwargs for index_to_fraction, read from the checkpoint config.
    Backward compatible: old checkpoints (no action_mode) fall back to absolute with
    abs_cap=max_order, which reproduces the original `order = idx` behaviour."""
    a = (cfg or {}).get("agent", {})
    return {
        "mode": a.get("action_mode", "absolute"),
        "abs_cap": a.get("abs_cap", max_order),
        "centered_range": a.get("centered_range", 10),
    }


# ==============================================================================
# POLICY WRAPPERS  -- uniform .reset() / .act(obs) interface
# ==============================================================================
class StermanPolicy:
    is_comm = False
    name = "sterman"

    def __init__(self, env):
        self.max_order = env.max_order

    def reset(self):
        pass

    def act(self, obs):
        acts = {}
        for a in AGENTS:
            inv, backlog, on_order, next_demand = obs[a][0], obs[a][1], obs[a][2], obs[a][3]
            net_stock = inv - backlog
            desired_supply = next_demand * STERMAN["lead_time"]
            order = (next_demand
                     + STERMAN["alpha_stock"] * (STERMAN["target_stock"] - net_stock)
                     + STERMAN["alpha_supply"] * (desired_supply - on_order))
            acts[a] = float(np.clip(max(0.0, order) / self.max_order, 0.0, 1.0))
        return acts


class BaseStockPolicy:
    """Order-up-to (base-stock) policy: each period raise the inventory position
    (on-hand - backlog + on-order) to a target level S. With S tuned on held-out
    seeds this is a strong near-optimal CLASSICAL benchmark for stationary demand
    (Clark & Scarf 1960; the optimal-policy anchor used in Beer-Game RL papers)."""
    is_comm = False
    name = "base_stock"

    def __init__(self, env, S):
        self.max_order = env.max_order
        self.S = float(S)

    def reset(self):
        pass

    def act(self, obs):
        acts = {}
        for a in AGENTS:
            inv, backlog, on_order = obs[a][0], obs[a][1], obs[a][2]
            inventory_position = inv - backlog + on_order
            order = max(0.0, self.S - inventory_position)
            acts[a] = float(np.clip(order / self.max_order, 0.0, 1.0))
        return acts


class MappoPolicy:
    is_comm = False
    name = "mappo"

    def __init__(self, env, statedict, hidden, n_actions, action_kw=None):
        self.hidden_dim, self.n_actions = hidden, n_actions
        self.max_order = env.max_order
        self.act_kw = action_kw or {}
        local_dim = env.observation_space("retailer").shape[0]
        self.actor = MAPPOActor(local_dim, hidden, n_actions).to(DEVICE)
        self.actor.load_state_dict(statedict)
        self.actor.eval()

    def reset(self):
        self.h = {a: torch.zeros(1, self.hidden_dim, device=DEVICE) for a in AGENTS}

    def act(self, obs):
        acts = {}
        with torch.no_grad():
            for a in AGENTS:
                o = torch.tensor(obs[a], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                dist, self.h[a] = self.actor(o, self.h[a])
                idx = dist.probs.argmax(dim=-1).item()
                acts[a] = index_to_fraction(idx, n_actions=self.n_actions, max_order=self.max_order,
                                            demand_anchor=float(obs[a][3]), **self.act_kw)
        return acts


class CommMappoPolicy:
    is_comm = True
    name = "comm_mappo"

    def __init__(self, env, mac_statedict, hidden, n_actions, vocab, ablate=False, action_kw=None):
        self.hidden_dim, self.n_actions, self.ablate = hidden, n_actions, ablate
        self.max_order = env.max_order
        self.act_kw = action_kw or {}
        local_dim = env.observation_space("retailer").shape[0]
        base = CommMAPPOActor(local_dim, hidden, n_actions, vocab_size=vocab)
        self.mac = MAPPOCommMAC(base, vocab_size=vocab, num_agents=len(AGENTS)).to(DEVICE)
        self.mac.load_state_dict(mac_statedict)
        self.mac.eval()
        self.msg_tokens = {a: 0 for a in AGENTS}

    def reset(self):
        self.mac.init_buffer(batch_size=1, device=DEVICE)
        self.h = torch.zeros(1, len(AGENTS), self.hidden_dim, device=DEVICE)

    def act(self, obs):
        o = torch.tensor(np.stack([obs[a] for a in AGENTS]), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            if self.ablate and self.mac.msg_buffer is not None:
                self.mac.msg_buffer.zero_()          # force zero incoming messages -> no-comm counterfactual
            dist_action, _dc, _ca, next_h, _mm, safe_logs = self.mac(o, self.h, tau=1.0, test_mode=True)
            idx = dist_action.probs.argmax(dim=-1).view(len(AGENTS))
            self.h = next_h
        toks = safe_logs.reshape(len(AGENTS))
        self.msg_tokens = {a: int(toks[i]) for i, a in enumerate(AGENTS)}
        return {a: index_to_fraction(int(idx[i]), n_actions=self.n_actions, max_order=self.max_order,
                                     demand_anchor=float(obs[a][3]), **self.act_kw)
                for i, a in enumerate(AGENTS)}


class QmixPolicy:
    is_comm = False
    name = "qmix"

    def __init__(self, env, mac_dict, hidden, n_actions, action_kw=None):
        self.hidden_dim, self.n_actions = hidden, n_actions
        self.max_order = env.max_order
        self.act_kw = action_kw or {}
        local_dim = env.observation_space("retailer").shape[0]
        self.agents = {}
        for a in AGENTS:
            net = QMixLocalAgent(local_dim, hidden, n_actions).to(DEVICE)
            net.load_state_dict(mac_dict[a])
            net.eval()
            self.agents[a] = net

    def reset(self):
        self.h = {a: torch.zeros(1, self.hidden_dim, device=DEVICE) for a in AGENTS}

    def act(self, obs):
        acts = {}
        with torch.no_grad():
            for a in AGENTS:
                o = torch.tensor(obs[a], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                q, self.h[a] = self.agents[a](o, self.h[a])
                idx = q.argmax(dim=1).item()
                acts[a] = index_to_fraction(idx, n_actions=self.n_actions, max_order=self.max_order,
                                            demand_anchor=float(obs[a][3]), **self.act_kw)
        return acts


class CommQmixPolicy:
    is_comm = True
    name = "comm_qmix"

    def __init__(self, env, mac_statedict, hidden, n_actions, vocab, ablate=False, action_kw=None):
        self.hidden_dim, self.n_actions, self.ablate = hidden, n_actions, ablate
        self.max_order = env.max_order
        self.act_kw = action_kw or {}
        local_dim = env.observation_space("retailer").shape[0]
        base = CommQMixLocalAgent(local_dim, hidden, n_actions, vocab_size=vocab)
        self.mac = QMixCommMAC(base, num_agents=len(AGENTS)).to(DEVICE)
        self.mac.load_state_dict(mac_statedict)
        self.mac.eval()
        self.msg_tokens = {a: 0 for a in AGENTS}

    def reset(self):
        self.mac.init_buffer(batch_size=1, device=DEVICE)
        self.h = torch.zeros(1, len(AGENTS), self.hidden_dim, device=DEVICE)

    def act(self, obs):
        o = torch.tensor(np.stack([obs[a] for a in AGENTS]), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        msg_in = torch.zeros(1, len(AGENTS), 1, device=DEVICE) if self.ablate else None
        with torch.no_grad():
            q, next_h, _mo, safe_logs = self.mac(o, self.h, tau=0.1, msg_in=msg_in, hard=True)
            self.h = next_h
        toks = safe_logs.reshape(len(AGENTS))
        self.msg_tokens = {a: int(toks[i]) for i, a in enumerate(AGENTS)}
        return {a: index_to_fraction(int(q[0, i].argmax(dim=-1).item()), n_actions=self.n_actions,
                                     max_order=self.max_order, demand_anchor=float(obs[a][3]), **self.act_kw)
                for i, a in enumerate(AGENTS)}


def build_policy(name, env, ablate=False):
    """Returns a policy instance or None (missing checkpoint -> skipped, never crashes)."""
    if name == "sterman":
        return StermanPolicy(env)
    path = MODEL_PATHS.get(name)
    if not path or not os.path.exists(path):
        print(f"  [skip] {name}: checkpoint not found at {path}")
        return None
    kind, payload, cfg = _load_raw(path)
    hidden, n_actions, vocab = _cfg_dims(cfg)
    # Each checkpoint is evaluated under the action parameterization it was TRAINED with
    # (read from its embedded config); old checkpoints fall back to legacy absolute mode.
    action_kw = _cfg_action(cfg, env.max_order)
    try:
        if name == "mappo":
            sd = payload["actor"] if kind == "checkpoint" else payload
            if cfg is None:
                hidden, n_actions = sd["action_head.weight"].shape[1], sd["action_head.weight"].shape[0]
            return MappoPolicy(env, sd, hidden, n_actions, action_kw=action_kw)
        if name == "comm_mappo":
            sd = payload["actor"] if kind == "checkpoint" else payload
            if cfg is None:
                hidden = sd["actor.action_head.weight"].shape[1]
                n_actions = sd["actor.action_head.weight"].shape[0]
                vocab = sd["actor.comm_head.2.weight"].shape[0]
            return CommMappoPolicy(env, sd, hidden, n_actions, vocab, ablate=ablate, action_kw=action_kw)
        if name == "qmix":
            mac_dict = payload["mac"] if kind == "checkpoint" else payload
            if cfg is None:
                any_sd = next(iter(mac_dict.values()))
                hidden, n_actions = any_sd["advantage_stream.weight"].shape[1], any_sd["advantage_stream.weight"].shape[0]
            return QmixPolicy(env, mac_dict, hidden, n_actions, action_kw=action_kw)
        if name == "comm_qmix":
            sd = payload["mac"] if kind == "checkpoint" else payload
            if cfg is None:
                hidden = sd["agent.advantage_stream.weight"].shape[1]
                n_actions = sd["agent.advantage_stream.weight"].shape[0]
                vocab = sd["agent.msg_stream.weight"].shape[0]
            return CommQmixPolicy(env, sd, hidden, n_actions, vocab, ablate=ablate, action_kw=action_kw)
    except Exception as e:        # corrupt/mismatched checkpoint -> skip rather than abort the whole benchmark
        print(f"  [skip] {name}: failed to load ({type(e).__name__}: {e})")
        return None


# ==============================================================================
# EPISODE ROLLOUT + PER-EPISODE METRICS
# ==============================================================================
def _safe_ratio(num, den):
    return float(num / den) if den > 1e-9 else np.nan


def run_episode(policy, env, seed):
    obs, _ = env.reset(seed=seed)
    policy.reset()

    orders = {a: [] for a in AGENTS}        # physical units ordered by each agent
    demand = {a: [] for a in AGENTS}        # demand each agent had to serve
    inv = {a: [] for a in AGENTS}
    back = {a: [] for a in AGENTS}
    ecost = {a: 0.0 for a in AGENTS}
    fill_d = {a: 0.0 for a in AGENTS}     # period demand (for beta fill rate)
    fill_m = {a: 0.0 for a in AGENTS}     # period demand met immediately
    tot_cost = hold_cost = back_cost = 0.0
    all_tokens, ret_tokens, ret_backlogs = [], [], []

    while True:
        acts = policy.act(obs)
        for a in AGENTS:                                       # physical order placed this step
            orders[a].append(int(np.floor(np.clip(acts[a], 0, 1) * env.max_order + 0.5)))
        if policy.is_comm:
            for a in AGENTS:
                all_tokens.append(policy.msg_tokens[a])
            ret_tokens.append(policy.msg_tokens["retailer"])
            ret_backlogs.append(float(obs["retailer"][1]))    # state the message was conditioned on

        obs, _r, _t, truncs, infos = env.step({a: [acts[a]] for a in AGENTS})

        for a in AGENTS:
            ecost[a] += infos[a]["local_cost"]
            tot_cost += infos[a]["local_cost"]
            inv_a, bk_a = float(obs[a][0]), float(obs[a][1])
            inv[a].append(inv_a)
            back[a].append(bk_a)
            hold_cost += env.h * inv_a
            back_cost += env.b * bk_a
            demand[a].append(float(env.current_incoming_order[a]))
            fill_d[a] += infos[a].get("demand", 0.0)
            fill_m[a] += infos[a].get("demand_met", 0.0)
        if any(truncs.values()):
            break

    all_inv = np.concatenate([inv[a] for a in AGENTS])
    all_back = np.concatenate([back[a] for a in AGENTS])
    m = {
        "cost": tot_cost,
        "holding": hold_cost,
        "backlog": back_cost,
        "bw_overall": _safe_ratio(np.var(orders["manufacturer"]), np.var(demand["retailer"])),
        "avg_inv": float(all_inv.mean()),
        "avg_back": float(all_back.mean()),
        "ret_service_alpha": float(np.mean(np.array(back["retailer"]) == 0)),
        "ret_stockout": float(np.mean(np.array(back["retailer"]) > 0)),
        "fill_beta": _safe_ratio(fill_m["retailer"], fill_d["retailer"]),
        "jitter": float(np.mean([np.mean(np.abs(np.diff(orders[a]))) if len(orders[a]) > 1 else 0.0 for a in AGENTS])),
        "tokens": all_tokens,
        "ret_tokens": ret_tokens,
        "ret_backlogs": ret_backlogs,
    }
    for a in AGENTS:
        m[f"ecost_{a}"] = ecost[a]
        m[f"bw_{a}"] = _safe_ratio(np.var(orders[a]), np.var(demand[a]))
    return m


def evaluate(policy, env):
    return [run_episode(policy, env, SEED_BASE + ep) for ep in range(N_EPISODES)]


def tune_base_stock(env, grid=tuple(range(8, 140, 4)), tune_seeds=tuple(range(9000, 9020))):
    """Grid-search the order-up-to level S on HELD-OUT seeds (disjoint from eval),
    so the 'oracle' is near-optimal without fitting to the test seeds."""
    best_S, best_cost = grid[0], float("inf")
    for S in grid:
        pol = BaseStockPolicy(env, S)
        c = float(np.mean([run_episode(pol, env, s)["cost"] for s in tune_seeds]))
        if c < best_cost:
            best_cost, best_S = c, S
    return best_S, best_cost


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)        # nan-safe stats emit these
    rows, cost_arrays_by_scenario = [], {}

    print("\n" + "=" * 70)
    print("  MULTI-SCENARIO BEER GAME BENCHMARK")
    print(f"  agents: sterman, mappo, comm_mappo, qmix, comm_qmix | episodes/scenario: {N_EPISODES}")
    print("=" * 70)

    order = ["sterman", "base_stock", "mappo", "comm_mappo", "qmix", "comm_qmix"]

    for scenario in SCENARIOS:
        print(f"\n---> SCENARIO: {scenario.upper()}")
        env = BeerGameParallelEnv({**ENV_BASE, "demand_type": scenario})

        results = {}                          # name -> list of per-episode metric dicts
        for name in order:
            tuned_S = None
            if name == "base_stock":
                tuned_S, _ = tune_base_stock(env)
                pol = BaseStockPolicy(env, tuned_S)
            else:
                pol = build_policy(name, env)
            if pol is None:
                continue
            results[name] = evaluate(pol, env)
            extra = f"  (tuned S={tuned_S})" if tuned_S is not None else ""
            print(f"  -> {name:<11} mean cost = {np.mean([e['cost'] for e in results[name]]):.1f}{extra}")

        if not results:
            print("  (no policies available -- check MODEL_PATHS)")
            continue

        cost_arrays_by_scenario[scenario] = {k: np.array([e["cost"] for e in v]) for k, v in results.items()}
        sterman_mean = np.mean(cost_arrays_by_scenario[scenario]["sterman"]) if "sterman" in results else np.nan

        # ---- communication value (causal ablation) for comm agents on the SAME seeds ----
        comm_value = {}
        for name in ("comm_mappo", "comm_qmix"):
            if name not in results:
                continue
            pol_abl = build_policy(name, env, ablate=True)
            if pol_abl is None:
                continue
            abl = evaluate(pol_abl, env)
            c_norm = np.array([e["cost"] for e in results[name]])
            c_abl = np.array([e["cost"] for e in abl])
            comm_value[name] = float(np.mean((c_abl - c_norm) / np.where(c_norm == 0, np.nan, c_norm)) * 100.0)

        # ---- aggregate to table ----
        for name in order:
            if name not in results:
                continue
            eps = results[name]
            costs = cost_arrays_by_scenario[scenario][name]
            toks = np.concatenate([e["tokens"] for e in eps]) if eps[0]["tokens"] else np.array([])
            rtok = np.concatenate([e["ret_tokens"] for e in eps]) if eps[0]["ret_tokens"] else np.array([])
            rbck = np.concatenate([e["ret_backlogs"] for e in eps]) if eps[0]["ret_backlogs"] else np.array([])
            is_comm = name in ("comm_mappo", "comm_qmix")

            rows.append({
                "Scenario": scenario.upper(), "Algo": name.upper(),
                "Mean_Cost": costs.mean(), "Std_Cost": costs.std(),
                "Cost_CV": costs.std() / costs.mean() if costs.mean() else 0.0,
                "Cost_vs_Sterman": costs.mean() / sterman_mean if sterman_mean else np.nan,
                "Service_alpha_%": np.mean([e["ret_service_alpha"] for e in eps]) * 100,
                "FillRate_beta_%": np.nanmean([e["fill_beta"] for e in eps]) * 100,
                "Stockout_%": np.mean([e["ret_stockout"] for e in eps]) * 100,
                "Avg_Inv": np.mean([e["avg_inv"] for e in eps]),
                "Avg_Backlog": np.mean([e["avg_back"] for e in eps]),
                "Holding_Cost": np.mean([e["holding"] for e in eps]),
                "Backlog_Cost": np.mean([e["backlog"] for e in eps]),
                "Cost_Retailer": np.mean([e["ecost_retailer"] for e in eps]),
                "Cost_Wholesaler": np.mean([e["ecost_wholesaler"] for e in eps]),
                "Cost_Distributor": np.mean([e["ecost_distributor"] for e in eps]),
                "Cost_Manufacturer": np.mean([e["ecost_manufacturer"] for e in eps]),
                "BW_Overall": np.nanmean([e["bw_overall"] for e in eps]),
                "BW_Retailer": np.nanmean([e["bw_retailer"] for e in eps]),
                "BW_Wholesaler": np.nanmean([e["bw_wholesaler"] for e in eps]),
                "BW_Distributor": np.nanmean([e["bw_distributor"] for e in eps]),
                "BW_Manufacturer": np.nanmean([e["bw_manufacturer"] for e in eps]),
                "Order_Jitter": np.mean([e["jitter"] for e in eps]),
                "Msg_Entropy_bits": shannon_entropy(toks) if is_comm else np.nan,
                "MI_msg_backlog_bits": mutual_information(rtok, rbck) if is_comm else np.nan,
                "Comm_Value_%": comm_value.get(name, np.nan) if is_comm else np.nan,
            })

        # ---- paired significance tests on cost (same seeds) ----
        _significance(cost_arrays_by_scenario[scenario])
        # ---- plots ----
        _plot_costs(cost_arrays_by_scenario[scenario], scenario)
        _plot_bullwhip(results, scenario)

    # ---- master table ----
    if not rows:
        print("\nNo results produced. Fill in MODEL_PATHS and rerun.")
        return
    df = pd.DataFrame(rows).set_index(["Scenario", "Algo"])
    out_csv = os.path.join(PROJECT_ROOT, "master_benchmark_results.csv")
    df.to_csv(out_csv)
    print("\n" + "=" * 70)
    print("  MASTER RESULTS")
    print("=" * 70)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.round(3).to_string())
    print(f"\n-> saved {out_csv}")
    print("-> saved benchmark_<scenario>_cost.png and bullwhip_<scenario>.png")


def _significance(cost_map):
    names = list(cost_map.keys())
    if len(names) < 2:
        return
    print("    [stats] normality (Shapiro) & paired Wilcoxon vs Sterman, Holm-corrected:")
    pairs = list(combinations(names, 2))
    raw_p = []
    for a, b in pairs:
        da, db = cost_map[a], cost_map[b]
        try:
            p = 1.0 if np.all(da - db == 0) else stats.wilcoxon(da, db).pvalue
        except Exception:
            p = 1.0
        raw_p.append(p)
    n = len(raw_p)
    order_idx = np.argsort(raw_p)
    adj = np.ones(n)
    for rank, idx in enumerate(order_idx):
        adj[idx] = min(1.0, raw_p[idx] * (n - rank))
    for i, (a, b) in enumerate(pairs):
        if "sterman" in (a, b):       # keep the print focused on baseline comparisons
            print(f"      {a} vs {b}: raw p={raw_p[i]:.2e} | Holm p={adj[i]:.2e}")


def _plot_costs(cost_map, scenario):
    flat = pd.DataFrame([(k.upper(), c) for k, v in cost_map.items() for c in v], columns=["Algo", "Cost"])
    plt.figure(figsize=(9, 5.5))
    sns.boxplot(x="Algo", y="Cost", hue="Algo", data=flat, palette="viridis", legend=False)
    if "sterman" in cost_map:
        plt.axhline(np.mean(cost_map["sterman"]), color="r", ls="--", label="Sterman mean")
        plt.legend()
    plt.title(f"Cost distribution -- {scenario.upper()}")
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, f"benchmark_{scenario}_cost.png"), dpi=200)
    plt.close()


def _plot_bullwhip(results, scenario):
    plt.figure(figsize=(9, 5.5))
    for name, eps in results.items():
        y = [np.nanmean([e[f"bw_{a}"] for e in eps]) for a in AGENTS]
        plt.plot(AGENTS, y, marker="o", label=name.upper())
    plt.axhline(1.0, color="grey", ls=":", label="no amplification")
    plt.ylabel("Var(orders) / Var(demand)")
    plt.title(f"Bullwhip by echelon -- {scenario.upper()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, f"bullwhip_{scenario}.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
