"""
================================================================================
 BEER GAME -- MULTI-SCENARIO BENCHMARK  (Sterman / Base-stock / MAPPO / Comm-MAPPO / QMIX / Comm-QMIX)
================================================================================
Evaluates the six in-scope policies under four demand regimes (step, poisson,
black_swan, extreme_chaos) and reports a literature-grounded metric suite.

--------------------------------------------------------------------------------
CHANGES IN THIS REVISION (each marked with a "# FIX n:" tag at the edit site)
--------------------------------------------------------------------------------
 FIX 1  comm_qmix evaluation temperature was hardcoded to tau=0.1. It is now
        read from the checkpoint (the trainer saves the tau in effect at save
        time; fallback: the config's tau_min; final fallback 0.1). Evaluating
        at a temperature the policy never trained with changes the message
        distribution and silently invalidates the comm results.
 FIX 2  torch RNG is now seeded per evaluation episode. comm_qmix messages are
        sampled via Gumbel-softmax (stochastic even with hard=True), so without
        this the benchmark was NOT reproducible run-to-run and the "same seeds"
        pairing assumption did not hold for comm_qmix.
 FIX 3  checkpoint resolution now prefers *_checkpoint_final.pt over
        *_checkpoint_best.pt. The thesis metric is converged (final-window)
        performance; "best" snapshots can capture transient dips (the comm_mappo
        695 -> 1600 blow-up is the documented example). Best remains a fallback.
 FIX 4  Holm-Bonferroni correction now enforces step-down monotonicity
        (np.maximum.accumulate); the previous version could output adjusted
        p-values that decreased down the ranking, which is not Holm.
 FIX 5  Comm_Value_% (zero-message ablation) now also reports a PAIRED Wilcoxon
        p-value (ablated vs intact on identical seeds) so the causal claim
        carries a significance level, not just a point estimate.
 NOTE   The comm_mappo zero-ablation (mac.msg_buffer.zero_()) was AUDITED and is
        CORRECT: MAPPOCommMAC (unlike QMixCommMAC) has no separate
        rollout_msg_state -- forward() reads msg_buffer directly, so zeroing it
        before each forward genuinely silences all incoming messages. The
        ablation is additionally applied in reset() as belt-and-braces.
 NOTE   The deeper communication analysis (per-agent MI with shuffled-null
        significance, multi-condition interventions, per-decision causal
        influence, token-semantics heatmaps -- "Tests 1-4") lives in the
        companion script scripts/comm_analysis.py, which reuses the policy
        wrappers defined here.

--------------------------------------------------------------------------------
METRIC SUITE -- WHAT EACH NUMBER MEANS AND WHY IT IS REPORTED
--------------------------------------------------------------------------------
  COST / OBJECTIVE
    * Mean_Cost            -- the training objective, averaged over eval episodes.
    * Std_Cost, Cost_CV    -- dispersion across demand draws; CV = Std/Mean makes
                              agents with different cost scales comparable.
    * Cost_vs_Sterman      -- ratio anchoring RL numbers to the classical
                              behavioural heuristic (values < 1 beat Sterman).
    * Cost_<echelon>       -- where cost concentrates along the chain.
    * Holding/Backlog cost -- behavioural signature: hoarding vs starving.
  SERVICE / INVENTORY (cost alone hides starvation)
    * Service_alpha_%      -- P(period with zero retailer backlog) (Type-1 service).
    * FillRate_beta_%      -- fraction of customer demand met immediately from
                              stock (Type-2 service; uses env demand_met info).
    * Stockout_%           -- complement view of alpha at the retailer.
    * Avg_Inv / Avg_Backlog-- average physical state across all echelons.
  BASELINES (bracket the RL agents from below and above)
    * sterman              -- anchoring-and-adjustment heuristic (Sterman 1989);
                              competent parameterization, not a strawman.
    * base_stock           -- order-up-to ORACLE; level S grid-tuned on held-out
                              seeds (Clark & Scarf 1960 optimal-policy anchor).
  BULLWHIP (Lee et al. 1997; Sterman 1989)
    * BW_<echelon>         -- Var(orders_echelon)/Var(demand_echelon).
    * BW_Overall           -- Var(manufacturer orders)/Var(retailer demand):
                              end-to-end variance amplification.
    * Order_Jitter         -- mean |order_t - order_{t-1}|: oscillation.
  COMMUNICATION (comm agents only; Lowe et al. 2019 framing)
    * Msg_Entropy_bits     -- channel usage (descriptive; high entropy does NOT
                              imply useful content -- see comm_analysis.py).
    * MI_msg_backlog_bits  -- MI(retailer token; retailer backlog): quick
                              positive-SIGNALLING screen (full test: Test 1).
    * Comm_Value_%         -- % cost increase when incoming messages are zeroed
                              at test time on identical seeds: quick positive-
                              LISTENING screen (full test: Test 2). FIX 5 adds a
                              paired Wilcoxon p-value to this estimate.
  STATISTICS
    * Paired Wilcoxon signed-rank on identical eval seeds; Holm step-down
      correction across all pairwise comparisons (printout focuses on the
      vs-Sterman pairs). NOTE: this tests robustness to DEMAND DRAWS for one
      trained model. Robustness to TRAINING seeds is Phase 2/3 (multi-seed
      checkpoints + rliable IQM), not this script's job.

USAGE
  1. After Phase-2 training, point MODEL_PATHS at the run directories (the
     loader picks *_checkpoint_final.pt, falling back to *_checkpoint_best.pt;
     the embedded config auto-detects architecture dims and action mode).
  2. python scripts/benchmark.py
  Missing paths are skipped with a warning -- the script never hard-crashes.
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
# CONFIG
# ==============================================================================
DEVICE = torch.device("cpu")            # eval is tiny; CPU avoids contending with a running sweep
N_EPISODES = 100                        # paired across algos (same seeds) for Wilcoxon
SEED_BASE = 2000                        # eval seeds SEED_BASE..+N-1; disjoint from training (42+ep)
SCENARIOS = ["step", "poisson", "black_swan", "extreme_chaos"]
AGENTS = ["retailer", "wholesaler", "distributor", "manufacturer"]   # downstream -> upstream
ENV_BASE = {"horizon": 50, "max_order": 100, "holding_cost": 0.5,
            "backorder_cost": 1.0, "lookahead": 4}                   # must match training (obs_dim=8)

# ------------------------------------------------------------------------------
# CHECKPOINT LOCATIONS
# ------------------------------------------------------------------------------
# Each entry is a LIST of candidate paths tried in order (FIX 3: final preferred,
# best as fallback). Point these at the Phase-2 winners' run directories. The
# *_checkpoint_*.pt files embed the resolved training config, so hidden_dim /
# n_actions / vocab_size / action_mode / tau are auto-detected from the file.
# ------------------------------------------------------------------------------
def _candidates(algo):
    d = os.path.join(PROJECT_ROOT, f"weights_{algo}")
    return [os.path.join(d, f"{algo}_checkpoint_final.pt"),   # FIX 3: converged policy first
            os.path.join(d, f"{algo}_checkpoint_best.pt")]    # fallback: best-Avg_Cost_50 snapshot

MODEL_PATHS = {
    "mappo":      _candidates("mappo"),
    "comm_mappo": _candidates("comm_mappo"),
    "qmix":       _candidates("qmix"),
    "comm_qmix":  _candidates("comm_qmix"),
}

# Sterman (1989) anchoring-and-adjustment ordering rule (supply-line aware baseline):
#   O = max(0, D_hat + alpha_S*(S' - S) + alpha_SL*(SL' - SL))
# S = net stock (on-hand - backlog), S' = desired stock, SL = on-order,
# SL' = desired supply line = D_hat * lead_time. Under-weighting alpha_SL is what
# Sterman showed produces the bullwhip; we use a competent (not strawman) setting.
STERMAN = {"target_stock": 12.0, "lead_time": 4, "alpha_stock": 0.5, "alpha_supply": 0.5}


# ==============================================================================
# INFORMATION-THEORETIC HELPERS
# ==============================================================================
def shannon_entropy(tokens):
    """Entropy (bits) of the emitted token distribution.

    Measures how much of the channel's capacity is USED (uniform over k tokens
    -> log2(k) bits; collapsed to one token -> 0). Purely descriptive: a noisy
    channel can have maximal entropy and zero useful content."""
    if len(tokens) == 0:
        return np.nan
    _, counts = np.unique(tokens, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def mutual_information(tokens, state, state_bins=8):
    """MI(message ; sender state) in bits -- the positive-SIGNALLING screen.

    tokens: discrete ints (one per step); state: continuous, binned into
    `state_bins` equal-width bins via histogram2d. Plug-in (empirical) MI
    estimator: I = sum p(x,y) log2( p(x,y) / p(x)p(y) ).
    Degenerate inputs (constant state, single token, <2 samples) return 0.0.
    NOTE: plug-in MI is biased upward for small samples -- comm_analysis.py
    Test 1 adds a shuffled-token null distribution to calibrate significance."""
    tokens = np.asarray(tokens)
    state = np.asarray(state)
    if len(tokens) < 2 or np.var(state) < 1e-9 or np.unique(tokens).size < 2:
        return 0.0
    tok_edges = np.unique(tokens).size
    c, _, _ = np.histogram2d(tokens, state, bins=[tok_edges, state_bins])
    c = c / c.sum()                       # joint p(x,y)
    px = c.sum(axis=1, keepdims=True)     # marginal p(token)
    py = c.sum(axis=0, keepdims=True)     # marginal p(state-bin)
    nz = c > 0                            # avoid log(0)
    return float(np.sum(c[nz] * np.log2(c[nz] / (px @ py)[nz])))


# ==============================================================================
# CHECKPOINT LOADING  (auto-detects architecture from the embedded config)
# ==============================================================================
def _resolve_path(name):
    """Return the first existing candidate path for an algo, else None. (FIX 3)"""
    paths = MODEL_PATHS.get(name, [])
    if isinstance(paths, str):            # tolerate old single-string config
        paths = [paths]
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _load_raw(path):
    """Load a checkpoint file. Returns (kind, payload, cfg).
    kind 'checkpoint' = full training dict (has actor/mac + embedded config);
    kind 'statedict'  = bare state_dict from the older *_best.pth files."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and any(k in ckpt for k in ("actor", "mac", "config")):
        return "checkpoint", ckpt, ckpt.get("config")
    return "statedict", ckpt, None


def _cfg_dims(cfg, fallback=(256, 101, 3)):
    """(hidden_dim, n_actions, vocab_size) from the embedded config, with the
    legacy fallback (256, 101, 3) for very old checkpoints without a config."""
    if cfg is None:
        return fallback
    a = cfg.get("agent", {})
    return a.get("hidden_dim", fallback[0]), a.get("n_actions", fallback[1]), a.get("vocab_size", fallback[2])


def _cfg_action(cfg, max_order):
    """Action-space kwargs for index_to_fraction, read from the checkpoint config,
    so every policy is EVALUATED under the parameterization it was TRAINED with.
    Old checkpoints (no action_mode) fall back to absolute with abs_cap=max_order,
    reproducing the original `order = idx` behaviour exactly."""
    a = (cfg or {}).get("agent", {})
    return {
        "mode": a.get("action_mode", "absolute"),
        "abs_cap": a.get("abs_cap", max_order),
        "centered_range": a.get("centered_range", 10),
    }


def _cfg_eval_tau(payload, cfg):
    """FIX 1: the Gumbel temperature to use when EVALUATING a comm_qmix policy.
    Priority: (1) the tau stored in the checkpoint (the value in effect when the
    weights were saved -- exactly what the policy was running at), (2) the
    config's tau_min (the floor the schedule was heading to), (3) legacy 0.1."""
    if isinstance(payload, dict) and "tau" in payload:
        return float(payload["tau"])
    a = (cfg or {}).get("agent", {})
    return float(a.get("tau_min", 0.1))


# ==============================================================================
# POLICY WRAPPERS -- uniform .reset() / .act(obs) interface
# Each .act(obs) takes the env's obs dict and returns {agent: fraction in [0,1]}
# (the env contract: order = round(fraction * max_order)).
# ==============================================================================
class StermanPolicy:
    """Behavioural anchoring-and-adjustment heuristic (Sterman 1989).
    No learning, no fit to the eval seeds -- the classical human-like anchor."""
    is_comm = False
    name = "sterman"

    def __init__(self, env):
        self.max_order = env.max_order

    def reset(self):
        pass

    def act(self, obs):
        acts = {}
        for a in AGENTS:
            # obs layout: [inventory, backlog, on_order, next_incoming_demand, ...]
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
    (Clark & Scarf 1960; the optimality anchor used in beer-game RL papers)."""
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
            order = max(0.0, self.S - inventory_position)   # order the shortfall to S
            acts[a] = float(np.clip(order / self.max_order, 0.0, 1.0))
        return acts


class MappoPolicy:
    """MAPPO actor, evaluated GREEDILY (argmax of the action distribution).
    Decentralized execution: each echelon runs the shared actor on its own obs
    with its own GRU hidden state. No communication."""
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
        # fresh recurrent state at the start of every episode
        self.h = {a: torch.zeros(1, self.hidden_dim, device=DEVICE) for a in AGENTS}

    def act(self, obs):
        acts = {}
        with torch.no_grad():
            for a in AGENTS:
                o = torch.tensor(obs[a], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                dist, self.h[a] = self.actor(o, self.h[a])
                idx = dist.probs.argmax(dim=-1).item()      # greedy action
                acts[a] = index_to_fraction(idx, n_actions=self.n_actions, max_order=self.max_order,
                                            demand_anchor=float(obs[a][3]), **self.act_kw)
        return acts


class CommMappoPolicy:
    """Comm-MAPPO MAC. One joint forward per step: messages from step t-1 are
    routed through the chain adjacency mask to neighbours and consumed at t.

    ABLATION (ablate=True): incoming messages are forced to zero before every
    forward. AUDITED CORRECT: MAPPOCommMAC.forward reads self.msg_buffer
    directly (it has no separate rollout state), so zeroing msg_buffer is a true
    zero-message intervention. Note the semantics: a zero message equals the
    middle (0-valued) vocabulary token, i.e. the counterfactual is "everyone
    constantly sends the neutral token", which carries zero information."""
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
        self.msg_tokens = {a: 0 for a in AGENTS}    # last emitted token per agent (for logging)

    def reset(self):
        self.mac.init_buffer(batch_size=1, device=DEVICE)
        if self.ablate and self.mac.msg_buffer is not None:
            self.mac.msg_buffer.zero_()             # belt-and-braces: silent from step 0
        self.h = torch.zeros(1, len(AGENTS), self.hidden_dim, device=DEVICE)

    def act(self, obs):
        o = torch.tensor(np.stack([obs[a] for a in AGENTS]), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            if self.ablate and self.mac.msg_buffer is not None:
                self.mac.msg_buffer.zero_()         # silence all incoming messages this step
            # test_mode=True -> comm tokens taken greedily (argmax), so comm_mappo
            # evaluation is fully deterministic given the seed.
            dist_action, _dc, _ca, next_h, _mm, safe_logs = self.mac(o, self.h, tau=1.0, test_mode=True)
            idx = dist_action.probs.argmax(dim=-1).view(len(AGENTS))
            self.h = next_h
        toks = safe_logs.reshape(len(AGENTS))
        self.msg_tokens = {a: int(toks[i]) for i, a in enumerate(AGENTS)}
        return {a: index_to_fraction(int(idx[i]), n_actions=self.n_actions, max_order=self.max_order,
                                     demand_anchor=float(obs[a][3]), **self.act_kw)
                for i, a in enumerate(AGENTS)}


class QmixPolicy:
    """Vanilla QMIX local agents (one independent network per echelon at eval;
    the mixer is a TRAINING device only and plays no role in greedy execution)."""
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
                idx = q.argmax(dim=1).item()                # greedy Q action
                acts[a] = index_to_fraction(idx, n_actions=self.n_actions, max_order=self.max_order,
                                            demand_anchor=float(obs[a][3]), **self.act_kw)
        return acts


class CommQmixPolicy:
    """Comm-QMIX MAC. Messages from step t-1 are routed through the chain
    adjacency to neighbours and consumed at t.

    FIX 1: evaluation temperature `eval_tau` comes from the checkpoint (the tau
    the policy was actually running at when saved), not a hardcoded 0.1.

    DETERMINISM CAVEAT (FIX 2 context): message emission uses Gumbel-softmax,
    which draws Gumbel noise even with hard=True -- message tokens are sampled,
    not argmaxed. This matches the training-time rollout behaviour (so the
    evaluated policy is the trained policy), and run_episode() seeds torch per
    episode so the sampling is reproducible and identically paired across
    policies and ablations.

    ABLATION (ablate=True): an explicit zero msg_in is passed every step, which
    both silences incoming messages and (by the MAC's contract) leaves the
    internal rollout message state untouched."""
    is_comm = True
    name = "comm_qmix"

    def __init__(self, env, mac_statedict, hidden, n_actions, vocab, ablate=False,
                 action_kw=None, eval_tau=0.1):
        self.hidden_dim, self.n_actions, self.ablate = hidden, n_actions, ablate
        self.max_order = env.max_order
        self.act_kw = action_kw or {}
        self.eval_tau = float(eval_tau)                     # FIX 1
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
        # ablate -> force a zero incoming-message vector (constant neutral token)
        msg_in = torch.zeros(1, len(AGENTS), 1, device=DEVICE) if self.ablate else None
        with torch.no_grad():
            q, next_h, _mo, safe_logs = self.mac(o, self.h, tau=self.eval_tau,   # FIX 1
                                                 msg_in=msg_in, hard=True)
            self.h = next_h
        toks = safe_logs.reshape(len(AGENTS))
        self.msg_tokens = {a: int(toks[i]) for i, a in enumerate(AGENTS)}
        return {a: index_to_fraction(int(q[0, i].argmax(dim=-1).item()), n_actions=self.n_actions,
                                     max_order=self.max_order, demand_anchor=float(obs[a][3]), **self.act_kw)
                for i, a in enumerate(AGENTS)}


def build_policy(name, env, ablate=False, ckpt_path=None):
    """Construct a policy by name. Returns the policy or None (missing/corrupt
    checkpoint -> skipped with a warning, never crashes the whole benchmark).

    ckpt_path overrides the MODEL_PATHS lookup -- used by comm_analysis.py and
    by multi-seed evaluation loops to point at specific run checkpoints."""
    if name == "sterman":
        return StermanPolicy(env)
    path = ckpt_path or _resolve_path(name)                 # FIX 3
    if not path or not os.path.exists(path):
        print(f"  [skip] {name}: checkpoint not found (tried {MODEL_PATHS.get(name)})")
        return None
    kind, payload, cfg = _load_raw(path)
    hidden, n_actions, vocab = _cfg_dims(cfg)
    # Evaluate each checkpoint under the action parameterization it TRAINED with.
    action_kw = _cfg_action(cfg, env.max_order)
    try:
        if name == "mappo":
            sd = payload["actor"] if kind == "checkpoint" else payload
            if cfg is None:   # legacy checkpoints: infer dims from weight shapes
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
            eval_tau = _cfg_eval_tau(payload if kind == "checkpoint" else None, cfg)   # FIX 1
            return CommQmixPolicy(env, sd, hidden, n_actions, vocab, ablate=ablate,
                                  action_kw=action_kw, eval_tau=eval_tau)
    except Exception as e:    # corrupt/mismatched checkpoint -> skip, don't abort the run
        print(f"  [skip] {name}: failed to load ({type(e).__name__}: {e})")
        return None


# ==============================================================================
# EPISODE ROLLOUT + PER-EPISODE METRICS
# ==============================================================================
def _safe_ratio(num, den):
    """num/den with NaN (not crash / not inf) when the denominator is ~0
    (e.g. Var(demand)=0 in the deterministic 'step' scenario pre-jump)."""
    return float(num / den) if den > 1e-9 else np.nan


def run_episode(policy, env, seed):
    """Roll ONE evaluation episode with the given policy on the given env seed
    and return a dict of per-episode metrics. The same seed across policies
    yields the same demand sequence -> all cross-policy comparisons are PAIRED."""
    # FIX 2: comm_qmix message emission is Gumbel-SAMPLED, so torch's RNG state
    # affects the trajectory. Seeding per episode makes every evaluation
    # reproducible and keeps the pairing assumption true for stochastic policies.
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    policy.reset()

    orders = {a: [] for a in AGENTS}      # physical units ordered per step
    demand = {a: [] for a in AGENTS}      # demand each agent had to serve per step
    inv = {a: [] for a in AGENTS}         # on-hand inventory trajectory
    back = {a: [] for a in AGENTS}        # backlog trajectory
    ecost = {a: 0.0 for a in AGENTS}      # per-echelon cumulative cost
    fill_d = {a: 0.0 for a in AGENTS}     # cumulative period demand (beta denominator)
    fill_m = {a: 0.0 for a in AGENTS}     # cumulative demand met immediately (beta numerator)
    tot_cost = hold_cost = back_cost = 0.0
    all_tokens, ret_tokens, ret_backlogs = [], [], []

    while True:
        acts = policy.act(obs)
        for a in AGENTS:                  # reconstruct the integer order actually placed
            orders[a].append(int(np.floor(np.clip(acts[a], 0, 1) * env.max_order + 0.5)))
        if policy.is_comm:
            # tokens emitted THIS step were conditioned on the obs BEFORE env.step
            for a in AGENTS:
                all_tokens.append(policy.msg_tokens[a])
            ret_tokens.append(policy.msg_tokens["retailer"])
            ret_backlogs.append(float(obs["retailer"][1]))   # the state the message encodes

        obs, _r, _t, truncs, infos = env.step({a: [acts[a]] for a in AGENTS})

        for a in AGENTS:
            ecost[a] += infos[a]["local_cost"]
            tot_cost += infos[a]["local_cost"]
            inv_a, bk_a = float(obs[a][0]), float(obs[a][1])
            inv[a].append(inv_a)
            back[a].append(bk_a)
            hold_cost += env.h * inv_a               # holding-cost component
            back_cost += env.b * bk_a                # backlog-cost component
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
        # end-to-end bullwhip: manufacturer order variance vs customer demand variance
        "bw_overall": _safe_ratio(np.var(orders["manufacturer"]), np.var(demand["retailer"])),
        "avg_inv": float(all_inv.mean()),
        "avg_back": float(all_back.mean()),
        "ret_service_alpha": float(np.mean(np.array(back["retailer"]) == 0)),  # Type-1 service
        "ret_stockout": float(np.mean(np.array(back["retailer"]) > 0)),
        "fill_beta": _safe_ratio(fill_m["retailer"], fill_d["retailer"]),       # Type-2 service
        # mean |delta order| across echelons: order-stream smoothness
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
    """N_EPISODES rollouts on the FIXED eval-seed ladder -> list of metric dicts.
    Identical seeds across policies make every comparison paired."""
    return [run_episode(policy, env, SEED_BASE + ep) for ep in range(N_EPISODES)]


def tune_base_stock(env, grid=tuple(range(8, 140, 4)), tune_seeds=tuple(range(9000, 9020))):
    """Grid-search the order-up-to level S on HELD-OUT seeds (disjoint from both
    training and evaluation), so the 'oracle' is near-optimal without ever
    fitting to the test seeds. Returns (best_S, its tuning cost)."""
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
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # nan-safe stats emit these
    rows, cost_arrays_by_scenario = [], {}

    print("\n" + "=" * 70)
    print("  MULTI-SCENARIO BEER GAME BENCHMARK")
    print(f"  agents: sterman, base_stock, mappo, comm_mappo, qmix, comm_qmix | episodes/scenario: {N_EPISODES}")
    print("=" * 70)

    order = ["sterman", "base_stock", "mappo", "comm_mappo", "qmix", "comm_qmix"]

    for scenario in SCENARIOS:
        print(f"\n---> SCENARIO: {scenario.upper()}")
        env = BeerGameParallelEnv({**ENV_BASE, "demand_type": scenario})

        results = {}                      # name -> list of per-episode metric dicts
        for name in order:
            tuned_S = None
            if name == "base_stock":
                tuned_S, _ = tune_base_stock(env)    # re-tuned PER SCENARIO (demand-dependent!)
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

        # ---- communication value: zero-message causal ablation on the SAME seeds ----
        # Quick positive-LISTENING screen. The full multi-condition intervention
        # suite (zero / marginal-shuffle / uniform-random + per-decision causal
        # influence) is in scripts/comm_analysis.py.
        comm_value, comm_value_p = {}, {}
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
            # FIX 5: paired significance for the causal claim (same seeds -> Wilcoxon)
            try:
                comm_value_p[name] = float(stats.wilcoxon(c_abl, c_norm).pvalue) \
                    if not np.all(c_abl - c_norm == 0) else 1.0
            except Exception:
                comm_value_p[name] = np.nan
            print(f"     {name}: Comm_Value = {comm_value[name]:+.1f}%  (paired Wilcoxon p={comm_value_p[name]:.2e})")

        # ---- aggregate per-episode metrics into one table row per (scenario, algo) ----
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
                "Comm_Value_p": comm_value_p.get(name, np.nan) if is_comm else np.nan,   # FIX 5
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
    """Paired Wilcoxon signed-rank on identical eval seeds for every algo pair,
    with HOLM step-down correction across all pairs. Printout is filtered to
    the vs-Sterman comparisons to keep the console readable; all adjusted
    p-values are computed over the FULL set of pairs (correct family size).

    Holm (FIX 4): sort raw p ascending; adj_i = max over j<=i of p_j*(n-j);
    the running-max (monotonicity) step was missing before and could yield
    adjusted p-values that DECREASE down the ranking, which is not Holm."""
    names = list(cost_map.keys())
    if len(names) < 2:
        return
    print("    [stats] paired Wilcoxon, Holm step-down corrected (printing vs-Sterman pairs):")
    pairs = list(combinations(names, 2))
    raw_p = []
    for a, b in pairs:
        da, db = cost_map[a], cost_map[b]
        try:
            p = 1.0 if np.all(da - db == 0) else stats.wilcoxon(da, db).pvalue
        except Exception:
            p = 1.0
        raw_p.append(p)
    raw_p = np.asarray(raw_p)
    n = len(raw_p)
    order_idx = np.argsort(raw_p)                       # ascending raw p
    scaled = raw_p[order_idx] * (n - np.arange(n))      # p_(j) * (n - j),  j=0..n-1
    adj_sorted = np.minimum(1.0, np.maximum.accumulate(scaled))   # FIX 4: enforce monotonicity
    adj = np.empty(n)
    adj[order_idx] = adj_sorted
    for i, (a, b) in enumerate(pairs):
        if "sterman" in (a, b):
            print(f"      {a} vs {b}: raw p={raw_p[i]:.2e} | Holm p={adj[i]:.2e}")


def _plot_costs(cost_map, scenario):
    """Boxplot of the per-episode cost distribution per algo, with the Sterman
    mean as a reference line. One PNG per scenario."""
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
    """Per-echelon variance-amplification profile (the bullwhip signature plot):
    a flat line at 1.0 means no amplification along the chain."""
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