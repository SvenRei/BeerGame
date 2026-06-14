"""
================================================================================
 COMMUNICATION ANALYSIS -- Tests 1-4  (companion to scripts/benchmark.py)
================================================================================
Answers the Lowe et al. (2019) question the cost table cannot: do the learned
messages MEAN anything? Two independent properties must BOTH hold before
"communication is used" may be claimed:

  POSITIVE SIGNALLING  -- messages are correlated with the sender's private
                          state (Test 1, with a shuffled-token null for
                          significance, because plug-in MI is biased upward).
  POSITIVE LISTENING   -- messages causally change receiver behaviour
                          (Test 2: episode-level cost interventions;
                           Test 3: per-decision counterfactual influence).

Test 4 produces the interpretability figures (token-vs-state heatmaps).

PRE-REGISTERED DECISION RULE (state this in the thesis BEFORE looking):
  claim "communication is used" iff
    (a) Test 1 MI exceeds the 95th percentile of its shuffled null, AND
    (b) Test 2 zero-intervention increases cost with paired Wilcoxon p < 0.05.
  Anything less is reported as "channel active but not demonstrably useful".

DESIGN NOTES
  * All policy wrappers / checkpoint loading are IMPORTED from benchmark.py --
    single source of truth, no duplicated model code.
  * All rollouts use the same fixed eval-seed ladder as benchmark.py and seed
    torch per episode, so every condition is PAIRED on demand AND on the
    Gumbel noise driving comm_qmix's message sampling.
  * "shuffle" here = marginal-preserving randomization: incoming tokens are
    replaced by i.i.d. draws from each sender's EMPIRICAL token marginal
    (collected in a first pass). This destroys timing/content while preserving
    token statistics -- it separates "content matters" (intact >> shuffle)
    from "any traffic matters" (shuffle ~ intact but both >> zero).

USAGE
  1. Set CHECKPOINTS below (one path per comm algo; lists allowed for Phase-2
     multi-seed -- results are then reported per seed).
  2. python scripts/comm_analysis.py
  Outputs: comm_analysis_results.csv + comm_semantics_<algo>_<agent>.png
================================================================================
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))   # sibling import of benchmark.py

from envs.beer_game_env import BeerGameParallelEnv
from agents.rl.comm_utils import get_vocab_tensor
from agents.action_space import index_to_fraction
# Reuse the audited wrappers + loaders -- no duplicated model code here.
from benchmark import (AGENTS, ENV_BASE, SEED_BASE, DEVICE,
                       build_policy, _load_raw, _cfg_dims, mutual_information)

# ==============================================================================
# CONFIG
# ==============================================================================
N_EPISODES = 100            # episodes per condition (paired seeds, same ladder as benchmark)
N_EPISODES_KL = 20          # Test 3 is per-step x per-counterfactual -> subsample episodes
N_NULL_SHUFFLES = 200       # permutations for the Test-1 shuffled-MI null
SCENARIO = "poisson"        # primary regime; rerun with "step" etc. for the regime grid
STATE_FEATURES = {          # sender-state features for MI / heatmaps: obs index -> name
    0: "inventory", 1: "backlog", 2: "on_order", 3: "incoming_demand",
}

# One or more checkpoints per comm algo. Lists => per-seed rows in the output
# (Phase 2: put the five seed checkpoints here and aggregate downstream).
CHECKPOINTS = {
    "comm_mappo": [os.path.join(PROJECT_ROOT, "weights_comm_mappo", "comm_mappo_checkpoint_final.pt")],
    "comm_qmix":  [os.path.join(PROJECT_ROOT, "weights_comm_qmix",  "comm_qmix_checkpoint_final.pt")],
}


# ==============================================================================
# LOW-LEVEL ROLLOUT WITH MESSAGE CONTROL
# ------------------------------------------------------------------------------
# benchmark.py's wrappers only support intact / zero. Tests 2-4 need arbitrary
# per-step control of the INCOMING message vector, so we drive the MACs directly.
# Contract per step:
#   true_msgs   = what the agents actually emitted last step (advances normally)
#   fed_msgs    = condition(true_msgs)  -> what the receivers are SHOWN
# i.e. senders always behave naturally; only the wire is intervened on. The MAC
# applies the chain adjacency to whatever it is fed, so topology is preserved.
# ==============================================================================
class _MsgCondition:
    """Transforms the incoming message vector [1, N, 1] per step."""
    def __init__(self, kind, vocab_values=None, marginals=None, rng=None):
        self.kind = kind                      # "intact" | "zero" | "shuffle" | "uniform"
        self.vocab_values = vocab_values      # tensor of token VALUES (e.g. [-1,0,1])
        self.marginals = marginals            # per-agent empirical token distribution
        self.rng = rng or np.random.default_rng(0)

    def __call__(self, true_msgs):
        if self.kind == "intact":
            return true_msgs
        if self.kind == "zero":               # constant neutral (0-valued) token
            return torch.zeros_like(true_msgs)
        n = true_msgs.shape[1]
        vals = self.vocab_values.cpu().numpy()
        if self.kind == "uniform":            # uniform random tokens (max disruption)
            idx = self.rng.integers(0, len(vals), size=n)
        else:                                 # "shuffle": draw from each sender's marginal
            idx = np.array([self.rng.choice(len(vals), p=self.marginals[i]) for i in range(n)])
        out = torch.tensor(vals[idx], dtype=true_msgs.dtype, device=true_msgs.device)
        return out.view(1, n, 1)


def _greedy_orders_from(policy, q_or_dist, obs):
    """Map per-agent greedy action indices to env fractions (same path as benchmark)."""
    acts = {}
    for i, a in enumerate(AGENTS):
        if policy.name == "comm_mappo":
            idx = int(q_or_dist.probs.argmax(dim=-1).view(len(AGENTS))[i].item())
        else:                                  # comm_qmix: q_or_dist is the Q tensor [1, N, A]
            idx = int(q_or_dist[0, i].argmax(dim=-1).item())
        acts[a] = index_to_fraction(idx, n_actions=policy.n_actions, max_order=policy.max_order,
                                    demand_anchor=float(obs[a][3]), **policy.act_kw)
    return acts


def rollout_with_condition(policy, env, seed, condition, record_states=False):
    """One episode under a message-intervention condition.

    Returns (total_cost, records) where records (if requested) is a list of
    dicts {agent_idx, token, inventory, backlog, on_order, incoming_demand} --
    the (token, sender-state) pairs Tests 1 & 4 are built from. Tokens recorded
    are the TRUE emitted tokens (sender side), independent of the intervention.
    """
    torch.manual_seed(seed)                    # pair Gumbel noise across conditions
    obs, _ = env.reset(seed=seed)
    policy.reset()
    mac = policy.mac
    n = len(AGENTS)
    true_msgs = torch.zeros(1, n, 1, device=DEVICE)   # message state we advance ourselves
    total_cost, records = 0.0, []

    while True:
        o = torch.tensor(np.stack([obs[a] for a in AGENTS]), dtype=torch.float32,
                         device=DEVICE).unsqueeze(0)
        fed = condition(true_msgs)             # intervene on the wire only
        with torch.no_grad():
            if policy.name == "comm_qmix":
                # explicit msg_in: MAC consumes it and does NOT touch its own
                # rollout state -- we own the message state entirely.
                q, next_h, msg_out, safe_logs = mac(o, policy.h, tau=policy.eval_tau,
                                                    msg_in=fed, hard=True)
                head = q
            else:                              # comm_mappo: MAC reads self.msg_buffer
                mac.msg_buffer = fed.clone()   # inject the (possibly intervened) incoming msgs
                dist_action, _dc, _ca, next_h, _mm, safe_logs = mac(o, policy.h, tau=1.0,
                                                                    test_mode=True)
                msg_out = mac.msg_buffer.clone()   # forward stored the TRUE emitted tokens here
                head = dist_action
        policy.h = next_h
        true_msgs = msg_out.detach()           # senders advance naturally regardless of condition

        if record_states:
            toks = safe_logs.reshape(n)
            for i, a in enumerate(AGENTS):
                rec = {"agent": a, "token": int(toks[i])}
                for k, fname in STATE_FEATURES.items():
                    rec[fname] = float(obs[a][k])
                records.append(rec)

        acts = _greedy_orders_from(policy, head, obs)
        obs, _r, _t, truncs, infos = env.step({a: [acts[a]] for a in AGENTS})
        total_cost += sum(infos[a]["local_cost"] for a in AGENTS)
        if any(truncs.values()):
            break
    return total_cost, records


# ==============================================================================
# TEST 1 -- POSITIVE SIGNALLING: MI(token; sender state) vs shuffled null
# ==============================================================================
def test1_signalling(df_records, vocab_size, rng):
    """For each (agent, state feature): empirical MI of (token, feature) pairs,
    plus a permutation null (tokens shuffled within agent -> destroys the
    pairing, preserves both marginals). Significant = MI > 95th pct of null.
    Output rows: agent, feature, MI_bits, null95, p_value, significant."""
    rows = []
    for agent in AGENTS:
        sub = df_records[df_records.agent == agent]
        toks = sub["token"].to_numpy()
        for fname in STATE_FEATURES.values():
            state = sub[fname].to_numpy()
            mi = mutual_information(toks, state)
            null = np.array([mutual_information(rng.permutation(toks), state)
                             for _ in range(N_NULL_SHUFFLES)])
            p = float((np.sum(null >= mi) + 1) / (len(null) + 1))   # permutation p-value
            rows.append({"test": "T1_signalling", "agent": agent, "feature": fname,
                         "MI_bits": mi, "null95_bits": float(np.quantile(null, 0.95)),
                         "p_value": p, "significant": p < 0.05,
                         "capacity_bits": float(np.log2(max(vocab_size, 2)))})
    return rows


# ==============================================================================
# TEST 2 -- POSITIVE LISTENING: episode-cost interventions (paired)
# ==============================================================================
def test2_listening(policy, env, marginals, vocab_values, rng):
    """Cost under intact / zero / shuffle / uniform on identical seeds.
    Reported per non-intact condition: mean %Delta cost vs intact and the
    paired Wilcoxon p. Interpretation guide:
      intact << zero               -> agents listen (content or traffic)
      intact << shuffle            -> CONTENT matters (timing/values, not just traffic)
      shuffle ~ intact << zero     -> only traffic statistics matter (weak listening)
      all ~ equal                  -> nobody listens; channel is dead weight."""
    conds = {
        "intact":  _MsgCondition("intact"),
        "zero":    _MsgCondition("zero"),
        "shuffle": _MsgCondition("shuffle", vocab_values, marginals, rng),
        "uniform": _MsgCondition("uniform", vocab_values, marginals, rng),
    }
    costs = {k: [] for k in conds}
    for ep in range(N_EPISODES):
        for k, cond in conds.items():
            c, _ = rollout_with_condition(policy, env, SEED_BASE + ep, cond)
            costs[k].append(c)
    base = np.array(costs["intact"])
    rows = []
    for k in ("zero", "shuffle", "uniform"):
        arr = np.array(costs[k])
        delta = float(np.mean((arr - base) / np.where(base == 0, np.nan, base)) * 100.0)
        try:
            p = float(stats.wilcoxon(arr, base).pvalue) if not np.all(arr - base == 0) else 1.0
        except Exception:
            p = np.nan
        rows.append({"test": "T2_listening", "condition": k,
                     "mean_cost_intact": float(base.mean()), "mean_cost_cond": float(arr.mean()),
                     "delta_cost_pct": delta, "p_value": p, "significant": (p < 0.05) if p == p else False})
    return rows


# ==============================================================================
# TEST 3 -- PER-DECISION CAUSAL INFLUENCE (Jaques et al. 2019 style)
# ==============================================================================
def _action_probs(policy, mac, o, h, fed):
    """Per-agent action distribution [N, A] for a given incoming-message vector,
    WITHOUT advancing any state (hidden/message buffers restored by caller)."""
    with torch.no_grad():
        if policy.name == "comm_qmix":
            q, _h, _m, _l = mac(o, h, tau=policy.eval_tau, msg_in=fed, hard=True)
            return torch.softmax(q[0], dim=-1)            # Boltzmann(Q, T=1): a distance proxy
        saved = mac.msg_buffer.clone() if mac.msg_buffer is not None else None
        mac.msg_buffer = fed.clone()
        dist_action, _dc, _ca, _h, _mm, _l = mac(o, h, tau=1.0, test_mode=True)
        if saved is not None:
            mac.msg_buffer = saved                        # restore: counterfactual must not leak
        return dist_action.probs.view(len(AGENTS), -1)


def test3_influence(policy, env, vocab_values):
    """At each step: KL(action dist | true incoming msgs || action dist | each
    constant counterfactual token), plus the greedy-action flip rate. High KL /
    flip rate = the incoming message is causally steering decisions THIS step.
    For comm_qmix the 'distribution' is softmax(Q) -- an ordinal proxy, flagged
    as such in the output (flip rate is the cleaner QMIX statistic)."""
    kls, flips, steps = [], 0, 0
    for ep in range(N_EPISODES_KL):
        seed = SEED_BASE + ep
        torch.manual_seed(seed)
        obs, _ = env.reset(seed=seed)
        policy.reset()
        mac = policy.mac
        n = len(AGENTS)
        true_msgs = torch.zeros(1, n, 1, device=DEVICE)
        while True:
            o = torch.tensor(np.stack([obs[a] for a in AGENTS]), dtype=torch.float32,
                             device=DEVICE).unsqueeze(0)
            h_snap = policy.h.clone()                     # counterfactuals reuse the SAME hidden
            p_fact = _action_probs(policy, mac, o, h_snap, true_msgs)
            a_fact = p_fact.argmax(dim=-1)
            for v in vocab_values.tolist():               # constant-token counterfactuals
                cf = torch.full_like(true_msgs, float(v))
                p_cf = _action_probs(policy, mac, o, h_snap, cf)
                kl = (p_fact * (torch.log(p_fact + 1e-12) - torch.log(p_cf + 1e-12))).sum(-1)
                kls.append(float(kl.mean()))
                flips += int((p_cf.argmax(dim=-1) != a_fact).sum())
                steps += n
            # advance the episode normally (intact channel)
            with torch.no_grad():
                if policy.name == "comm_qmix":
                    q, next_h, msg_out, _ = mac(o, policy.h, tau=policy.eval_tau,
                                                msg_in=true_msgs, hard=True)
                    head = q
                else:
                    mac.msg_buffer = true_msgs.clone()
                    head, _dc, _ca, next_h, _mm, _l = mac(o, policy.h, tau=1.0, test_mode=True)
                    msg_out = mac.msg_buffer.clone()
            policy.h = next_h
            true_msgs = msg_out.detach()
            acts = _greedy_orders_from(policy, head, obs)
            obs, _r, _t, truncs, _i = env.step({a: [acts[a]] for a in AGENTS})
            if any(truncs.values()):
                break
    return [{"test": "T3_influence",
             "mean_KL_nats": float(np.mean(kls)) if kls else np.nan,
             "action_flip_rate": flips / max(steps, 1),
             "note": "QMIX KL uses softmax(Q,T=1) as a proxy; flip rate is exact"}]


# ==============================================================================
# TEST 4 -- SEMANTICS HEATMAPS: P(token | state bin)
# ==============================================================================
def test4_semantics(df_records, algo, vocab_size, out_dir, n_bins=6):
    """One PNG per agent: rows = tokens, cols = quantile bins of each state
    feature, cell = P(token | bin). An interpretable protocol shows clear
    column structure (e.g. token -1 dominant in high-backlog bins)."""
    paths = []
    for agent in AGENTS:
        sub = df_records[df_records.agent == agent]
        if len(sub) < 10:
            continue
        fig, axes = plt.subplots(1, len(STATE_FEATURES), figsize=(4.2 * len(STATE_FEATURES), 3.4))
        for ax, fname in zip(np.atleast_1d(axes), STATE_FEATURES.values()):
            s = sub[fname].to_numpy()
            # quantile bins (degenerate features -> single bin handled by duplicates='drop')
            try:
                bins = pd.qcut(s, q=n_bins, duplicates="drop")
            except ValueError:
                ax.set_visible(False)
                continue
            tab = pd.crosstab(sub["token"], bins, normalize="columns") \
                    .reindex(range(vocab_size), fill_value=0.0)
            sns.heatmap(tab, ax=ax, cmap="viridis", vmin=0, vmax=1,
                        cbar_kws={"label": "P(token | bin)"})
            ax.set_title(fname)
            ax.set_xlabel("")
        fig.suptitle(f"{algo} -- {agent}: token semantics")
        fig.tight_layout()
        p = os.path.join(out_dir, f"comm_semantics_{algo}_{agent}.png")
        fig.savefig(p, dpi=200)
        plt.close(fig)
        paths.append(p)
    return paths


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    rng = np.random.default_rng(0)
    env = BeerGameParallelEnv({**ENV_BASE, "demand_type": SCENARIO})
    all_rows = []

    for algo, paths in CHECKPOINTS.items():
        for ckpt in paths:
            if not os.path.exists(ckpt):
                print(f"[skip] {algo}: {ckpt} not found")
                continue
            tag = os.path.basename(os.path.dirname(ckpt)) or ckpt
            print(f"\n=== {algo}  ({tag})  scenario={SCENARIO} ===")
            policy = build_policy(algo, env, ablate=False, ckpt_path=ckpt)
            if policy is None:
                continue
            _, _, vocab = _cfg_dims(_load_raw(ckpt)[2])
            vocab_values = get_vocab_tensor(vocab, DEVICE)

            # ---- Pass 1: intact rollouts -> (token, state) records + token marginals
            records = []
            for ep in range(N_EPISODES):
                _, recs = rollout_with_condition(policy, env, SEED_BASE + ep,
                                                 _MsgCondition("intact"), record_states=True)
                records.extend(recs)
            df_rec = pd.DataFrame(records)
            marginals = []
            for a in AGENTS:                  # empirical token marginal per agent (for "shuffle")
                counts = np.bincount(df_rec[df_rec.agent == a]["token"], minlength=vocab).astype(float)
                marginals.append(counts / counts.sum() if counts.sum() else np.full(vocab, 1.0 / vocab))

            # ---- Tests
            t1 = test1_signalling(df_rec, vocab, rng)
            t2 = test2_listening(policy, env, marginals, vocab_values, rng)
            t3 = test3_influence(policy, env, vocab_values)
            t4_paths = test4_semantics(df_rec, algo, vocab, PROJECT_ROOT)

            for r in t1 + t2 + t3:
                r.update({"algo": algo, "checkpoint": ckpt, "scenario": SCENARIO})
                all_rows.append(r)

            # ---- Console verdict per the pre-registered rule
            sig_any = any(r["significant"] for r in t1)
            zero_row = next(r for r in t2 if r["condition"] == "zero")
            listens = zero_row["significant"] and zero_row["delta_cost_pct"] > 0
            print(f"  T1 signalling significant (any agent/feature): {sig_any}")
            print(f"  T2 zero-intervention: {zero_row['delta_cost_pct']:+.1f}% cost, p={zero_row['p_value']:.2e}")
            print(f"  T3 influence: KL={t3[0]['mean_KL_nats']:.4f} nats, flip={t3[0]['action_flip_rate']:.3f}")
            print(f"  -> VERDICT: {'COMMUNICATION USED' if (sig_any and listens) else 'NOT DEMONSTRABLY USED'}")
            print(f"  heatmaps: {len(t4_paths)} saved")

    if all_rows:
        out = os.path.join(PROJECT_ROOT, "comm_analysis_results.csv")
        pd.DataFrame(all_rows).to_csv(out, index=False)
        print(f"\n-> saved {out}")
    else:
        print("\nNo results. Fill in CHECKPOINTS and rerun.")


if __name__ == "__main__":
    main()