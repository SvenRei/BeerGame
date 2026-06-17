"""
draco.py -- Distributionally-Robust Adaptive Communicating Order-policy.

A greenfield policy-based MARL architecture for the Beer Game, designed to train
on (randomized) Poisson demand yet survive the black_swan / extreme_chaos test
regimes. See DRACO_architecture_proposal.md for the full justification. Modules:

  A  base-stock-structured actor      order = clip(S - IP, 0, cap), S = f(z, msg)
  B  context encoder + info-bottleneck z_t infers the demand regime online
  C  neighbor-only belief communication continuous msg = g(z), masked by chain adj
  D  distributional CVaR critic        quantile critic on global state, tail value
  E  demand randomization              DemandRandomizedBeerGame (training only)

Backbone: HAPPO (heterogeneous actors + centralized critic + sequential M-factor
update). The encoder is trained SELF-SUPERVISED (predict next demand + IB) and
its latent z is fed to the actors DETACHED -- so z is a clean demand
representation disentangled from the policy, and the per-agent policy stays tiny.

BRING-UP FIXES (this revision):
  * base-stock warm-start: the actor's S head is biased so initial S sits near a
    sane order-up-to level (s_bias_init), avoiding the dead cold-start where
    S<IP => order 0 for thousands of episodes.
  * multi-episode batching: DRACOTrainer.update() takes a LIST of episode buffers,
    encodes each episode separately (clean recurrent/causal state), and batches
    the feedforward actor/critic update -> far lower gradient variance.

Compile-verified only (no torch in the authoring env). Run a smoke test first.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from envs.beer_game_env import BeerGameParallelEnv


# Chain adjacency = the neighbour-only communication matrix
#   retailer(0) <-> wholesaler(1) <-> distributor(2) <-> manufacturer(3)
ADJ = torch.tensor([
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
])


def _inv_softplus(y):
    """x such that softplus(x) = y (so a bias init produces a target post-softplus value)."""
    return float(math.log(math.expm1(y))) if y < 20 else float(y)


# ==============================================================================
# Module E -- demand randomization (training only; env file untouched)
# ==============================================================================
class DemandRandomizedBeerGame(BeerGameParallelEnv):
    """Domain randomization around the nominal Poisson process (Tobin et al. 2017).
    Per episode: random Poisson rate lambda ~ U[lo,hi], and with prob p_shift a
    single synthetic mid-episode level shift. Applies ONLY when demand_type is
    'poisson' (training); black_swan / extreme_chaos pass straight through, so the
    benchmark test distributions are never altered."""
    def __init__(self, config, lam_lo=4.0, lam_hi=16.0, p_shift=0.5, shift_scale=2.0):
        super().__init__(config)
        self._dr = dict(lo=lam_lo, hi=lam_hi, p_shift=p_shift, scale=shift_scale)
        self._dr_rng = np.random.default_rng()
        self._dr_lambda = 8.0
        self._dr_shift_t = None
        self._dr_shift_lambda = 8.0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._dr_rng = np.random.default_rng(seed + 99991)
        self._dr_lambda = float(self._dr_rng.uniform(self._dr["lo"], self._dr["hi"]))
        if self._dr_rng.random() < self._dr["p_shift"]:
            self._dr_shift_t = int(self._dr_rng.integers(self.horizon // 4, 3 * self.horizon // 4))
            factor = self._dr["scale"] if self._dr_rng.random() < 0.5 else 1.0 / self._dr["scale"]
            self._dr_shift_lambda = max(0.0, self._dr_lambda * factor)
        else:
            self._dr_shift_t = None
        return super().reset(seed=seed, options=options)

    def _roll_stochastic_demand(self, step):
        if self._config.get("demand_type") == "poisson":
            lam = self._dr_lambda
            if self._dr_shift_t is not None and step >= self._dr_shift_t:
                lam = self._dr_shift_lambda
            return self.np_random.poisson(lam)
        return super()._roll_stochastic_demand(step)


# ==============================================================================
# Module B -- context encoder with information bottleneck (SHARED)
# ==============================================================================
class ContextEncoder(nn.Module):
    """GRU over (obs, prev_action, incoming_message) -> latent demand belief z.
    z = mu (deterministic for the policy); the variational (mu, logstd) feeds a
    KL bottleneck and a next-demand prediction head (self-supervised)."""
    def __init__(self, obs_dim, msg_dim, z_dim, hidden):
        super().__init__()
        self.hidden, self.z_dim = hidden, z_dim
        self.gru = nn.GRU(obs_dim + 1 + msg_dim, hidden)
        self.mu = nn.Linear(hidden, z_dim)
        self.logstd = nn.Linear(hidden, z_dim)
        self.demand_head = nn.Sequential(nn.Linear(z_dim, hidden // 2), nn.ReLU(),
                                         nn.Linear(hidden // 2, 1))

    def step(self, obs, prev_a, msg_in, h):
        """Batched single step. obs [B,od], prev_a [B,1], msg_in [B,msg], h [1,B,H]."""
        x = torch.cat([obs / 100.0, prev_a, msg_in], dim=-1).unsqueeze(0)   # [1,B,*]
        out, h = self.gru(x, h)
        return self.mu(out.squeeze(0)), h                                  # z=mu [B,z]

    def evaluate(self, obs_seq, prev_a_seq, msg_in_seq, h0):
        """Sequence eval. *_seq [T,B,*], h0 [1,B,H]. Returns mu, logstd, demand_pred."""
        x = torch.cat([obs_seq / 100.0, prev_a_seq, msg_in_seq], dim=-1)
        out, _ = self.gru(x, h0)
        mu = self.mu(out)
        logstd = self.logstd(out).clamp(-5.0, 2.0)
        return mu, logstd, self.demand_head(mu)


# ==============================================================================
# Module A + C -- per-agent base-stock actor with a belief-message head
# ==============================================================================
class DRACOActor(nn.Module):
    """Heterogeneous (one per echelon). Emits a base-stock TARGET level S and a
    continuous neighbour MESSAGE, both as Gaussian policy heads (PPO actions).
    The env order is the order-up-to shortfall clip(S - IP, 0, max_order).

    s_bias_init warm-starts the target level near a sane order-up-to value so the
    agent orders from episode 0 instead of sitting in the S<IP dead zone."""
    def __init__(self, obs_dim, z_dim, msg_dim, hidden, max_order, s_bias_init=40.0, s_logstd_init=1.0):
        super().__init__()
        self.max_order = float(max_order)
        self.obs_enc = nn.Linear(obs_dim, hidden // 2)
        self.trunk = nn.Sequential(nn.Linear(hidden // 2 + z_dim + msg_dim, hidden), nn.ReLU())
        self.s_mu = nn.Linear(hidden, 1)
        self.s_logstd = nn.Parameter(torch.zeros(1) + s_logstd_init)   # exploration on the target level
        self.m_mu = nn.Linear(hidden, msg_dim)
        self.m_logstd = nn.Parameter(torch.zeros(msg_dim) - 0.5)
        # warm-start the LEVEL via the bias ONLY; keep default (state-responsive) weights.
        # (do NOT shrink s_mu.weight -- that freezes the actor into a constant base-stock.)
        nn.init.constant_(self.s_mu.bias, _inv_softplus(s_bias_init))

    def forward(self, obs, z, msg_in):
        f = self.trunk(torch.cat([F.relu(self.obs_enc(obs / 100.0)), z, msg_in], dim=-1))
        s_mu = F.softplus(self.s_mu(f))                                   # positive target level
        s_std = self.s_logstd.exp().clamp(1e-3, 5.0)
        m_mu = torch.tanh(self.m_mu(f))
        m_std = self.m_logstd.exp().clamp(1e-3, 5.0)
        return s_mu, s_std, m_mu, m_std

    @staticmethod
    def order_from_S(S, obs, max_order):
        """order = clip(S - IP, 0, cap); IP = inv - backlog + on_order = o0 - o1 + o2."""
        IP = obs[..., 0:1] - obs[..., 1:2] + obs[..., 2:3]
        order = torch.clamp(S - IP, min=0.0, max=max_order)
        return order, IP


# ==============================================================================
# Module D -- distributional (quantile) centralized critic + CVaR value
# ==============================================================================
class DistributionalCritic(nn.Module):
    def __init__(self, state_dim, hidden, n_quantiles=8):
        super().__init__()
        self.n_q = n_quantiles
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_quantiles))
        taus = (torch.arange(n_quantiles).float() + 0.5) / n_quantiles
        self.register_buffer("taus", taus)                                # [n_q]

    def forward(self, state):
        return self.net(state / 100.0)                                    # [B, n_q] quantile values

    def cvar_value(self, state, alpha, eta):
        """Risk-distorted scalar value: (1-eta)*mean + eta*CVaR_alpha. return=-cost,
        so the LOWER-tau quantiles are the worst (high-cost) episodes."""
        q = self.forward(state)                                           # [B,n_q]
        mean_v = q.mean(dim=-1, keepdim=True)
        mask = (self.taus <= alpha).float()
        cvar = (q * mask).sum(dim=-1, keepdim=True) / mask.sum().clamp(min=1.0)
        return (1.0 - eta) * mean_v + eta * cvar


def quantile_huber_loss(pred, target, taus, kappa=1.0):
    """pred [B,nq], target [B,nq] (detached). Standard QR-Huber (Dabney et al.)."""
    u = target.unsqueeze(1) - pred.unsqueeze(2)                           # [B, nq_pred, nq_tgt]
    huber = torch.where(u.abs() <= kappa, 0.5 * u.pow(2), kappa * (u.abs() - 0.5 * kappa))
    rho = (taus.view(1, -1, 1) - (u.detach() < 0).float()).abs() * huber
    return rho.sum(dim=2).mean()


# ==============================================================================
# Rollout buffer (one per episode)
# ==============================================================================
class DRACORolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        for k in ("obs", "g", "prev_a", "msg_in", "S_act", "m_act",
                  "logp", "reward", "done", "demand_tgt"):
            setattr(self, k, [])

    def push(self, **kw):
        for k, v in kw.items():
            getattr(self, k).append(v)

    def __len__(self):
        return len(self.obs)


# ==============================================================================
# Trainer -- CVaR-GAE advantage + HAPPO sequential update + self-supervised encoder
# ==============================================================================
class DRACOTrainer:
    def __init__(self, encoder, actors, critic, cfg, total_episodes, device):
        self.encoder, self.actors, self.critic = encoder, actors, critic
        self.device = device
        self.N = len(actors)
        self.gamma = cfg.get("gamma", 0.99)
        self.gae_lambda = cfg.get("gae_lambda", 0.95)
        self.k_epochs = cfg.get("k_epochs", 4)
        self.eps_clip = cfg.get("eps_clip", 0.2)
        self.max_grad_norm = cfg.get("max_grad_norm", 0.5)
        self.cvar_alpha = cfg.get("cvar_alpha", 0.2)
        self.risk_eta = cfg.get("risk_eta", 0.5)        # 0 = risk-neutral, 1 = pure CVaR
        self.ib_beta = cfg.get("ib_beta", 1e-3)
        self.pred_coef = cfg.get("pred_coef", 1.0)
        self.msg_penalty = cfg.get("msg_penalty_coef", 1e-4)
        self.entropy_coef = cfg.get("entropy_coef", 0.01)
        self.use_comm = cfg.get("use_comm", True)        # False = isolate base-stock core (no messages)
        self.use_context = cfg.get("use_context", True)  # False = actor sees zero belief z
        self.hidden = encoder.hidden

        self.actor_opt = [torch.optim.Adam(a.parameters(), lr=cfg.get("lr_actor", 3e-4)) for a in actors]
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.get("lr_critic", 1e-3))
        self.enc_opt = torch.optim.Adam(encoder.parameters(), lr=cfg.get("lr_encoder", 3e-4))

    def _encode(self, obs, prev_a, msg_in, rew, h0):
        """Encoder backend hook. Default = GRU context encoder (obs, prev_action,
        incoming message). Overridden by the CRAFT variant for an action-free
        transformer. Returns (mu_z, logstd_z, demand_pred)."""
        return self.encoder.evaluate(obs, prev_a, msg_in, h0)

    def _gae(self, rewards, values, dones):
        T = rewards.size(0)
        ve = torch.cat([values, torch.zeros(1, 1, device=self.device)], dim=0)
        adv = torch.zeros_like(rewards)
        last = torch.zeros_like(rewards[0])
        for t in reversed(range(T)):
            nonterm = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * ve[t + 1] * nonterm - ve[t]
            last = delta + self.gamma * self.gae_lambda * nonterm * last
            adv[t] = last
        return adv

    def update(self, episodes):
        """episodes: a list of per-episode DRACORolloutBuffer (a single buffer is
        also accepted). Each episode is encoded separately (clean recurrent/causal
        state); the feedforward actor/critic update is batched across all of them."""
        if isinstance(episodes, DRACORolloutBuffer):
            episodes = [episodes]
        episodes = [b for b in episodes if len(b) > 0]
        if not episodes:
            return 0.0, 0.0, 0.0
        N = self.N

        E = []
        for buf in episodes:
            E.append(dict(
                obs=torch.stack(buf.obs).detach(),
                g=torch.stack(buf.g).detach(),
                prev_a=torch.stack(buf.prev_a).detach(),
                msg_in=torch.stack(buf.msg_in).detach(),
                S_act=torch.stack(buf.S_act).detach(),
                m_act=torch.stack(buf.m_act).detach(),
                old_logp=torch.stack(buf.logp).detach(),
                rew=torch.stack(buf.reward).detach(),
                done=torch.stack(buf.done).detach(),
                dtgt=torch.stack(buf.demand_tgt).detach(),
            ))

        # ---- per-episode CVaR-GAE advantage, normalized across the whole batch ----
        with torch.no_grad():
            for d in E:
                V = self.critic.cvar_value(d["g"], self.cvar_alpha, self.risk_eta)
                d["adv"] = self._gae(d["rew"], V, d["done"])
            all_adv = torch.cat([d["adv"] for d in E], dim=0)
            amean, astd = all_adv.mean(), all_adv.std() + 1e-8
            for d in E:
                d["adv"] = (d["adv"] - amean) / astd

        a_loss_tot, c_loss_tot, e_loss_tot = 0.0, 0.0, 0.0
        for _ in range(self.k_epochs):
            # ---- encoder: self-supervised, PER EPISODE (clean recurrent/causal state) ----
            self.enc_opt.zero_grad()
            for d in E:
                h0 = torch.zeros(1, N, self.hidden, device=self.device)
                mu_z, logstd_z, dpred = self._encode(d["obs"], d["prev_a"], d["msg_in"], d["rew"], h0)
                pred_loss = F.mse_loss(dpred, d["dtgt"] / 100.0)
                kl = (-0.5 * (1 + 2 * logstd_z - mu_z.pow(2) - (2 * logstd_z).exp())).mean()
                enc_loss = (self.pred_coef * pred_loss + self.ib_beta * kl) / len(E)
                enc_loss.backward()
                e_loss_tot += enc_loss.item()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
            self.enc_opt.step()

            # ---- recompute detached belief per episode, then concatenate the batch ----
            z_parts = []
            with torch.no_grad():
                for d in E:
                    h0 = torch.zeros(1, N, self.hidden, device=self.device)
                    mu_z, _, _ = self._encode(d["obs"], d["prev_a"], d["msg_in"], d["rew"], h0)
                    z_parts.append(mu_z)
            obs_c = torch.cat([d["obs"] for d in E], dim=0)
            msg_c = torch.cat([d["msg_in"] for d in E], dim=0)
            S_c = torch.cat([d["S_act"] for d in E], dim=0)
            m_c = torch.cat([d["m_act"] for d in E], dim=0)
            olp_c = torch.cat([d["old_logp"] for d in E], dim=0)
            adv_c = torch.cat([d["adv"] for d in E], dim=0)
            z_c = torch.cat(z_parts, dim=0)
            if not self.use_context:
                z_c = torch.zeros_like(z_c)
            if not self.use_comm:
                msg_c = torch.zeros_like(msg_c)
            g_c = torch.cat([d["g"] for d in E], dim=0)
            rew_c = torch.cat([d["rew"] for d in E], dim=0)
            done_c = torch.cat([d["done"] for d in E], dim=0)
            gnext_c = torch.cat([torch.cat([d["g"][1:], d["g"][-1:]], dim=0) for d in E], dim=0)
            B = obs_c.size(0)

            # ---- HAPPO sequential per-agent update over (S, message) ----
            perm = torch.randperm(N).tolist()
            M = torch.ones(B, 1, device=self.device)
            for i in perm:
                s_mu, s_std, m_mu, m_std = self.actors[i](obs_c[:, i], z_c[:, i], msg_c[:, i])
                logp = Normal(s_mu, s_std).log_prob(S_c[:, i]).sum(-1, keepdim=True)
                ent = Normal(s_mu, s_std).entropy().mean()
                if self.use_comm:
                    logp = logp + Normal(m_mu, m_std).log_prob(m_c[:, i]).sum(-1, keepdim=True)
                    ent = ent + Normal(m_mu, m_std).entropy().mean()
                ratio = torch.exp(logp - olp_c[:, i])
                surr1 = ratio * M * adv_c
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * M * adv_c
                msg_pen = self.msg_penalty * m_mu.pow(2).mean() if self.use_comm else 0.0
                loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * ent + msg_pen
                self.actor_opt[i].zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.max_grad_norm)
                self.actor_opt[i].step()
                a_loss_tot += loss.item()
                with torch.no_grad():
                    s2, ss2, mm2, ms2 = self.actors[i](obs_c[:, i], z_c[:, i], msg_c[:, i])
                    lp2 = Normal(s2, ss2).log_prob(S_c[:, i]).sum(-1, keepdim=True)
                    if self.use_comm:
                        lp2 = lp2 + Normal(mm2, ms2).log_prob(m_c[:, i]).sum(-1, keepdim=True)
                    M = (M * torch.exp(lp2 - olp_c[:, i])).clamp(0.1, 10.0)

            # ---- distributional critic (1-step QR-TD) on the concatenated batch ----
            Z = self.critic(g_c)
            with torch.no_grad():
                target = rew_c + self.gamma * (1.0 - done_c) * self.critic(gnext_c)
            c_loss = quantile_huber_loss(Z, target, self.critic.taus)
            self.critic_opt.zero_grad(); c_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_opt.step()
            c_loss_tot += c_loss.item()

        d = max(1, self.k_epochs)
        return a_loss_tot / (d * N), c_loss_tot / d, e_loss_tot / d