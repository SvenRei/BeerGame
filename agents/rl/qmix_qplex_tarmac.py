"""
qmix_qplex_tarmac.py
====================
Publication-baseline variant of the comm_qmix stack with two orthogonal,
independently-toggleable upgrades, so you get a clean 2x2 ablation:

  MIXER  in {qmix, qplex}
      qmix  : monotonic QMIX mixer (Rashid et al. 2018) -- the existing baseline,
              re-implemented here with the unified (full-Q + onehot) signature.
      qplex : QPLEX duplex-dueling mixer (Wang et al., ICLR 2021). Represents the
              COMPLETE IGM function class (not just QMIX's monotonic subset) via a
              dueling V/A decomposition with positive transforms and an
              attention-based, action-dependent advantage weighting lambda_i>0.

  READER in {concat, tarmac}
      concat : per-edge directional concatenation [downstream, upstream] -- the
               existing baseline reader (byte-equivalent wiring to qmix.py).
      tarmac : TarMAC-style targeted reading (Das et al., ICLR 2019). Each sender
               emits a discrete token (content) AND a continuous signature key
               from its hidden state; each receiver attends over its neighbours'
               (key, token-embedding) pairs with a query from its own observation.

DESIGN INVARIANTS
  * The public MAC return is the SAME 5-tuple as qmix.py
        (q_vals, next_hiddens, msg_out, safe_logs, incoming_msgs)
    so the trainer / comm_analysis unpacking is unchanged.
  * `incoming_msgs` (fed to the NDQ expressiveness decoder) is ALWAYS the raw
    per-edge neighbour token VALUES [B, N, 2], independent of the reader -- the
    expressiveness objective stays grounded in the transmitted symbols, not in
    the (learned) reading mechanism.
  * vocab_size=1 stays an inert no-comm control under both readers.

NOTE ON tarmac + comm_analysis: the T2/T3 message INTERVENTIONS in comm_analysis
operate on the scalar value channel; for tarmac the routing also depends on the
internal (idx, key) state, so the intact/cost benchmark is valid but the
zero/shuffle interventions need a tarmac-aware extension before they are
meaningful. Use qmix/qplex + concat for the full T1-T3 causal battery, and use
tarmac for the cost benchmark + T1 signalling.

I could not execute this in the authoring environment -- run the smoke test
(test_qplex_tarmac.py) before any training.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .comm_utils import get_vocab_tensor
except ImportError:
    from comm_utils import get_vocab_tensor


# ==============================================================================
# NDQ expressiveness decoder (unchanged; msg_dim = width of decoder's incoming)
# ==============================================================================
class MessageDecoder(nn.Module):
    def __init__(self, obs_dim, n_actions, msg_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + msg_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs, msg_in):
        return self.net(torch.cat([obs, msg_in], dim=-1))


# ==============================================================================
# Communicating agent with selectable reader
# ==============================================================================
class CommQMixAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions, vocab_size=3,
                 reader="concat", key_dim=16, n_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.reader = reader
        self.key_dim = key_dim
        self.n_heads = n_heads
        half = hidden_dim // 2

        self.obs_encoder = nn.Linear(input_dim, half)

        if reader == "concat":
            self.msg_encoder = nn.Linear(2, half)
        elif reader == "tarmac":
            # token content -> value vec. Linear(one_hot) is DIFFERENTIABLE w.r.t.
            # the (straight-through) one-hot, so TD gradient reaches msg_stream.
            self.token_embed = nn.Linear(vocab_size, half, bias=False)
            self.query_proj = nn.Linear(half, key_dim * n_heads)       # receiver query
            self.msg_proj = nn.Linear(half, half)
            self.key_stream = nn.Linear(hidden_dim, key_dim * n_heads) # sender signature
        else:
            raise ValueError(f"reader must be concat|tarmac, got {reader}")

        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.value_stream = nn.Linear(hidden_dim, 1)
        self.advantage_stream = nn.Linear(hidden_dim, n_actions)
        self.msg_stream = nn.Linear(hidden_dim, vocab_size)

    def _emit(self, h, tau, hard, sample, device):
        """Returns msg_out (value [.,1]), msg_idx [.,1], msg_oh (straight-through
        one-hot [.,vocab], differentiable)."""
        if self.vocab_size == 1:
            z = torch.zeros(h.size(0), 1, device=device)
            return z, z.long(), torch.ones(h.size(0), 1, device=device)
        logits = self.msg_stream(h)
        if sample:
            probs = F.gumbel_softmax(logits, tau=tau, hard=hard)       # straight-through one-hot
        else:
            idx = logits.argmax(dim=-1, keepdim=True)
            probs = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
        vocab = get_vocab_tensor(self.vocab_size, device)
        msg_out = (probs * vocab).sum(dim=-1, keepdim=True)
        msg_idx = probs.argmax(dim=-1, keepdim=True)
        return msg_out, msg_idx, probs

    def forward(self, obs, hidden, tau=1.0, hard=True, sample=True,
                concat_msg=None, nbr_oh=None, nbr_key=None, nbr_mask=None):
        scaled_obs = obs / 100.0
        obs_feat = F.relu(self.obs_encoder(scaled_obs))               # [M, half]

        if self.reader == "concat":
            msg_feat = F.relu(self.msg_encoder(concat_msg))           # [M, half]
        else:  # tarmac: attention over neighbour (key, token-embedding)
            M = obs.size(0); nh, Kd = self.n_heads, self.key_dim
            query = self.query_proj(obs_feat).view(M, nh, Kd)         # [M, nh, Kd]
            keys = nbr_key.view(M, 2, nh, Kd)                         # [M, 2, nh, Kd]
            scores = (query.unsqueeze(1) * keys).sum(-1) / math.sqrt(Kd)  # [M, 2, nh]
            scores = scores.permute(0, 2, 1)                         # [M, nh, 2]
            scores = scores.masked_fill((nbr_mask < 0.5).unsqueeze(1), float("-inf"))
            attn = torch.softmax(scores, dim=-1).mean(dim=1)         # [M, 2]
            val_emb = self.token_embed(nbr_oh)                       # [M, 2, half]  (differentiable)
            msg_feat = (attn.unsqueeze(-1) * val_emb).sum(dim=1)     # [M, half]
            msg_feat = F.relu(self.msg_proj(msg_feat))

        x = torch.cat([obs_feat, msg_feat], dim=-1)                  # [M, hidden]
        h = self.rnn(x, hidden.reshape(-1, self.hidden_dim))

        V = self.value_stream(h)
        A = self.advantage_stream(h)
        q_vals = V + A - A.mean(dim=1, keepdim=True)

        msg_out, msg_idx, msg_oh = self._emit(h, tau, hard, sample, obs.device)
        key_out = self.key_stream(h) if self.reader == "tarmac" else None
        return q_vals, msg_out, msg_idx, key_out, h, msg_oh


class QMixCommMAC(nn.Module):
    """Per-edge routing + selectable reader. For tarmac the message state
    (straight-through one-hot + key) is threaded DIFFERENTIABLY across the
    training unroll (attached when an explicit msg_in is supplied, i.e. during
    the batched update; detached during single-step rollout). This is what lets
    gradient from step t+1's attention reach msg_stream AND key_stream at step t.
    """
    def __init__(self, agent_network, num_agents=4):
        super().__init__()
        self.agent = agent_network
        self.num_agents = num_agents
        self.reader = getattr(agent_network, "reader", "concat")
        self.vocab = agent_network.vocab_size
        self.Hk = agent_network.key_dim * agent_network.n_heads if self.reader == "tarmac" else 0
        N = num_agents
        m = torch.zeros(N, 2); m[1:, 0] = 1.0; m[:-1, 1] = 1.0       # topological validity
        self.register_buffer("nbr_mask", m)
        self._val = None      # token values [B,N,1]
        self._oh = None       # straight-through one-hot [B,N,vocab] (tarmac)
        self._key = None      # sender keys [B,N,Hk] (tarmac)
        self.last_incoming_msgs = None

    def init_buffer(self, batch_size, device):
        self.reset_comm_state(batch_size, device)

    def reset_comm_state(self, B, device):
        self._val = torch.zeros(B, self.num_agents, 1, device=device)
        if self.reader == "tarmac":
            self._oh = torch.zeros(B, self.num_agents, self.vocab, device=device)
            self._key = torch.zeros(B, self.num_agents, self.Hk, device=device)

    @staticmethod
    def _route(x):
        """x [B,N,C] -> (down,up): agent i gets i-1 in down, i+1 in up."""
        down = torch.zeros_like(x); up = torch.zeros_like(x)
        down[:, 1:] = x[:, :-1]
        up[:, :-1] = x[:, 1:]
        return down, up

    def forward(self, obs, hiddens, tau=1.0, msg_in=None, hard=True, sample=True):
        B = obs.size(0); N = self.num_agents; dev = obs.device
        if self._val is None or self._val.size(0) != B or self._val.device != dev:
            self.reset_comm_state(B, dev)

        using_explicit = msg_in is not None
        cur_val = msg_in if using_explicit else self._val
        dval, uval = self._route(cur_val)
        incoming_vals = torch.cat([dval, uval], dim=-1)              # [B,N,2] (decoder + concat)
        mask = self.nbr_mask.unsqueeze(0).expand(B, N, 2).reshape(B * N, 2)

        if self.reader == "concat":
            q, msg_out, msg_idx, _k, nh_out, _oh = self.agent(
                obs.reshape(B * N, -1), hiddens.reshape(B * N, -1),
                tau=tau, hard=hard, sample=sample,
                concat_msg=incoming_vals.reshape(B * N, 2))
        else:  # tarmac: route neighbour one-hot + key from internal state
            doh, uoh = self._route(self._oh)
            nbr_oh = torch.stack([doh, uoh], dim=2).reshape(B * N, 2, self.vocab)
            dkey, ukey = self._route(self._key)
            nbr_key = torch.stack([dkey, ukey], dim=2).reshape(B * N, 2, self.Hk)
            q, msg_out, msg_idx, key_out, nh_out, msg_oh = self.agent(
                obs.reshape(B * N, -1), hiddens.reshape(B * N, -1),
                tau=tau, hard=hard, sample=sample,
                nbr_oh=nbr_oh, nbr_key=nbr_key, nbr_mask=mask)

        msg_out_r = msg_out.reshape(B, N, 1)
        safe_logs = msg_idx.reshape(B, N, 1).detach().cpu().numpy()

        # advance state for next step. KEEP attached during the differentiable
        # unroll (explicit msg_in); detach for single-step rollout.
        keep = using_explicit
        self._val = msg_out_r if keep else msg_out_r.detach()
        if self.reader == "tarmac":
            oh_r = msg_oh.reshape(B, N, self.vocab)
            key_r = key_out.reshape(B, N, self.Hk)
            self._oh = oh_r if keep else oh_r.detach()
            self._key = key_r if keep else key_r.detach()

        self.last_incoming_msgs = incoming_vals
        return (q.reshape(B, N, -1), nh_out.reshape(B, N, -1),
                msg_out_r, safe_logs, incoming_vals)


class QMixer(nn.Module):
    """Monotonic QMIX (Rashid et al. 2018). Unified signature -- computes the
    chosen Q internally so the trainer call matches the QPLEX mixer."""
    def __init__(self, n_agents, state_dim, mixing_embed_dim=256, hypernet_embed=64,
                 action_dim=None):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim
        self.hyper_w_1 = nn.Sequential(nn.Linear(state_dim, hypernet_embed), nn.ReLU(),
                                       nn.Linear(hypernet_embed, n_agents * mixing_embed_dim))
        self.hyper_w_2 = nn.Sequential(nn.Linear(state_dim, hypernet_embed), nn.ReLU(),
                                       nn.Linear(hypernet_embed, mixing_embed_dim))
        self.hyper_b_1 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_b_2 = nn.Sequential(nn.Linear(state_dim, mixing_embed_dim), nn.ReLU(),
                                       nn.Linear(mixing_embed_dim, 1))

    def forward(self, agent_qs, states, actions_onehot):
        M = agent_qs.size(0)
        chosen = (agent_qs * actions_onehot).sum(-1)              # [M, N]
        s = states.reshape(M, self.state_dim) / 100.0
        qs = chosen.view(M, 1, self.n_agents)
        w1 = torch.abs(self.hyper_w_1(s)).view(M, self.n_agents, self.mixing_embed_dim)
        b1 = self.hyper_b_1(s).view(M, 1, self.mixing_embed_dim)
        hidden = F.elu(torch.bmm(qs, w1) + b1)
        w2 = torch.abs(self.hyper_w_2(s)).view(M, self.mixing_embed_dim, 1)
        b2 = self.hyper_b_2(s).view(M, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(M, 1)


class _LambdaAttention(nn.Module):
    """Action-dependent positive advantage weights lambda_i>0 (qatten-style).
    Multi-head attention over agents; lambda_i = 1 + N * sum_h hw_h * softmax_i."""
    def __init__(self, n_agents, state_dim, action_dim, embed=32, n_heads=4):
        super().__init__()
        self.n_agents = n_agents; self.n_heads = n_heads; self.embed = embed
        self.key = nn.Linear(state_dim + action_dim, embed * n_heads)
        self.query = nn.Linear(state_dim, embed * n_heads)
        self.head_w = nn.Linear(state_dim, n_heads)

    def forward(self, states, actions_onehot):
        M = states.size(0); N = self.n_agents; H = self.n_heads; E = self.embed
        s_rep = states.unsqueeze(1).expand(M, N, states.size(-1))
        ka = torch.cat([s_rep, actions_onehot], dim=-1)          # [M,N,S+A]
        keys = self.key(ka).view(M, N, H, E).permute(0, 2, 1, 3) # [M,H,N,E]
        q = self.query(states).view(M, H, 1, E)                  # [M,H,1,E]
        scores = (keys * q).sum(-1) / math.sqrt(E)               # [M,H,N]
        attn = torch.softmax(scores, dim=-1)                     # [M,H,N]
        hw = torch.abs(self.head_w(states)).view(M, H, 1)        # [M,H,1] >=0
        lam = (hw * attn).sum(1) * N + 1.0                       # [M,N] > 0
        return lam


class QPLEXMixer(nn.Module):
    """QPLEX duplex-dueling mixer (Wang et al., ICLR 2021).
       Q_tot = sum_i w_i V_i + b  +  sum_i lambda_i(s,a) w_i A_i,
       with w_i>0, lambda_i>0, A_i = chosen_i - max_a Q_i <= 0  => full IGM class.
    """
    def __init__(self, n_agents, state_dim, mixing_embed_dim=64, hypernet_embed=64,
                 action_dim=None, n_heads=4):
        super().__init__()
        assert action_dim is not None, "QPLEXMixer needs action_dim"
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hyper_w = nn.Sequential(nn.Linear(state_dim, hypernet_embed), nn.ReLU(),
                                     nn.Linear(hypernet_embed, n_agents))
        self.hyper_b = nn.Sequential(nn.Linear(state_dim, hypernet_embed), nn.ReLU(),
                                     nn.Linear(hypernet_embed, 1))
        self.lam = _LambdaAttention(n_agents, state_dim, action_dim,
                                    embed=mixing_embed_dim, n_heads=n_heads)

    def forward(self, agent_qs, states, actions_onehot):
        M = agent_qs.size(0)
        s = states.reshape(M, self.state_dim) / 100.0
        V_i = agent_qs.max(dim=-1).values                        # [M,N]
        chosen = (agent_qs * actions_onehot).sum(-1)             # [M,N]
        A_i = chosen - V_i                                       # [M,N] <= 0
        w = torch.abs(self.hyper_w(s)) + 1e-6                    # [M,N] > 0
        b = self.hyper_b(s)                                      # [M,1]
        V_tot = (w * V_i).sum(-1, keepdim=True) + b
        lam = self.lam(s, actions_onehot)                       # [M,N] > 0
        A_tot = (lam * w * A_i).sum(-1, keepdim=True)
        return (V_tot + A_tot).view(M, 1)


def build_mixer(kind, n_agents, state_dim, action_dim, cfg):
    if kind == "qmix":
        return QMixer(n_agents, state_dim,
                      cfg.agent.get("mixing_embed_dim", 256),
                      cfg.agent.get("hypernet_embed", 64), action_dim=action_dim)
    if kind == "qplex":
        return QPLEXMixer(n_agents, state_dim,
                          cfg.agent.get("qplex_embed", 64),
                          cfg.agent.get("hypernet_embed", 64),
                          action_dim=action_dim,
                          n_heads=cfg.agent.get("qplex_heads", 4))
    raise ValueError(f"mixer must be qmix|qplex, got {kind}")