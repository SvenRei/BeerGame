import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .comm_utils import get_vocab_tensor
except ImportError:  # Allows direct smoke tests from this file's folder.
    from comm_utils import get_vocab_tensor


class MessageDecoder(nn.Module):
    """NDQ expressiveness head (Wang et al., ICLR 2020, Eq. 3/7).

    A shared variational posterior q_xi(a_j | o_j, m_in_j) that predicts the
    RECEIVER's action from the receiver's own observation plus the INCOMING
    message. Trained with cross-entropy against the receiver's (detached)
    greedy action. Because the gradient w.r.t. m_in_j flows back into the
    sender's msg_stream (messages are differentiable during training), this
    pushes the channel to carry information that is predictive of the
    receiver's decision. A constant message cannot lower this loss below the
    obs-only baseline, so -- unlike a pure CIC/KL listening bonus -- it cannot
    be satisfied by a collapsed single-token channel.
    """

    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs, msg_in):
        # obs: [..., obs_dim] (already scaled), msg_in: [..., 1]
        return self.net(torch.cat([obs, msg_in], dim=-1))


class QMixLocalAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions):
        super(QMixLocalAgent, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.value_stream = nn.Linear(hidden_dim, 1)
        self.advantage_stream = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs, hidden):
        scaled_obs = obs / 100.0
        x = F.relu(self.fc1(scaled_obs))
        h_in = hidden.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        V = self.value_stream(h)
        A = self.advantage_stream(h)
        q_vals = V + A - A.mean(dim=1, keepdim=True)
        return q_vals, h


class CommQMixLocalAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions, vocab_size=3):
        super(CommQMixLocalAgent, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.obs_encoder = nn.Linear(input_dim, hidden_dim // 2)
        self.msg_encoder = nn.Linear(2, hidden_dim // 2)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.value_stream = nn.Linear(hidden_dim, 1)
        self.advantage_stream = nn.Linear(hidden_dim, n_actions)
        self.msg_stream = nn.Linear(hidden_dim, vocab_size)

    def forward(self, obs, msg_in, hidden, tau=1.0, hard=True, sample=True):
        scaled_obs = obs / 100.0
        obs_feat = F.relu(self.obs_encoder(scaled_obs))
        msg_feat = F.relu(self.msg_encoder(msg_in))

        x = torch.cat([obs_feat, msg_feat], dim=-1)
        h_in = hidden.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)

        V = self.value_stream(h)
        A = self.advantage_stream(h)
        q_vals = V + A - A.mean(dim=1, keepdim=True)

        if self.vocab_size == 1:
            msg_out = torch.zeros(h.size(0), 1, device=obs.device)
            msg_indices = torch.zeros(h.size(0), 1, dtype=torch.long, device=obs.device)
        else:
            msg_logits = self.msg_stream(h)
            if sample:
                msg_probs = F.gumbel_softmax(msg_logits, tau=tau, hard=hard)
            else:  # deterministic eval: argmax, no Gumbel noise
                idx = msg_logits.argmax(dim=-1, keepdim=True)
                msg_probs = torch.zeros_like(msg_logits).scatter_(-1, idx, 1.0)

        return q_vals, msg_out, msg_indices, h


class QMixCommMAC(nn.Module):
    def __init__(self, agent_network, num_agents=4):
        super().__init__()
        self.agent = agent_network
        self.num_agents = num_agents
        self.register_buffer("adj_mask", torch.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0]
        ]))
        self.msg_buffer = None
        self.rollout_msg_state = None

    def init_buffer(self, batch_size, device):
        self.msg_buffer = torch.zeros(batch_size, self.num_agents, 1, device=device)
        if batch_size == 1:
            self.rollout_msg_state = torch.zeros(1, self.num_agents, 1, device=device)

    def forward(self, obs, hiddens, tau=1.0, msg_in=None, hard=True):
        B = obs.size(0)

        using_explicit_msg = msg_in is not None
        if using_explicit_msg:
            current_msgs = msg_in
        else:
            if B == 1:
                if self.rollout_msg_state is None or self.rollout_msg_state.device != obs.device:
                    self.rollout_msg_state = torch.zeros(1, self.num_agents, 1, device=obs.device)
                current_msgs = self.rollout_msg_state
            else:
                if self.msg_buffer is None or self.msg_buffer.size(0) != B or self.msg_buffer.device != obs.device:
                    self.msg_buffer = torch.zeros(B, self.num_agents, 1, device=obs.device)
                current_msgs = self.msg_buffer

        Bq = current_msgs.size(0); dev = current_msgs.device
        down = torch.zeros(Bq, self.num_agents, 1, device=dev)
        up   = torch.zeros(Bq, self.num_agents, 1, device=dev)
        down[:, 1:, :] = current_msgs[:, :-1, :]   # message from downstream neighbour (i-1)
        up[:,  :-1, :] = current_msgs[:, 1:,  :]   # message from upstream neighbour  (i+1)
        masked_msgs = torch.cat([down, up], dim=-1)   # [B, N, 2]
        obs_flat = obs.reshape(B * self.num_agents, -1)
        msg_flat = masked_msgs.reshape(B * self.num_agents, -1)
        hiddens_flat = hiddens.reshape(B * self.num_agents, -1)

        q_vals, msg_out, msg_indices, next_hiddens = self.agent(obs_flat, msg_flat, hiddens_flat, tau=tau, hard=hard)
        msg_out_reshaped = msg_out.reshape(B, self.num_agents, 1)
        safe_logs = msg_indices.reshape(B, self.num_agents, 1).detach().cpu().numpy()

        if not using_explicit_msg:
            if B == 1:
                self.rollout_msg_state = msg_out_reshaped.detach()
            else:
                self.msg_buffer = msg_out_reshaped.detach()

        # NEW: also return the per-agent INCOMING message (after adjacency routing).
        # In training (explicit msg, soft Gumbel) this tensor is differentiable and
        # carries gradient back to the sender's msg_stream -- needed for the NDQ
        # expressiveness loss.
        self.last_incoming_msgs = masked_msgs.reshape(B, self.num_agents, 1)
        return q_vals.reshape(B, self.num_agents, -1), next_hiddens.reshape(B, self.num_agents, -1), msg_out_reshaped, safe_logs


class QMixer(nn.Module):
    def __init__(self, n_agents, state_dim, mixing_embed_dim=256, hypernet_embed=64):
        super(QMixer, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim

        self.hyper_w_1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed), nn.ReLU(),
            nn.Linear(hypernet_embed, n_agents * mixing_embed_dim)
        )
        self.hyper_w_2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed), nn.ReLU(),
            nn.Linear(hypernet_embed, mixing_embed_dim)
        )
        self.hyper_b_1 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_b_2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim), nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        batch_size = agent_qs.size(0)
        scaled_states = states.reshape(-1, self.state_dim) / 100.0
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        # QMIX monotonicity: nonnegative mixing weights keep decentralized argmax consistent.
        w1 = torch.abs(self.hyper_w_1(scaled_states)).view(-1, self.n_agents, self.mixing_embed_dim)
        b1 = self.hyper_b_1(scaled_states).view(-1, 1, self.mixing_embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        w2 = torch.abs(self.hyper_w_2(scaled_states)).view(-1, self.mixing_embed_dim, 1)
        b2 = self.hyper_b_2(scaled_states).view(-1, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(batch_size, -1, 1)