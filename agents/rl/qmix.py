import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. THE LOCAL AGENT NETWORK (Vanilla QMIX)
# ==============================================================================
class QMixLocalAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions):
        super(QMixLocalAgent, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Feature Extraction
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Recurrent Memory (GRU)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Dueling Streams
        self.value_stream = nn.Linear(hidden_dim, 1)
        self.advantage_stream = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs, hidden):
        x = F.relu(self.fc1(obs))
        h_in = hidden.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        
        V = self.value_stream(h)
        A = self.advantage_stream(h)
        # Dueling aggregation
        q_vals = V + A - A.mean(dim=1, keepdim=True)
        
        return q_vals, h


# ==============================================================================
# 2. THE COMMUNICATIVE AGENT NETWORK (DIAL QMIX)
# ==============================================================================
class CommQMixLocalAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions):
        super(CommQMixLocalAgent, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Feature extraction now takes the Observation + 1 Incoming Message
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Dueling physical action streams
        self.value_stream = nn.Linear(hidden_dim, 1)
        self.advantage_stream = nn.Linear(hidden_dim, n_actions)
        
        # DIAL Message Generator: Outputs logits for 3 discrete tokens (-1, 0, 1)
        self.msg_stream = nn.Linear(hidden_dim, 3)

    def forward(self, obs, msg_in, hidden):
        # 1. Fuse the local physical observation with the latent signal from upstream
        x = torch.cat([obs, msg_in], dim=-1)
        x = F.relu(self.fc1(x))
        
        h_in = hidden.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        
        # 2. Generate Physical Q-Values (Dueling)
        V = self.value_stream(h)
        A = self.advantage_stream(h)
        q_vals = V + A - A.mean(dim=1, keepdim=True)
        
        # 3. Generate Differentiable Latent Message
        msg_logits = self.msg_stream(h)
        # Apply Gumbel-Softmax. hard=True ensures the output is a strict one-hot vector,
        # but the Straight-Through Estimator allows continuous gradients to pass backward.
        msg_probs = F.gumbel_softmax(msg_logits, tau=1.0, hard=True)
        
        # Map the one-hot selection to your literal vocabulary mapping
        vocab = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32, device=obs.device)
        # Multiply the one-hot probability by the vocab to get a single discrete float
        msg_out = (msg_probs * vocab).sum(dim=-1, keepdim=True)
        
        return q_vals, msg_out, h


# ==============================================================================
# 3. THE MIXING HYPERNETWORK (Shared by both Vanilla and Comm-QMIX)
# ==============================================================================
class QMixer(nn.Module):
    def __init__(self, n_agents, state_dim, mixing_embed_dim=256, hypernet_embed=64):
        super(QMixer, self).__init__()
        
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim
        
        # Hypernetwork 1
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, n_agents * mixing_embed_dim)
        )
        
        # Hypernetwork 2
        self.hyper_w_2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, mixing_embed_dim)
        )
        
        # Bias Generators
        self.hyper_b_1 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_b_2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        batch_size = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        
        # Layer 1 Mixing (Enforcing strict monotonicity with absolute value)
        w1 = torch.abs(self.hyper_w_1(states)).view(-1, self.n_agents, self.mixing_embed_dim)
        b1 = self.hyper_b_1(states).view(-1, 1, self.mixing_embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        
        # Layer 2 Mixing
        w2 = torch.abs(self.hyper_w_2(states)).view(-1, self.mixing_embed_dim, 1)
        b2 = self.hyper_b_2(states).view(-1, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        
        return q_tot.view(batch_size, -1, 1)