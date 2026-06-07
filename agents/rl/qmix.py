import torch
import torch.nn as nn
import torch.nn.functional as F
from .comm_utils import get_vocab_tensor

class QMixLocalAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions):
        super(QMixLocalAgent, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.value_stream = nn.Linear(hidden_dim, 1)
        self.advantage_stream = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs, hidden):
        # ENGINEER FIX: Scale observations to prevent GRU saturation
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
        self.msg_encoder = nn.Linear(1, hidden_dim // 2)
        
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        
        self.value_stream = nn.Linear(hidden_dim, 1)
        self.advantage_stream = nn.Linear(hidden_dim, n_actions)
        
        self.msg_stream = nn.Linear(hidden_dim, vocab_size)

    def forward(self, obs, msg_in, hidden, tau=1.0):
        # ENGINEER FIX: Scale observations to prevent GRU saturation
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
            # Add dummy index
            msg_indices = torch.zeros(h.size(0), 1, dtype=torch.long, device=obs.device) 
        else:
            msg_logits = self.msg_stream(h)
            msg_probs = F.gumbel_softmax(msg_logits, tau=tau, hard=True)
            vocab = get_vocab_tensor(self.vocab_size, obs.device)
            msg_out = (msg_probs * vocab).sum(dim=-1, keepdim=True)
            
            # ENGINEER FIX: Extract the discrete index for W&B Logging
            msg_indices = msg_probs.argmax(dim=-1, keepdim=True)
        
        # Return msg_indices as a new 3rd output
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

    def forward(self, obs, hiddens, tau=1.0, msg_in=None):
        B = obs.size(0)
        
        # ENGINEER FIX: Stateless Replay override
        if msg_in is not None:
            current_msgs = msg_in
        else:
            if B == 1:
                if self.rollout_msg_state is None or self.rollout_msg_state.device != obs.device:
                    self.rollout_msg_state = torch.zeros(1, self.num_agents, 1, device=obs.device)
                current_msgs = self.rollout_msg_state
            else:
                if self.msg_buffer is None or self.msg_buffer.size(0) != B:
                    self.msg_buffer = torch.zeros(B, self.num_agents, 1, device=obs.device)
                current_msgs = self.msg_buffer

        masked_msgs = torch.matmul(self.adj_mask, current_msgs)
        
        obs_flat = obs.view(B * self.num_agents, -1)
        msg_flat = masked_msgs.view(B * self.num_agents, -1)
        hiddens_flat = hiddens.view(B * self.num_agents, -1)
        
        # Extract the continuous msg_out and discrete msg_indices
        q_vals, msg_out, msg_indices, next_hiddens = self.agent(obs_flat, msg_flat, hiddens_flat, tau=tau)
        
        msg_out_reshaped = msg_out.view(B, self.num_agents, 1)
        safe_logs = msg_indices.view(B, self.num_agents, 1).detach().cpu().numpy()
        
        if B == 1:
            self.rollout_msg_state = msg_out_reshaped.detach()
        else:
            self.msg_buffer = msg_out_reshaped.detach()
        
        # ENGINEER FIX: Return BOTH the differentiable msg_out AND the safe_logs
        return q_vals.view(B, self.num_agents, -1), next_hiddens.view(B, self.num_agents, -1), msg_out_reshaped, safe_logs

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
        
        # ENGINEER FIX: Scale the global state for the hypernetwork
        scaled_states = states.reshape(-1, self.state_dim) / 100.0
        
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        
        w1 = torch.abs(self.hyper_w_1(scaled_states)).view(-1, self.n_agents, self.mixing_embed_dim)
        b1 = self.hyper_b_1(scaled_states).view(-1, 1, self.mixing_embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        
        w2 = torch.abs(self.hyper_w_2(scaled_states)).view(-1, self.mixing_embed_dim, 1)
        b2 = self.hyper_b_2(scaled_states).view(-1, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        
        return q_tot.view(batch_size, -1, 1)