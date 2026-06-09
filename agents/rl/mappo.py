import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F

try:
    from .comm_utils import get_vocab_tensor
except ImportError:  # Allows this file to be run directly during smoke tests.
    from comm_utils import get_vocab_tensor


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.local_obs, self.global_states, self.hidden_states = [], [], []
        self.comm_in, self.actions, self.log_probs = [], [], []
        self.comm_actions = []
        self.rewards, self.is_terminals = [], []

    def __len__(self):
        return len(self.rewards)

    def push(self, obs, g_state, hidden, comm_in, action, log_prob, comm_action, reward, terminal):
        self.local_obs.append(obs)             # [num_agents, obs_dim]
        self.global_states.append(g_state)     # [1, global_dim]
        self.hidden_states.append(hidden)      # [num_agents, hidden_dim]
        self.comm_in.append(comm_in)           # [num_agents, 1]
        self.actions.append(action)            # [num_agents, 1] integer action index
        self.log_probs.append(log_prob)        # [num_agents, 1]
        self.comm_actions.append(comm_action)  # [num_agents, 1] integer comm token
        self.rewards.append(reward)            # [num_agents, 1]
        self.is_terminals.append(terminal)     # [num_agents, 1]


class MAPPOActor(nn.Module):
    def __init__(self, obs_dim, hidden_dim, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=False)

        # FEEDBACK 2.4 + 3.3: The old Normal policy sampled unbounded continuous actions
        # while the env clipped them, and MAPPO/QMIX used different executable action sets.
        # A categorical action head makes PPO log_probs match executed actions and shares
        # the exact same n_actions bins as QMIX.
        self.action_head = nn.Linear(hidden_dim, n_actions)
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.constant_(self.action_head.bias, 0.0)

    def forward(self, obs, hidden):
        scaled_obs = obs / 100.0
        x = self.fc(scaled_obs).unsqueeze(0)
        hidden = hidden.unsqueeze(0)
        x, hidden = self.gru(x, hidden)
        x = x.squeeze(0)
        hidden = hidden.squeeze(0)
        return Categorical(logits=self.action_head(x)), hidden

    def evaluate_actions(self, obs_seq, hidden_0, action_seq):
        """Recurrent PPO evaluation over a full rollout sequence."""
        scaled_obs_seq = obs_seq / 100.0
        x = self.fc(scaled_obs_seq)
        x, _ = self.gru(x, hidden_0)
        dist = Categorical(logits=self.action_head(x))
        action_log_probs = dist.log_prob(action_seq.squeeze(-1).long()).unsqueeze(-1)
        return action_log_probs, dist.entropy().mean()


class CommMAPPOActor(nn.Module):
    def __init__(self, obs_dim, hidden_dim, n_actions, vocab_size):
        super().__init__()
        self.n_actions = n_actions
        self.vocab_size = vocab_size
        comm_dim = 1

        self.fc = nn.Sequential(
            nn.Linear(obs_dim + comm_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=False)

        # FEEDBACK 2.4 + 3.3: Same discrete action bins as QMIX.
        self.action_head = nn.Linear(hidden_dim, n_actions)
        self.comm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.constant_(self.action_head.bias, 0.0)

    def forward(self, obs, comm_in, hidden, tau=1.0):
        scaled_obs = obs / 100.0
        x = torch.cat([scaled_obs, comm_in], dim=-1)
        x = self.fc(x).unsqueeze(0)
        hidden = hidden.unsqueeze(0)
        x, hidden = self.gru(x, hidden)

        x_flat = x.squeeze(0)
        hidden = hidden.squeeze(0)
        dist_action = Categorical(logits=self.action_head(x_flat))

        msg_logits = self.comm_head(x_flat)
        dist_comm = Categorical(logits=msg_logits)
        msg_probs = F.gumbel_softmax(msg_logits, tau=tau, hard=True)
        return dist_action, dist_comm, msg_probs, hidden

    def evaluate_actions(self, obs_seq, comm_in_seq, hidden_0, action_seq, comm_action_seq):
        scaled_obs_seq = obs_seq / 100.0
        x = torch.cat([scaled_obs_seq, comm_in_seq], dim=-1)
        x = self.fc(x)
        x, _ = self.gru(x, hidden_0)

        dist_action = Categorical(logits=self.action_head(x))
        msg_logits = self.comm_head(x)
        dist_comm = Categorical(logits=msg_logits)

        action_log_probs = dist_action.log_prob(action_seq.squeeze(-1).long()).unsqueeze(-1)
        comm_log_probs = dist_comm.log_prob(comm_action_seq.squeeze(-1).long())
        entropy = dist_action.entropy().mean() + dist_comm.entropy().mean()
        return action_log_probs, comm_log_probs, dist_comm.probs, entropy


class MAPPOCommMAC(nn.Module):
    def __init__(self, actor_network, vocab_size=3, num_agents=4):
        super().__init__()
        self.actor = actor_network
        self.num_agents = num_agents
        self.vocab_size = vocab_size

        self.register_buffer("adj_mask", torch.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0]
        ]))
        self.msg_buffer = None

    def init_buffer(self, batch_size, device):
        self.msg_buffer = torch.zeros(batch_size, self.num_agents, 1, device=device)

    def evaluate_actions_bptt(self, obs_seq, hidden_0, action_seq, comm_action_seq):
        """Unrolls the BPTT loop to allow gradients to flow through communication."""
        T, N, _ = obs_seq.shape
        vocab_tensor = get_vocab_tensor(self.vocab_size, obs_seq.device)

        hiddens = hidden_0.view(N, -1)
        msg_buffer = torch.zeros(N, 1, device=obs_seq.device)

        action_log_probs_list = []
        comm_log_probs_list = []
        entropy_list = []

        for t in range(T):
            masked_msgs = torch.matmul(self.adj_mask, msg_buffer)

            # Pass through base actor; use tau=1.0 for smooth gradients during BPTT eval
            dist_action, dist_comm, msg_probs, next_hiddens = self.actor(
                obs_seq[t], masked_msgs, hiddens, tau=1.0 
            )

            # Evaluate the actions that were ACTUALLY taken during the rollout
            act_log_prob = dist_action.log_prob(action_seq[t].squeeze(-1).long()).unsqueeze(-1)
            comm_log_prob = dist_comm.log_prob(comm_action_seq[t].squeeze(-1).long())
            entropy = dist_action.entropy().mean() + dist_comm.entropy().mean()

            action_log_probs_list.append(act_log_prob)
            comm_log_probs_list.append(comm_log_prob)
            entropy_list.append(entropy)

            # Generate differentiable message for the NEXT timestep
            # The straight-through estimator handles the gradient pass
            msg_out = (msg_probs * vocab_tensor).sum(dim=-1, keepdim=True)
            
            # DO NOT DETACH THIS TENSOR! This wires the graph across timesteps.
            msg_buffer = msg_out  

            hiddens = next_hiddens

        return torch.stack(action_log_probs_list), torch.stack(comm_log_probs_list), torch.stack(entropy_list).mean()

    def forward(self, obs, hiddens, tau=1.0, test_mode=False):
        B = obs.size(0)
        if self.msg_buffer is None or self.msg_buffer.size(0) != B or self.msg_buffer.device != obs.device:
            self.init_buffer(B, obs.device)

        vocab_tensor = get_vocab_tensor(self.vocab_size, obs.device)
        masked_msgs = torch.matmul(self.adj_mask, self.msg_buffer)

        obs_flat = obs.view(B * self.num_agents, -1)
        msg_flat = masked_msgs.view(B * self.num_agents, -1)
        hiddens_flat = hiddens.view(B * self.num_agents, -1)

        dist_action, dist_comm, msg_probs, next_hiddens = self.actor(obs_flat, msg_flat, hiddens_flat, tau=tau)

        if test_mode:
            comm_actions = dist_comm.probs.argmax(dim=-1)
        else:
            comm_actions = dist_comm.sample()

        # FEEDBACK 5: Store the discrete token sent during rollout for on-policy comm log_probs.
        comm_out = vocab_tensor[comm_actions].unsqueeze(-1)
        self.msg_buffer = comm_out.view(B, self.num_agents, 1).detach().clone()
        safe_logs = comm_actions.view(B, self.num_agents, 1).detach().cpu().numpy()

        return (
            dist_action,
            dist_comm,
            comm_actions,
            next_hiddens.view(B, self.num_agents, -1),
            masked_msgs.view(B, self.num_agents, 1),
            safe_logs,
        )


class MAPPOCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.fc(state / 100.0)


class MAPPOTrainer:
    def __init__(self, actor, critic, cfg, total_episodes, device, algo):
        self.actor, self.critic, self.device, self.algo = actor, critic, device, algo
        self.gamma = cfg.get("gamma", 0.99)
        self.entropy_coef = cfg.get("entropy_coef", 0.05)
        self.comm_penalty_coef = cfg.get("comm_penalty_coef", 0.0)
        self.k_epochs = cfg.get("k_epochs", 4)

        # FEEDBACK 2.6: Use configured PPO clipping epsilon instead of hardcoded 0.8/1.2.
        self.eps_clip = cfg.get("eps_clip", 0.2)

        self.warm_up_episodes = cfg.get("warm_up_episodes", 1000)
        self.total_episodes = total_episodes
        self.tau_start = 1.0
        self.tau_min = 0.1
        self.tau_decay_episodes = cfg.get("tau_decay_episodes", total_episodes * 0.5)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.get("lr_actor", 3e-4))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.get("lr_critic", 1e-3))
        self.max_grad_norm = cfg.get("max_grad_norm", 0.5)

    def get_current_tau(self, current_ep):
        if current_ep >= self.tau_decay_episodes:
            return self.tau_min
        return self.tau_start - (self.tau_start - self.tau_min) * (current_ep / self.tau_decay_episodes)

    def update(self, buffer, current_ep):
        # FEEDBACK 2.1/2.3: This update is intended for a full episode or rollout chunk, not one step.
        if len(buffer) == 0:
            return 0.0, 0.0

        num_agents = 4
        obs_seq = torch.stack(buffer.local_obs).detach()          # [T, N, obs_dim]
        actions_seq = torch.stack(buffer.actions).detach()       # [T, N, 1] integer indices
        old_log_probs_seq = torch.stack(buffer.log_probs).detach()
        rewards_seq = torch.stack(buffer.rewards).detach()       # [T, N, 1]
        terminals_seq = torch.stack(buffer.is_terminals).detach()
        hidden_0 = torch.stack(buffer.hidden_states)[0].unsqueeze(0).detach()

        T = obs_seq.size(0)
        g_states_raw = torch.stack(buffer.global_states).detach()  # [T, 1, global_dim]
        g_states_seq = g_states_raw.repeat(1, num_agents, 1)       # [T, N, global_dim]

        # FEEDBACK 2.2: terminals_seq contains terminations OR truncations, so returns stop at horizons.
        returns = torch.zeros_like(rewards_seq)
        next_return = torch.zeros_like(rewards_seq[0])
        for t in reversed(range(T)):
            next_return = rewards_seq[t] + self.gamma * next_return * (1.0 - terminals_seq[t])
            returns[t] = next_return

        if self.algo == "ippo":
            values_flat = self.critic(obs_seq.reshape(T * num_agents, -1))
        else:
            values_flat = self.critic(g_states_seq.reshape(T * num_agents, -1))
        values_seq = values_flat.view(T, num_agents, 1)

        advantages_seq = returns - values_seq.detach()
        advantages_seq = (advantages_seq - advantages_seq.mean()) / (advantages_seq.std() + 1e-8)

        actor_loss = torch.tensor(0.0, device=self.device)
        critic_loss = torch.tensor(0.0, device=self.device)

        for _ in range(self.k_epochs):
            if self.algo == "comm_mappo":
                comm_actions_seq = torch.stack(buffer.comm_actions).detach()
                comm_in_seq = torch.stack(buffer.comm_in).detach()  # [T, N, 1] actual rollout messages

                # Batched recurrent eval conditioned on the messages that ACTUALLY flowed
                # during rollout. One GRU call over the whole sequence instead of a T-step
                # loop. self.actor is the MAC; self.actor.actor is the CommMAPPOActor.
                act_log_probs, comm_log_probs, _comm_probs, entropy = self.actor.actor.evaluate_actions(
                    obs_seq, comm_in_seq, hidden_0, actions_seq, comm_actions_seq
                )
                current_log_probs_seq = act_log_probs + comm_log_probs.unsqueeze(-1)
            else:
                act_log_probs, entropy = self.actor.evaluate_actions(obs_seq, hidden_0, actions_seq)
                current_log_probs_seq = act_log_probs

            ratios = torch.exp(current_log_probs_seq - old_log_probs_seq)
            surr1 = ratios * advantages_seq
            surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages_seq
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            if self.algo == "ippo":
                values_flat = self.critic(obs_seq.reshape(T * num_agents, -1))
            else:
                values_flat = self.critic(g_states_seq.reshape(T * num_agents, -1))
            critic_loss = torch.nn.MSELoss()(values_flat, returns.reshape(T * num_agents, 1))

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

        buffer.clear()
        return actor_loss.item(), critic_loss.item()
