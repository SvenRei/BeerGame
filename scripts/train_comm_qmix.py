import io
import os
import sys
import random
from collections import deque

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict


def _torch_save(obj, path, _retries=6, _delay=5):
    """Save via BytesIO with retry so OneDrive sync locks don't crash training."""
    import time
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = io.BytesIO()
    torch.save(obj, buf)
    data = buf.getvalue()
    for attempt in range(_retries):
        try:
            with open(path, "wb") as f:
                f.write(data)
            return
        except PermissionError:
            if attempt < _retries - 1:
                time.sleep(_delay)
            else:
                raise

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.beer_game_env import BeerGameParallelEnv
from agents.action_space import index_to_fraction
from agents.rl.qmix import CommQMixLocalAgent, QMixCommMAC, QMixer, MessageDecoder


class EpisodeReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, states, obs, actions, rewards, dones):
        self.buffer.append({
            "states": np.array(states, dtype=np.float32),
            "obs": np.array(obs, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
        })

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return {k: torch.tensor(np.stack([b[k] for b in batch])) for k in batch[0].keys()}

    def __len__(self):
        return len(self.buffer)


def set_global_seeds(seed):
    # FEEDBACK 7.2: Seed all RNG sources used by epsilon-greedy, Gumbel, and networks.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def update_comm_qmix(batch, mac, target_mac, mixer, target_mixer, optimizer, all_params, cfg, device, hidden_dim, tau, msg_decoder=None):
    b_states = batch["states"].to(device)
    b_obs = batch["obs"].to(device)
    b_actions = batch["actions"].to(device)
    b_rewards = batch["rewards"].to(device)
    b_dones = batch["dones"].to(device)

    B, T_plus_1, N, _ = b_obs.shape
    T = T_plus_1 - 1

    h_train = torch.zeros(B, N, hidden_dim, device=device)
    target_h_train = torch.zeros(B, N, hidden_dim, device=device)
    msg_in = torch.zeros(B, N, 1, device=device)
    target_msg_in = torch.zeros(B, N, 1, device=device)

    q_evals_list, target_q_evals_list = [], []
    msg_out_list = []
    inc_msg_list = []  # NEW: incoming (post-adjacency) message per agent, per step

    # Online network: hard=False keeps Gumbel-softmax differentiable so gradients
    # flow back through msg_stream. Target network runs inside no_grad.
    for t in range(T_plus_1):
        q_t, h_train, msg_out, _, inc_msg = mac(b_obs[:, t], h_train, tau=tau, msg_in=msg_in, hard=False)
        msg_in = msg_out  # keep gradient flow through communication
        q_evals_list.append(q_t)
        msg_out_list.append(msg_out)
        inc_msg_list.append(inc_msg)  # [B, N, 1], differentiable -> sender's msg_stream

        with torch.no_grad():
            target_q_t, target_h_train, target_msg_out, _, _ = target_mac(b_obs[:, t], target_h_train, tau=tau, msg_in=target_msg_in, hard=False)
            target_msg_in = target_msg_out
            target_q_evals_list.append(target_q_t)

    q_evals = torch.stack(q_evals_list, dim=1)             # [B, T+1, N, A]
    target_q_evals = torch.stack(target_q_evals_list, dim=1)
    
    chosen_q = q_evals[:, :-1].gather(3, b_actions)       # [B, T, N, 1]
    best_next_actions = q_evals[:, 1:].argmax(dim=3, keepdim=True)
    target_q_gathered = target_q_evals[:, 1:].gather(3, best_next_actions)

    b_states_t = b_states[:, :-1, :]
    b_states_next = b_states[:, 1:, :]
    q_tot = mixer(chosen_q.reshape(B * T, N, 1), b_states_t.reshape(B * T, -1))
    with torch.no_grad():
        target_q_tot = target_mixer(target_q_gathered.reshape(B * T, N, 1), b_states_next.reshape(B * T, -1))

    q_tot_flat = q_tot.reshape(B * T, 1)
    targets = b_rewards.reshape(B * T, 1) + cfg.agent.gamma * (1.0 - b_dones.reshape(B * T, 1)) * target_q_tot.reshape(B * T, 1)
    base_loss = nn.MSELoss()(q_tot_flat, targets.detach())
    msg_outs_tensor = torch.stack(msg_out_list, dim=1) # Shape: [B, T+1, N, 1]
    comm_penalty = cfg.agent.comm_penalty_coef * (msg_outs_tensor ** 2).mean()

    listening_coef = cfg.agent.get("listening_coef", 0.0)
    # ------------------------------------------------------------------
    # NDQ EXPRESSIVENESS ("listening") LOSS  (Wang et al., ICLR 2020, Eq. 3/7)
    #
    # We REPLACE the old CIC/KL positive-listening bonus. That objective rewarded
    # ANY divergence between the with-message and zero-message policy, so a
    # CONSTANT (information-free) message could maximise it -> single-token
    # collapse. Here instead we train a shared decoder q_xi(a_j | o_j, m_in_j) to
    # predict the receiver's greedy action from its own obs + the incoming
    # message, and minimise the cross-entropy. The gradient w.r.t. the incoming
    # message flows into the sender's msg_stream, so the channel is pushed to
    # carry receiver-decision-relevant information. A constant message cannot
    # beat the obs-only baseline, so this is NOT gameable by collapse.
    #
    # NOTE: in QMIX the mixer already sees the global state, so the TD loss alone
    # gives ~0 gradient to the message head (messages are redundant for fitting
    # Q_tot). This decentralised auxiliary loss is what actually trains the channel.
    # ------------------------------------------------------------------
    expr_loss = torch.tensor(0.0, device=device)
    if listening_coef > 0.0 and cfg.agent.get("vocab_size", 3) > 1 and msg_decoder is not None:
        inc_msgs = torch.stack(inc_msg_list, dim=1)[:, :-1]      # [B, T, N, 1] differentiable
        obs_scaled = b_obs[:, :-1] / 100.0                       # [B, T, N, obs] (match agent's /100 scaling)
        logits = msg_decoder(obs_scaled, inc_msgs)              # [B, T, N, A]
        with torch.no_grad():
            # p(A_j | .) proxy: receiver's own informed (with-message) greedy action.
            teacher = q_evals[:, :-1].argmax(dim=-1)            # [B, T, N]
        A = logits.size(-1)
        expr_loss = nn.functional.cross_entropy(
            logits.reshape(-1, A), teacher.reshape(-1)
        )

    # Minimise expressiveness CE (note: + sign, unlike the old maximised KL term).
    loss = base_loss + comm_penalty + listening_coef * expr_loss

    optimizer.zero_grad()
    loss.backward()

    # FEEDBACK 5.2/5.3: Track whether communication parameters receive learning signal.
    msg_grad_norm = 0.0
    for name, param in mac.named_parameters():
        if "msg_stream" in name and param.grad is not None:
            msg_grad_norm += float(param.grad.detach().norm().cpu().item())

    torch.nn.utils.clip_grad_norm_(all_params, 5.0)
    optimizer.step()
    return loss.item(), msg_grad_norm, float(expr_loss.detach().item())


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    base_seed = cfg.get("seed", 1000)
    set_global_seeds(base_seed)

    run = wandb.init(project="BeerGame_Research", name="comm_qmix")
    # Sweep optimizes the run summary of Avg_Cost_50; take the BEST (min) over the run,
    # not the noisy last value (critical for the high-variance QMIX family).
    wandb.define_metric("Avg_Cost_50", summary="min")
    wandb.define_metric("Avg_Cost_500", summary="last")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    #with open_dict(cfg):
    #    for key in ["lr", "target_update_freq", "batch_size", "vocab_size", "hidden_dim", "n_actions", "lr_scheduler_step", "lr_scheduler_gamma"]:
    #        if key in wandb.config:
    #            cfg.agent[key] = wandb.config[key]
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)

    vocab_size = cfg.agent.get("vocab_size", 3)

    env = BeerGameParallelEnv(cfg.env)
    obs, _ = env.reset(seed=base_seed)
    state_dim = len(env.get_global_state())

    run_dir = os.path.join(_PROJECT_ROOT, "weights_comm_qmix", f"run_{run.name}_{run.id}")
    os.makedirs(run_dir, exist_ok=True)

    local_dim = env.observation_space("retailer").shape[0]
    n_actions = cfg.agent.n_actions
    if n_actions < 2:
        raise ValueError(f"n_actions must be >= 2, got {n_actions}")

    # Action parameterization (see agents/action_space.py). Swept over {absolute, centered}.
    action_mode = cfg.agent.get("action_mode", "absolute")
    abs_cap = cfg.agent.get("abs_cap", env.max_order)
    centered_range = cfg.agent.get("centered_range", 10)

    hidden_dim = cfg.agent.hidden_dim
    num_agents = len(env.possible_agents)
    agent_names = env.possible_agents

    base_agent = CommQMixLocalAgent(local_dim, hidden_dim, n_actions, vocab_size=vocab_size)
    mac = QMixCommMAC(base_agent, num_agents=num_agents).to(device)
    mixer = QMixer(num_agents, state_dim, cfg.agent.mixing_embed_dim, cfg.agent.hypernet_embed).to(device)

    target_base = CommQMixLocalAgent(local_dim, hidden_dim, n_actions, vocab_size=vocab_size)
    target_mac = QMixCommMAC(target_base, num_agents=num_agents).to(device)
    target_mixer = QMixer(num_agents, state_dim, cfg.agent.mixing_embed_dim, cfg.agent.hypernet_embed).to(device)

    target_mac.load_state_dict(mac.state_dict())
    target_mixer.load_state_dict(mixer.state_dict())

    # NDQ expressiveness decoder q_xi(a_j | o_j, m_in_j). Shared across agents
    # (like NDQ's shared variational posterior). Trained jointly via the optimizer.
    expr_decoder_hidden = cfg.agent.get("expr_decoder_hidden", 128)
    msg_decoder = MessageDecoder(local_dim, n_actions, hidden=expr_decoder_hidden).to(device)

    all_params = list(mixer.parameters()) + list(mac.parameters()) + list(msg_decoder.parameters())
    optimizer = optim.Adam(all_params, lr=cfg.agent.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.agent.lr_scheduler_step, gamma=cfg.agent.lr_scheduler_gamma)
    buffer = EpisodeReplayBuffer(cfg.agent.buffer_size)

    patience = cfg.agent.get("patience", 2000)
    warm_up = cfg.agent.get("warm_up_episodes", 1000)
    eps_decay_eps = cfg.agent.get("epsilon_decay_episodes", 5000)
    cost_history = deque(maxlen=50)
    cost_history_500 = deque(maxlen=500)
    best_avg_cost, since_imp, global_step = float("inf"), 0, 0
    epsilon = cfg.agent.epsilon_start
    last_loss, last_msg_grad_norm, last_listening_kl = 0.0, 0.0, 0.0

    tau_start, tau_min = 1.0, cfg.agent.get("tau_min", 0.1)
    tau_decay_episodes = cfg.agent.get("tau_decay_episodes", cfg.total_episodes * 0.5)

    print("--- Starting COMM_QMIX Sequence Replay Training ---")

    for ep in range(cfg.total_episodes):
        tau = tau_min if ep >= tau_decay_episodes else tau_start - (tau_start - tau_min) * (ep / tau_decay_episodes)
        obs, _ = env.reset(seed=base_seed + ep)
        mac.init_buffer(batch_size=1, device=device)
        hiddens_tensor = torch.zeros(1, num_agents, hidden_dim, device=device)

        ep_states, ep_obs, ep_actions, ep_rewards, ep_dones = [], [], [], [], []
        ep_cost = 0.0
        ep_agent_costs = {a: 0.0 for a in agent_names}
        episode_messages = []

        ep_obs.append(np.stack([obs[a] for a in agent_names]))
        ep_states.append(env.get_global_state())

        while True:
            obs_array = np.stack([obs[a] for a in agent_names])
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                q_vals, next_hiddens, _, safe_logs, _ = mac(obs_tensor, hiddens_tensor.detach(), tau=tau)
                episode_messages.append(safe_logs)

            env_acts, actions_list = {}, []
            for i, a in enumerate(agent_names):
                if random.random() < epsilon:
                    action_idx = random.randint(0, n_actions - 1)
                else:
                    action_idx = q_vals[0, i].argmax(dim=-1).item()
                actions_list.append([action_idx])
                env_acts[a] = [index_to_fraction(
                    action_idx, n_actions=n_actions, max_order=env.max_order, mode=action_mode,
                    abs_cap=abs_cap, centered_range=centered_range, demand_anchor=float(obs[a][3]),
                )]

            if env.current_step % 10 == 0:
                wandb.log({f"Order_Qty/{a}": float(np.round(env_acts[a][0] * env.max_order)) for a in agent_names}, commit=False)

            next_obs, rewards, terms, truncs, infos = env.step(env_acts)
            raw_cost = 0.0
            for a in agent_names:
                local_cost = infos[a]["local_cost"]
                raw_cost += local_cost
                ep_agent_costs[a] += local_cost
            ep_cost += raw_cost

            global_reward = -raw_cost / cfg.agent.get("reward_scale", 100.0)
            done = any(terms.values()) or any(truncs.values())

            ep_actions.append(actions_list)
            ep_rewards.append([global_reward])
            ep_dones.append([float(done)])
            ep_obs.append(np.stack([next_obs[a] for a in agent_names]))
            ep_states.append(env.get_global_state())

            obs = next_obs
            hiddens_tensor = next_hiddens.detach()
            global_step += 1
            if done:
                break

        buffer.push(ep_states, ep_obs, ep_actions, ep_rewards, ep_dones)

        if len(buffer) >= cfg.agent.batch_size and ep >= warm_up:
            updates_per_episode = cfg.agent.get("updates_per_episode", 1)
            for _ in range(updates_per_episode):
                batch = buffer.sample(cfg.agent.batch_size)
                last_loss, last_msg_grad_norm, last_listening_kl = update_comm_qmix(
                    batch, mac, target_mac, mixer, target_mixer, optimizer, all_params, cfg, device, hidden_dim, tau, msg_decoder=msg_decoder
                )

            if ep % cfg.agent.target_update_freq == 0 and ep > 0:
                target_mac.load_state_dict(mac.state_dict())
                target_mixer.load_state_dict(mixer.state_dict())

            scheduler.step()
        decay_step = (cfg.agent.epsilon_start - cfg.agent.epsilon_end) / max(1, eps_decay_eps)
        epsilon = max(cfg.agent.epsilon_end, cfg.agent.epsilon_start - decay_step * (ep + 1))

        cost_history.append(ep_cost)
        cost_history_500.append(ep_cost)
        avg_cost_500 = sum(cost_history_500) / len(cost_history_500)
        avg_cost = sum(cost_history) / len(cost_history)
        log_dict = {
            "Cost": ep_cost,
            "Avg_Cost_50": avg_cost,
            "Avg_Cost_500": avg_cost_500,
            "Epsilon": epsilon,
            "Tau": tau,
            "LR": scheduler.get_last_lr()[0],
            "Loss": last_loss,
            "Comm/Msg_Grad_Norm": last_msg_grad_norm,
            "Comm/Listening_KL": last_listening_kl,        # now holds the NDQ expressiveness CE (lower = better)
            "Comm/Expressiveness_CE": last_listening_kl
        }
        for a, cost in ep_agent_costs.items():
            log_dict[f"Cost/{a}"] = cost

        if len(episode_messages) > 0:
            all_msgs = np.concatenate(episode_messages, axis=0).flatten().astype(int)
            log_dict["Comm/Message_Distribution"] = wandb.Histogram(all_msgs)
            log_dict["Comm/Unique_Tokens"] = len(np.unique(all_msgs))
            token_counts = np.bincount(all_msgs, minlength=vocab_size)
            token_pcts = (token_counts / max(1, len(all_msgs))) * 100.0
            for v in range(vocab_size):
                log_dict[f"Comm/Token_{v}_Pct"] = token_pcts[v]

        wandb.log(log_dict)

        if ep == warm_up:
            best_avg_cost, since_imp = float("inf"), 0

        if avg_cost < best_avg_cost and len(cost_history) == 50:
            best_avg_cost = avg_cost
            since_imp = 0
            if ep >= warm_up:
                # FEEDBACK 7.3: Save full checkpoint plus MAC-only state for old evaluation code.
                checkpoint = {
                    "mac": mac.state_dict(),
                    "mixer": mixer.state_dict(),
                    "target_mac": target_mac.state_dict(),
                    "target_mixer": target_mixer.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "episode": ep,
                    "epsilon": epsilon,
                    "tau": tau,
                    "best_avg_cost": best_avg_cost,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                }
                _torch_save(checkpoint, os.path.join(run_dir, "comm_qmix_checkpoint_best.pt"))
                _torch_save(mac.state_dict(), os.path.join(run_dir, "comm_qmix_mac_best.pth"))
                if len(episode_messages) > 0:
                    np.save(os.path.join(run_dir, f"best_messages_ep_{ep}.npy"), np.concatenate(episode_messages, axis=0))
        else:
            if ep >= warm_up:
                since_imp += 1

        exploration_lock = max(warm_up, eps_decay_eps)
        if ep > exploration_lock and since_imp >= patience:
            break
        if ep % 10 == 0:
            print(f"Ep {ep} | Cost: {ep_cost:.2f} | 50-Ep Avg: {avg_cost:.2f} | Best: {best_avg_cost if best_avg_cost != float('inf') else 0.0:.2f} | Eps: {epsilon:.2f} | Tau: {tau:.2f}")

    wandb.finish()


if __name__ == "__main__":
    main()