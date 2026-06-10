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
from omegaconf import DictConfig, OmegaConf


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
from agents.rl.qmix import QMixLocalAgent, QMixer


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
    # FEEDBACK 7.2: Seed all RNG sources used by epsilon-greedy and networks.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def update_qmix(batch, mac, target_mac, mixer, target_mixer, optimizer, all_params, agent_names, cfg, device, hidden_dim):
    b_states = batch["states"].to(device)      # [B, T+1, state_dim]
    b_obs = batch["obs"].to(device)            # [B, T+1, N, obs_dim]
    b_actions = batch["actions"].to(device)    # [B, T, N, 1]
    b_rewards = batch["rewards"].to(device)    # [B, T, 1]
    b_dones = batch["dones"].to(device)        # [B, T, 1]

    B, T_plus_1, N, _ = b_obs.shape
    T = T_plus_1 - 1
    q_evals_agents, target_q_evals_agents = [], []

    # FEEDBACK 4.2: Train recurrent QMIX from complete sequences, not isolated transitions.
    for i, agent_name in enumerate(agent_names):
        h_train = torch.zeros(B, hidden_dim, device=device)
        target_h_train = torch.zeros(B, hidden_dim, device=device)
        q_agent, target_q_agent = [], []

        for t in range(T_plus_1):
            q, h_train = mac[agent_name](b_obs[:, t, i, :], h_train)
            q_agent.append(q)
            with torch.no_grad():
                target_q, target_h_train = target_mac[agent_name](b_obs[:, t, i, :], target_h_train)
                target_q_agent.append(target_q)

        q_agent = torch.stack(q_agent, dim=1)
        target_q_agent = torch.stack(target_q_agent, dim=1)

        chosen_q = q_agent[:, :-1, :].gather(2, b_actions[:, :, i, :])
        best_next_actions = q_agent[:, 1:, :].argmax(dim=2, keepdim=True)
        target_q_gathered = target_q_agent[:, 1:, :].gather(2, best_next_actions)
        q_evals_agents.append(chosen_q)
        target_q_evals_agents.append(target_q_gathered)

    q_evals = torch.cat(q_evals_agents, dim=2)              # [B, T, N]
    target_q_evals = torch.cat(target_q_evals_agents, dim=2)
    b_states_t = b_states[:, :-1, :]
    b_states_next = b_states[:, 1:, :]

    q_tot = mixer(q_evals.reshape(B * T, N, 1), b_states_t.reshape(B * T, -1))
    with torch.no_grad():
        target_q_tot = target_mixer(target_q_evals.reshape(B * T, N, 1), b_states_next.reshape(B * T, -1))

    q_tot_flat = q_tot.reshape(B * T, 1)
    targets = b_rewards.reshape(B * T, 1) + cfg.agent.gamma * (1.0 - b_dones.reshape(B * T, 1)) * target_q_tot.reshape(B * T, 1)
    loss = nn.MSELoss()(q_tot_flat, targets.detach())

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(all_params, 5.0)
    optimizer.step()
    return loss.item()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    base_seed = cfg.get("seed", 1000)
    set_global_seeds(base_seed)

    run = wandb.init(project="BeerGame_Research", config=OmegaConf.to_container(cfg, resolve=True), name="qmix")
    # Sweep optimizes the run summary of Avg_Cost_50; take the BEST (min) over the run,
    # not the noisy last value (critical for the high-variance QMIX family).
    wandb.define_metric("Avg_Cost_50", summary="min")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # FEEDBACK 3.1: Do not override demand_type; benchmark environment comes from config.
    env = BeerGameParallelEnv(cfg.env)
    obs, _ = env.reset(seed=base_seed)
    state_dim = len(env.get_global_state())

    run_dir = os.path.join(_PROJECT_ROOT, "weights_qmix", f"run_{run.name}_{run.id}")
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
    agent_names = env.possible_agents
    mac = {a: QMixLocalAgent(local_dim, hidden_dim, n_actions).to(device) for a in agent_names}
    mixer = QMixer(len(agent_names), state_dim, cfg.agent.mixing_embed_dim, cfg.agent.hypernet_embed).to(device)
    target_mac = {a: QMixLocalAgent(local_dim, hidden_dim, n_actions).to(device) for a in agent_names}
    target_mixer = QMixer(len(agent_names), state_dim, cfg.agent.mixing_embed_dim, cfg.agent.hypernet_embed).to(device)

    for a in agent_names:
        target_mac[a].load_state_dict(mac[a].state_dict())
    target_mixer.load_state_dict(mixer.state_dict())

    all_params = list(mixer.parameters())
    for a in agent_names:
        all_params += list(mac[a].parameters())
    optimizer = optim.Adam(all_params, lr=cfg.agent.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.agent.lr_scheduler_step, gamma=cfg.agent.lr_scheduler_gamma)
    buffer = EpisodeReplayBuffer(cfg.agent.buffer_size)

    patience = cfg.agent.get("patience", 2000)
    warm_up = cfg.agent.get("warm_up_episodes", 1000)
    eps_decay_eps = cfg.agent.get("epsilon_decay_episodes", 5000)
    cost_history = deque(maxlen=50)
    best_avg_cost, since_imp, global_step = float("inf"), 0, 0
    epsilon = cfg.agent.epsilon_start
    last_loss = 0.0

    print("--- Starting QMIX Sequence Replay Training ---")

    for ep in range(cfg.total_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        hidden = {a: torch.zeros(1, hidden_dim, device=device) for a in agent_names}
        ep_states, ep_obs, ep_actions, ep_rewards, ep_dones = [], [], [], [], []
        ep_cost = 0.0
        ep_agent_costs = {a: 0.0 for a in agent_names}

        ep_obs.append(np.stack([obs[a] for a in agent_names]))
        ep_states.append(env.get_global_state())

        while True:
            env_acts, actions_list = {}, []
            for a in agent_names:
                o_t = torch.tensor(obs[a], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    q_vals, next_h = mac[a](o_t, hidden[a].detach())
                hidden[a] = next_h

                if random.random() < epsilon:
                    action_idx = random.randint(0, n_actions - 1)
                else:
                    action_idx = q_vals.argmax(dim=1).item()

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

            # FEEDBACK 3.2: Same scalar training objective as PPO/Comm-QMIX.
            global_reward = -raw_cost / cfg.agent.get("reward_scale", 100.0)
            done = any(terms.values()) or any(truncs.values())

            ep_actions.append(actions_list)
            ep_rewards.append([global_reward])
            ep_dones.append([float(done)])
            ep_obs.append(np.stack([next_obs[a] for a in agent_names]))
            ep_states.append(env.get_global_state())

            obs = next_obs
            global_step += 1
            if done: break

        buffer.push(ep_states, ep_obs, ep_actions, ep_rewards, ep_dones)

        if len(buffer) > cfg.agent.batch_size:
            updates_per_episode = cfg.agent.get("updates_per_episode", 1)
            for _ in range(updates_per_episode):
                batch = buffer.sample(cfg.agent.batch_size)
                last_loss = update_qmix(batch, mac, target_mac, mixer, target_mixer, optimizer, all_params, agent_names, cfg, device, hidden_dim)

            if ep % cfg.agent.target_update_freq == 0 and ep > 0:
                for a in agent_names:
                    target_mac[a].load_state_dict(mac[a].state_dict())
                target_mixer.load_state_dict(mixer.state_dict())

            scheduler.step()
        decay_step = (cfg.agent.epsilon_start - cfg.agent.epsilon_end) / max(1, eps_decay_eps)
        epsilon = max(cfg.agent.epsilon_end, cfg.agent.epsilon_start - decay_step * (ep + 1))

        cost_history.append(ep_cost)
        avg_cost = sum(cost_history) / len(cost_history)
        log_dict = {"Cost": ep_cost, "Avg_Cost_50": avg_cost, "Epsilon": epsilon, "LR": scheduler.get_last_lr()[0], "Loss": last_loss}
        for a, cost in ep_agent_costs.items():
            log_dict[f"Cost/{a}"] = cost
        wandb.log(log_dict)

        if ep == warm_up:
            best_avg_cost, since_imp = float("inf"), 0

        if avg_cost < best_avg_cost and len(cost_history) == 50:
            best_avg_cost = avg_cost
            since_imp = 0
            if ep >= warm_up:
                # FEEDBACK 7.3: Save full checkpoint plus per-agent files for old evaluation code.
                checkpoint = {
                    "mac": {a: mac[a].state_dict() for a in agent_names},
                    "mixer": mixer.state_dict(),
                    "target_mac": {a: target_mac[a].state_dict() for a in agent_names},
                    "target_mixer": target_mixer.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "episode": ep,
                    "epsilon": epsilon,
                    "best_avg_cost": best_avg_cost,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                }
                _torch_save(checkpoint, os.path.join(run_dir, "qmix_checkpoint_best.pt"))
                for a in agent_names:
                    _torch_save(mac[a].state_dict(), os.path.join(run_dir, f"qmix_agent_{a}_best.pth"))
        else:
            if ep >= warm_up:
                since_imp += 1

        exploration_lock = max(warm_up, eps_decay_eps)
        if ep > exploration_lock and since_imp >= patience:
            break
        if ep % 10 == 0:
            print(f"Ep {ep} | Cost: {ep_cost:.2f} | 50-Ep Avg: {avg_cost:.2f} | Best: {best_avg_cost if best_avg_cost != float('inf') else 0.0:.2f} | Eps: {epsilon:.2f}")

    wandb.finish()


if __name__ == "__main__":
    main()
