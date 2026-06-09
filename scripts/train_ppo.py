import io
import os
import sys
import random
from collections import deque

import hydra
import numpy as np
import torch
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
from agents.rl.mappo import (
    MAPPOActor,
    CommMAPPOActor,
    MAPPOCommMAC,
    MAPPOCritic,
    MAPPOTrainer,
    RolloutBuffer,
)


def set_global_seeds(seed):
    # FEEDBACK 7.2: Reproducibility requires Python, NumPy, Torch, and env seeds.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_plain_dict(cfg_node):
    return OmegaConf.to_container(cfg_node, resolve=True) if hasattr(cfg_node, "keys") else dict(cfg_node)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    base_seed = cfg.get("seed", 1000)
    set_global_seeds(base_seed)

    # wandb.init first so sweep params are injected into wandb.config
    run = wandb.init(project="BeerGame_Research", name=cfg.agent.algorithm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Apply sweep overrides to cfg, then log the resolved config
    #with open_dict(cfg):
    #    for key in ["lr_actor", "lr_critic", "k_epochs", "eps_clip", "entropy_coef", "hidden_dim", "vocab_size", "lr_scheduler_step", "lr_scheduler_gamma"]:
    #        if key in wandb.config:
    #            cfg.agent[key] = wandb.config[key]
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)

    env = BeerGameParallelEnv(cfg.env)
    algo = cfg.agent.algorithm.lower()
    if algo not in {"ippo", "mappo", "comm_mappo"}:
        raise ValueError(f"Unsupported PPO algorithm: {algo}")

    obs, _ = env.reset(seed=base_seed)

    run_dir = os.path.join(_PROJECT_ROOT, f"weights_{algo}", f"run_{run.name}_{run.id}")
    os.makedirs(run_dir, exist_ok=True)

    local_dim = env.observation_space("retailer").shape[0]
    dummy_global = env.get_global_state()
    critic_in = local_dim if algo == "ippo" else len(dummy_global)
    num_agents = len(env.possible_agents)

    n_actions = cfg.agent.get("n_actions", 21)
    if n_actions < 2:
        raise ValueError(f"n_actions must be >= 2, got {n_actions}")

    if algo == "comm_mappo":
        vocab_size = cfg.agent.get("vocab_size", 3)
        base_actor = CommMAPPOActor(local_dim, cfg.agent.hidden_dim, n_actions=n_actions, vocab_size=vocab_size).to(device)
        actor = MAPPOCommMAC(base_actor, vocab_size=vocab_size, num_agents=num_agents).to(device)
    else:
        actor = MAPPOActor(local_dim, cfg.agent.hidden_dim, n_actions=n_actions).to(device)

    critic = MAPPOCritic(critic_in, cfg.agent.hidden_dim).to(device)
    trainer = MAPPOTrainer(actor, critic, cfg.agent, cfg.total_episodes, device, algo)

    actor_scheduler = torch.optim.lr_scheduler.StepLR(trainer.actor_optimizer, step_size=cfg.agent.lr_scheduler_step, gamma=cfg.agent.lr_scheduler_gamma)
    critic_scheduler = torch.optim.lr_scheduler.StepLR(trainer.critic_optimizer, step_size=cfg.agent.lr_scheduler_step, gamma=cfg.agent.lr_scheduler_gamma)

    patience = cfg.agent.get("patience", 500)
    warm_up = cfg.agent.get("warm_up_episodes", 1000)
    cost_history = deque(maxlen=50)
    best_avg_cost = float("inf")
    since_imp = 0

    print(f"--- Starting {algo.upper()} Training ---")
    print(f"Target Save Directory: {run_dir}")

    for ep in range(cfg.total_episodes):
        buffer = RolloutBuffer()
        obs, _ = env.reset(seed=base_seed + ep)
        current_agents = env.possible_agents

        if algo == "comm_mappo":
            actor.init_buffer(batch_size=1, device=device)

        hiddens_tensor = torch.zeros(1, num_agents, cfg.agent.hidden_dim, device=device)
        ep_cost = 0.0
        ep_agent_costs = {a: 0.0 for a in current_agents}
        episode_messages = []
        current_tau = trainer.get_current_tau(ep) if algo == "comm_mappo" else 1.0
        train_this_episode = ep >= warm_up
        actor_loss, critic_loss = 0.0, 0.0

        while True:
            obs_array = np.stack([obs[a] for a in current_agents])
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=device).unsqueeze(0)
            state_tensor = torch.tensor(env.get_global_state(), dtype=torch.float32, device=device).view(1, -1).detach()

            with torch.no_grad():
                if algo == "comm_mappo":
                    dist_action, dist_comm, comm_actions_raw, next_hiddens, masked_msg_in, safe_logs = actor(
                        obs_tensor, hiddens_tensor, tau=current_tau
                    )
                    if train_this_episode:
                        action_indices = dist_action.sample().view(num_agents, 1)
                    else:
                        # FEEDBACK 2.5: Warm-up random actions are off-policy, so they are not stored/updated.
                        action_indices = torch.randint(0, n_actions, (num_agents, 1), device=device)

                    actions_val = action_indices.view(1, num_agents, 1).long()
                    comm_actions = comm_actions_raw.view(1, num_agents)
                    episode_messages.append(comm_actions_raw.detach().cpu().numpy())
                    msg_in_buffer = masked_msg_in.squeeze(0)
                    comm_acts_buffer = comm_actions.squeeze(0).unsqueeze(-1).long()
                    log_probs_val = (
                        dist_action.log_prob(action_indices.squeeze(-1).long()).view(1, num_agents, 1)
                        + dist_comm.log_prob(comm_actions_raw).view(1, num_agents, 1)
                    )
                else:
                    dist_action, next_hiddens_raw = actor(
                        obs_tensor.view(-1, local_dim),
                        hiddens_tensor.view(-1, cfg.agent.hidden_dim),
                    )
                    if train_this_episode:
                        action_indices = dist_action.sample().view(num_agents, 1)
                    else:
                        # FEEDBACK 2.5: Warm-up random actions are off-policy, so they are not stored/updated.
                        action_indices = torch.randint(0, n_actions, (num_agents, 1), device=device)

                    actions_val = action_indices.view(1, num_agents, 1).long()
                    next_hiddens = next_hiddens_raw.view(1, num_agents, -1)
                    log_probs_val = dist_action.log_prob(action_indices.squeeze(-1).long()).view(1, num_agents, 1)
                    msg_in_buffer = torch.zeros(num_agents, 1, device=device)
                    comm_acts_buffer = torch.zeros(num_agents, 1, dtype=torch.long, device=device)

            acts = {a: [actions_val[0, i, 0].cpu().item() / (n_actions - 1)] for i, a in enumerate(current_agents)}

            if env.current_step % 10 == 0:
                order_quantities = {a: float(np.round(acts[a][0] * env.max_order)) for a in current_agents}
                wandb.log({f"Order_Qty/{a}": order_quantities[a] for a in current_agents}, commit=False)

            next_obs, rewards, terms, truncs, infos = env.step(acts)

            # FEEDBACK 2.7: env.agents is empty after truncation; use current_agents/infos for terminal cost.
            raw_cost = 0.0
            for a in current_agents:
                local_cost = infos[a]["local_cost"]
                raw_cost += local_cost
                ep_agent_costs[a] += local_cost
            ep_cost += raw_cost

            done = any(terms.values()) or any(truncs.values())

            # FEEDBACK 3.2: Use the same benchmark training reward as QMIX: -total_system_cost / scale.
            global_reward = -raw_cost / cfg.agent.get("reward_scale", 100.0)
            r_tensor = torch.full((num_agents, 1), float(global_reward), device=device)

            # FEEDBACK 2.2: Include truncations in terminal masks.
            t_tensor = torch.zeros(num_agents, 1, device=device)
            for i, a in enumerate(current_agents):
                t_tensor[i] = float(terms.get(a, False) or truncs.get(a, False))

            if train_this_episode:
                buffer.push(
                    obs=obs_tensor.squeeze(0).detach(),
                    g_state=state_tensor,
                    hidden=hiddens_tensor.squeeze(0).detach(),
                    comm_in=msg_in_buffer.detach(),
                    action=actions_val.squeeze(0).detach(),
                    log_prob=log_probs_val.squeeze(0).detach(),
                    comm_action=comm_acts_buffer.detach(),
                    reward=r_tensor,
                    terminal=t_tensor,
                )

            hiddens_tensor = next_hiddens.detach()
            obs = next_obs
            if done:
                break

        # FEEDBACK 2.1/2.3: PPO update happens once after collecting the full episode,
        # so recurrent BPTT and terminal transitions are actually used.
        if train_this_episode and len(buffer) > 0:
            actor_loss, critic_loss = trainer.update(buffer, ep)
            actor_scheduler.step()
            critic_scheduler.step()

        cost_history.append(ep_cost)
        avg_cost = sum(cost_history) / len(cost_history)

        log_dict = {
            "Cost": ep_cost,
            "Avg_Cost_50": avg_cost,
            "Actor_Loss": actor_loss,
            "Critic_Loss": critic_loss,
            "Actor_LR": actor_scheduler.get_last_lr()[0],
            "Tau": current_tau if algo == "comm_mappo" else 0.0,
        }
        for a, cost in ep_agent_costs.items():
            log_dict[f"Cost/{a}"] = cost

        if algo == "comm_mappo" and len(episode_messages) > 0:
            all_msgs = np.concatenate(episode_messages, axis=0).flatten().astype(int)
            log_dict["Comm/Message_Distribution"] = wandb.Histogram(all_msgs)
            log_dict["Comm/Unique_Tokens"] = len(np.unique(all_msgs))
            vocab_size = cfg.agent.get("vocab_size", 3)
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
                # FEEDBACK 7.3: Save a complete training checkpoint, not only the actor weights.
                checkpoint = {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "actor_optimizer": trainer.actor_optimizer.state_dict(),
                    "critic_optimizer": trainer.critic_optimizer.state_dict(),
                    "actor_scheduler": actor_scheduler.state_dict(),
                    "critic_scheduler": critic_scheduler.state_dict(),
                    "episode": ep,
                    "best_avg_cost": best_avg_cost,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                }
                _torch_save(checkpoint, os.path.join(run_dir, f"{algo}_checkpoint_best.pt"))
                _torch_save(actor.state_dict(), os.path.join(run_dir, f"{algo}_actor_best.pth"))
                if algo == "comm_mappo" and len(episode_messages) > 0:
                    np.save(os.path.join(run_dir, f"best_messages_ep_{ep}.npy"), np.concatenate(episode_messages, axis=0))
        else:
            if ep >= warm_up:
                since_imp += 1

        if ep > warm_up and since_imp >= patience:
            break
        if ep % 10 == 0:
            print(
                f"Ep {ep} | Cost: {ep_cost:.2f} | 50-Ep Avg: {avg_cost:.2f} | "
                f"Best: {best_avg_cost if best_avg_cost != float('inf') else 0.0:.2f}"
            )

    wandb.finish()


if __name__ == "__main__":
    main()
