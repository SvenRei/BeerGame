import io
import os
import sys
import csv
import random
from collections import deque

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.distributions import Normal


def _torch_save(obj, path, _retries=6, _delay=5):
    import time
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = io.BytesIO(); torch.save(obj, buf); data = buf.getvalue()
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
from agents.rl.draco import (
    ADJ, DemandRandomizedBeerGame, ContextEncoder, DRACOActor,
    DistributionalCritic, DRACOTrainer, DRACORolloutBuffer,
)


def set_global_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    base_seed = cfg.get("seed", 1000)
    set_global_seeds(base_seed)
    print("[draco] booting...", flush=True)

    run = wandb.init(project="BeerGame_Research", name=cfg.agent.algorithm)
    wandb.define_metric("Avg_Cost_50", summary="min")
    wandb.define_metric("Avg_Cost_500", summary="last")
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # demand randomization (Module E) -- only perturbs poisson (training)
    env = DemandRandomizedBeerGame(
        cfg.env,
        lam_lo=cfg.agent.get("dr_lambda_lo", 4.0),
        lam_hi=cfg.agent.get("dr_lambda_hi", 16.0),
        p_shift=cfg.agent.get("dr_p_shift", 0.5),
        shift_scale=cfg.agent.get("dr_shift_scale", 2.0),
    )
    obs, _ = env.reset(seed=base_seed)

    run_dir = os.path.join(_ROOT, "weights_draco", f"run_draco_{run.id}")
    os.makedirs(run_dir, exist_ok=True)

    agents = list(env.possible_agents)
    N = len(agents)
    local_dim = env.observation_space("retailer").shape[0]
    gdim = len(env.get_global_state())
    max_order = env.max_order

    hidden = cfg.agent.hidden_dim
    z_dim = cfg.agent.get("z_dim", 8)
    msg_dim = cfg.agent.get("msg_dim", 4)
    n_quant = cfg.agent.get("n_quantiles", 8)
    adj = ADJ.to(device)

    s_bias_init = cfg.agent.get("s_bias_init", 40.0)
    s_logstd_init = cfg.agent.get("s_logstd_init", 1.0)
    use_comm = cfg.agent.get("use_comm", True)
    use_context = cfg.agent.get("use_context", True)
    encoder = ContextEncoder(local_dim, msg_dim, z_dim, hidden).to(device)
    actors = [DRACOActor(local_dim, z_dim, msg_dim, hidden, max_order,
                         s_bias_init=s_bias_init, s_logstd_init=s_logstd_init).to(device)
              for _ in range(N)]
    critic = DistributionalCritic(gdim, hidden, n_quant).to(device)
    trainer = DRACOTrainer(encoder, actors, critic, cfg.agent, cfg.total_episodes, device)

    schedulers = [torch.optim.lr_scheduler.StepLR(o, cfg.agent.lr_scheduler_step, cfg.agent.lr_scheduler_gamma)
                  for o in trainer.actor_opt] + [
        torch.optim.lr_scheduler.StepLR(trainer.critic_opt, cfg.agent.lr_scheduler_step, cfg.agent.lr_scheduler_gamma),
        torch.optim.lr_scheduler.StepLR(trainer.enc_opt, cfg.agent.lr_scheduler_step, cfg.agent.lr_scheduler_gamma)]

    warm_up = cfg.agent.get("warm_up_episodes", 1000)
    patience = cfg.agent.get("patience", 800)
    trace_every = cfg.agent.get("trace_every", 0)          # >0 -> dump a per-step trace CSV for symbolic regression
    cost_hist, cost_hist_500 = deque(maxlen=50), deque(maxlen=500)
    best, since_imp = float("inf"), 0

    print(f"[draco] built: {N} actors, z={z_dim}, msg={msg_dim}, quantiles={n_quant}. Starting loop.", flush=True)

    def run_episode(ep, collect, deterministic=False, trace_rows=None):
        nonlocal obs
        obs, _ = env.reset(seed=base_seed + ep)
        cur = env.possible_agents
        h_enc = torch.zeros(1, N, hidden, device=device)
        m_buf = torch.zeros(N, msg_dim, device=device)            # previous-step messages (one-step delay)
        prev_a = torch.zeros(N, 1, device=device)
        buf = DRACORolloutBuffer()
        ep_cost = 0.0
        ep_costs = {a: 0.0 for a in cur}
        msgs_log = []
        s_sum = order_sum = 0.0
        nstep = 0
        while True:
            o_arr = np.stack([obs[a] for a in cur])
            o_t = torch.tensor(o_arr, dtype=torch.float32, device=device)         # [N,od]
            g_t = torch.tensor(env.get_global_state(), dtype=torch.float32, device=device).view(-1)
            m_tilde = adj @ m_buf                                                  # [N,msg] incoming (delayed)
            if not use_comm:
                m_tilde = torch.zeros_like(m_tilde)

            with torch.no_grad():
                z, h_enc = encoder.step(o_t, prev_a, m_tilde, h_enc)              # z [N,z]
                if not use_context:
                    z = torch.zeros_like(z)
                S = torch.zeros(N, 1, device=device)
                m_out = torch.zeros(N, msg_dim, device=device)
                logp = torch.zeros(N, 1, device=device)
                for i in range(N):
                    s_mu, s_std, mm, ms = actors[i](o_t[i:i+1], z[i:i+1], m_tilde[i:i+1])
                    S_i = s_mu if deterministic else Normal(s_mu, s_std).rsample()
                    logp_i = Normal(s_mu, s_std).log_prob(S_i).sum()
                    if use_comm:
                        m_i = mm if deterministic else Normal(mm, ms).rsample()
                        logp_i = logp_i + Normal(mm, ms).log_prob(m_i).sum()
                        m_out[i] = m_i
                    logp[i] = logp_i
                    S[i] = S_i

            order, IP = DRACOActor.order_from_S(S, o_t, max_order)                # [N,1]
            frac = (order / max_order).clamp(0.0, 1.0)
            acts = {a: [float(frac[i, 0].item())] for i, a in enumerate(cur)}
            s_sum += float(S.mean().item()); order_sum += float(order.mean().item()); nstep += 1

            if trace_rows is not None:
                for i, a in enumerate(cur):
                    trace_rows.append({
                        "ep": ep, "t": env.current_step, "agent": a,
                        "inv": float(o_t[i, 0]), "backlog": float(o_t[i, 1]),
                        "on_order": float(o_t[i, 2]), "next_demand": float(o_t[i, 3]),
                        "IP": float(IP[i, 0]), "S_target": float(S[i, 0]), "order": float(order[i, 0]),
                        **{f"z{j}": float(z[i, j]) for j in range(z_dim)},
                        **{f"msg_in{j}": float(m_tilde[i, j]) for j in range(msg_dim)},
                        **{f"msg_out{j}": float(m_out[i, j]) for j in range(msg_dim)},
                    })

            next_obs, rewards, terms, truncs, infos = env.step(acts)
            raw_cost = 0.0
            for i, a in enumerate(cur):
                lc = infos[a]["local_cost"]; raw_cost += lc; ep_costs[a] += lc
            ep_cost += raw_cost
            done = any(terms.values()) or any(truncs.values())
            team_r = torch.tensor([-raw_cost / cfg.agent.get("reward_scale", 100.0)], device=device)
            term = torch.tensor([1.0 if done else 0.0], device=device)
            demand_tgt = o_t[:, 3:4].clone()                                      # next incoming demand per agent

            if collect:
                buf.push(obs=o_t, g=g_t.view(-1), prev_a=prev_a.clone(), msg_in=m_tilde.detach(),
                         S_act=S.detach(), m_act=m_out.detach(), logp=logp.detach(),
                         reward=team_r, done=term, demand_tgt=demand_tgt)
            msgs_log.append(m_out.detach().cpu().numpy())
            m_buf = m_out.detach()
            prev_a = frac.detach()
            obs = next_obs
            if done:
                break
        return buf, ep_cost, ep_costs, msgs_log, (s_sum / max(1, nstep), order_sum / max(1, nstep))

    batch_eps = cfg.agent.get("batch_episodes", 8)
    episode_buffers = []
    a_loss = c_loss = e_loss = 0.0
    for ep in range(cfg.total_episodes):
        train_this = ep >= warm_up
        buf, ep_cost, ep_costs, msgs_log, (s_mean, order_mean) = run_episode(ep, collect=train_this, deterministic=False)
        if train_this and len(buf) > 0:
            episode_buffers.append(buf)
            if len(episode_buffers) >= batch_eps:
                a_loss, c_loss, e_loss = trainer.update(episode_buffers)
                for s in schedulers:
                    s.step()
                episode_buffers = []

        cost_hist.append(ep_cost); cost_hist_500.append(ep_cost)
        avg = sum(cost_hist) / len(cost_hist)
        log = {"Cost": ep_cost, "Avg_Cost_50": avg, "Avg_Cost_500": sum(cost_hist_500) / len(cost_hist_500),
               "Actor_Loss": a_loss, "Critic_Loss": c_loss, "Encoder_Loss": e_loss,
               "Diag/S_mean": s_mean, "Diag/Order_mean": order_mean}
        for a, c in ep_costs.items():
            log[f"Cost/{a}"] = c
        if msgs_log:
            arr = np.concatenate(msgs_log, axis=0)
            log["Comm/Msg_Mean_Abs"] = float(np.abs(arr).mean())
            log["Comm/Msg_Std"] = float(arr.std())
        wandb.log(log)

        if ep == warm_up:
            best, since_imp = float("inf"), 0
        if avg < best and len(cost_hist) == 50:
            best, since_imp = avg, 0
            if ep >= warm_up:
                _torch_save({"encoder": encoder.state_dict(),
                             "actors": [a.state_dict() for a in actors],
                             "critic": critic.state_dict(),
                             "config": OmegaConf.to_container(cfg, resolve=True),
                             "episode": ep, "best_avg_cost": best},
                            os.path.join(run_dir, "draco_checkpoint_best.pt"))
        else:
            if ep >= warm_up:
                since_imp += 1

        # periodically dump a deterministic trace for offline symbolic regression
        if trace_every and ep > warm_up and ep % trace_every == 0:
            rows = []
            run_episode(ep, collect=False, deterministic=True, trace_rows=rows)  # diag ignored
            if rows:
                path = os.path.join(run_dir, f"trace_ep{ep}.csv")
                with open(path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

        if ep > warm_up and since_imp >= patience:
            print(f"[draco] early stop at ep {ep}.", flush=True)
            break
        if ep % 10 == 0 or ep < 3:
            print(f"Ep {ep} | Cost {ep_cost:.1f} | 50-avg {avg:.1f} | best {best if best!=float('inf') else 0:.1f}", flush=True)

    wandb.finish()


if __name__ == "__main__":
    main()