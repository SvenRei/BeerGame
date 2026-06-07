import sys, os, torch, wandb, numpy as np
from omegaconf import DictConfig
import hydra
from collections import deque 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.beer_game_env import BeerGameParallelEnv
from agents.rl.mappo import MAPPOActor, CommMAPPOActor, MAPPOCommMAC, MAPPOCritic, MAPPOTrainer, RolloutBuffer

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Initialize W&B
    run = wandb.init(project="BeerGame_Research", config=dict(cfg), name=cfg.agent.algorithm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Environment Initialization
    cfg.env.demand_type = "poisson"
    env = BeerGameParallelEnv(cfg.env)
    algo = cfg.agent.algorithm.lower()
    
    # 2. Reset env IMMEDIATELY to populate state for dimension calculations
    obs, _ = env.reset(seed=1000)
    
    # 3. Parameter setup after algo definition
    if "lr_actor" in wandb.config: cfg.agent.lr_actor = wandb.config.get("lr_actor")
    if "lr_critic" in wandb.config: cfg.agent.lr_critic = wandb.config.get("lr_critic")
    if "k_epochs" in wandb.config: cfg.agent.k_epochs = wandb.config.get("k_epochs")
    if "eps_clip" in wandb.config: cfg.agent.eps_clip = wandb.config.get("eps_clip")
    if "entropy_coef" in wandb.config: cfg.agent.entropy_coef = wandb.config.get("entropy_coef")
    if "hidden_dim" in wandb.config: cfg.agent.hidden_dim = wandb.config.get("hidden_dim")
    if "vocab_size" in wandb.config: cfg.agent.vocab_size = wandb.config.get("vocab_size")
    
    run_dir = os.path.join(f"weights_{algo}", f"run_{run.name}_{run.id}")
    os.makedirs(run_dir, exist_ok=True)
    
    local_dim = env.observation_space("retailer").shape[0]
    dummy_global = env.get_global_state()
    critic_in = local_dim if algo == "ippo" else len(dummy_global)
    
    # Network Initialization
    if algo == "comm_mappo":
        vocab_size = cfg.agent.get("vocab_size", 3)
        base_actor = CommMAPPOActor(local_dim, cfg.agent.hidden_dim, vocab_size=vocab_size).to(device)
        actor = MAPPOCommMAC(base_actor, vocab_size=vocab_size, num_agents=len(env.agents)).to(device)
    else:
        actor = MAPPOActor(local_dim, cfg.agent.hidden_dim).to(device)
        
    critic = MAPPOCritic(critic_in, cfg.agent.hidden_dim).to(device)
    trainer = MAPPOTrainer(actor, critic, cfg.agent, cfg.total_episodes, device, algo)
    
    actor_scheduler = torch.optim.lr_scheduler.StepLR(trainer.actor_optimizer, step_size=2000, gamma=0.5)
    critic_scheduler = torch.optim.lr_scheduler.StepLR(trainer.critic_optimizer, step_size=2000, gamma=0.5)
    
    patience, since_imp = 500, 0
    warm_up = cfg.agent.get("warm_up_episodes", 1000)
    
    cost_history = deque(maxlen=50)
    best_avg_cost = float('inf')

    print(f"--- Starting {algo.upper()} Training Marathon ---")
    print(f"Target Save Directory: {run_dir}")

    for ep in range(cfg.total_episodes):
        buffer = RolloutBuffer() 
        obs, _ = env.reset(seed=1000 + ep)
        
        if algo == "comm_mappo": 
            actor.init_buffer(batch_size=1, device=device)
            
        hiddens_tensor = torch.zeros(1, len(env.agents), cfg.agent.hidden_dim).to(device)
        ep_cost = 0.0
        ep_agent_costs = {a: 0.0 for a in env.agents}
        episode_messages = []
        
        current_tau = trainer.get_current_tau(ep) if algo == "comm_mappo" else 1.0
        
        while True:
                # 1. Inference
                obs_array = np.stack([obs[a] for a in env.agents]) # [4, local_dim]
                obs_tensor = torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0).to(device)
                state = env.get_global_state() # [global_dim]
                
                # CRITICAL: Standardize the global state shape to [1, global_dim]
                state_tensor = torch.tensor(state, dtype=torch.float32).view(1, -1).to(device).detach()
                
                with torch.no_grad():
                    if algo == "comm_mappo":
                        dist_action, dist_comm, comm_actions_raw, next_hiddens, masked_msg_in, safe_logs = actor(
                            obs_tensor, hiddens_tensor, tau=current_tau
                        )
                        
                        # ENGINEER FIX: Forced Exploration during warm-up phase
                        if ep < cfg.agent.warm_up_episodes:
                            # Override policy with uniform random orders to force environment exploration
                            actions_raw = torch.rand((len(env.agents), 1), device=device)
                        else:
                            actions_raw = dist_action.sample()
                            
                        actions_val = actions_raw.view(1, len(env.agents), 1)
                        comm_actions = comm_actions_raw.view(1, len(env.agents))
                        
                        # ENGINEER FIX: Append the discrete indices [0, 1, 2] instead of 
                        # the continuous embeddings [-1.0, 0.0, 1.0]
                        episode_messages.append(comm_actions_raw.detach().cpu().numpy())
                        msg_in_buffer = masked_msg_in.squeeze(0)
                        comm_acts_buffer = comm_actions.squeeze(0).unsqueeze(-1)
                        
                        # Calculate log_probs based on the action taken (works for both forced and sampled)
                        log_probs_val = dist_action.log_prob(actions_raw).view(1, len(env.agents), 1) + \
                                        dist_comm.log_prob(comm_actions_raw).view(1, len(env.agents), 1)
                    else:
                        dist_action, next_hiddens_raw = actor(obs_tensor.view(-1, local_dim), hiddens_tensor.view(-1, cfg.agent.hidden_dim))
                        
                        # ENGINEER FIX: Forced Exploration for IPPO/MAPPO
                        if ep < cfg.agent.warm_up_episodes:
                            actions_raw = torch.rand((len(env.agents), 1), device=device)
                        else:
                            actions_raw = dist_action.sample()
                            
                        actions_val = actions_raw.view(1, len(env.agents), 1)
                        next_hiddens = next_hiddens_raw.view(1, len(env.agents), -1)
                        log_probs_val = dist_action.log_prob(actions_raw).view(1, len(env.agents), 1)
                        msg_in_buffer = None
                        comm_acts_buffer = None
                
                acts = {a: [actions_val[0, i, 0].cpu().item()] for i, a in enumerate(env.agents)}
                
                # --- DIAGNOSTIC LOGGING ---
                # Log orders to WandB to physically verify they are ordering
                if env.current_step % 10 == 0:
                    order_quantities = {a: float(np.round(acts[a][0] * env.max_order)) for a in env.agents}
                    wandb.log({f"Order_Qty/{a}": order_quantities[a] for a in env.agents}, commit=False)
                
                # 2. Step Env
                next_obs, rewards, terms, truncs, infos = env.step(acts)
                # ENGINEER FIX: Accumulate both total cost AND individual agent costs
                for a in env.agents:
                    local_cost = infos[a]["local_cost"]
                    ep_cost += local_cost
                    ep_agent_costs[a] += local_cost # This line was missing!
                
                # 3. Create Contract-Compliant Tensors
                r_tensor = torch.zeros(len(env.possible_agents), 1, device=device)
                t_tensor = torch.zeros(len(env.possible_agents), 1, device=device)
                for i, a in enumerate(env.possible_agents):
                    r_tensor[i] = -abs(rewards.get(a, 0.0)) / 100.0
                    t_tensor[i] = float(terms.get(a, False))

                # 4. PUSH ALL (Using the standardized state_tensor)
                buffer.push(
                    obs=obs_tensor.squeeze(0).detach(),
                    g_state=state_tensor, # Standardized to [1, global_dim]
                    hidden=hiddens_tensor.squeeze(0).detach(),
                    comm_in=msg_in_buffer.detach() if msg_in_buffer is not None else torch.zeros(len(env.agents), 1, device=device),
                    action=actions_val.squeeze(0).detach(),
                    log_prob=log_probs_val.squeeze(0).detach(),
                    comm_action=comm_acts_buffer.detach() if comm_acts_buffer is not None else torch.zeros(len(env.agents), 1, device=device),
                    reward=r_tensor,
                    terminal=t_tensor
                )

                hiddens_tensor = next_hiddens.detach()
                obs = next_obs
                if any(terms.values()) or any(truncs.values()): break
                
                actor_loss, critic_loss = trainer.update(buffer, ep)
                actor_scheduler.step()
                critic_scheduler.step()

        # Log to W&B
        log_dict = {
            "Cost": ep_cost, 
            "Actor_Loss": actor_loss, 
            "Critic_Loss": critic_loss, 
            "Actor_LR": actor_scheduler.get_last_lr()[0],
            "Tau": current_tau if algo == "comm_mappo" else 0.0
        }
        for a, cost in ep_agent_costs.items(): log_dict[f"Cost/{a}"] = cost
        
        # ENGINEER FIX: Communication Telemetry
        if algo == "comm_mappo" and len(episode_messages) > 0:
            # Flatten all messages sent in this episode by all agents
            # CRITICAL: .astype(int) is required for np.bincount to work without throwing an error
            all_msgs = np.concatenate(episode_messages, axis=0).flatten().astype(int)
            
            # 1. Log a histogram to W&B (You will see a bar chart of tokens 0, 1, 2)
            log_dict["Comm/Message_Distribution"] = wandb.Histogram(all_msgs)
            
            # 2. Log how many unique words they used this episode
            log_dict["Comm/Unique_Tokens"] = len(np.unique(all_msgs))
            
            # 3. Explicitly log the percentage of EACH token used as individual line charts!
            # minlength=3 ensures the array length matches vocab_size even if a token isn't used
            vocab_size = 3
            token_counts = np.bincount(all_msgs, minlength=vocab_size)
            token_pcts = (token_counts / len(all_msgs)) * 100.0
            
            for v in range(vocab_size):
                log_dict[f"Comm/Token_{v}_Pct"] = token_pcts[v]

        wandb.log(log_dict)
        
        cost_history.append(ep_cost)
        avg_cost = sum(cost_history) / len(cost_history)
        
        if ep == warm_up: best_avg_cost, since_imp = float('inf'), 0
            
        if avg_cost < best_avg_cost and len(cost_history) == 50: 
            best_avg_cost = avg_cost
            since_imp = 0
            if ep >= warm_up:
                torch.save(actor.state_dict(), os.path.join(run_dir, f"{algo}_best.pth"))
                if algo == "comm_mappo":
                    np.save(os.path.join(run_dir, f"best_messages_ep_{ep}.npy"), np.concatenate(episode_messages, axis=0))
        else: 
            if ep >= warm_up: since_imp += 1
        
        if ep > warm_up and since_imp >= patience: break
        if ep % 10 == 0: 
            print(f"Ep {ep} | Cost: {ep_cost:.2f} | 50-Ep Avg: {avg_cost:.2f} | Best: {best_avg_cost if best_avg_cost != float('inf') else 0.0:.2f}")
            
    wandb.finish()

if __name__ == "__main__": main()