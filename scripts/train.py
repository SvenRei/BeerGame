import sys, os, torch, wandb, numpy as np
from omegaconf import DictConfig
import hydra
from collections import deque 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.beer_game_env import BeerGameParallelEnv
from agents.rl.mappo import MAPPOActor, CommMAPPOActor, MAPPOCritic, MAPPOTrainer, RolloutBuffer

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    wandb.init(project="BeerGame_Research", config=dict(cfg), name=cfg.agent.algorithm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.env.demand_type = "poisson"
    env = BeerGameParallelEnv(cfg.env)
    algo = cfg.agent.algorithm.lower()
    
    local_dim = env.observation_space("retailer").shape[0]
    critic_in = local_dim if algo == "ippo" else local_dim * len(env.agents)
    
    if algo == "comm_mappo":
        actor = CommMAPPOActor(local_dim, cfg.agent.hidden_dim).to(device)
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
    print(f"Warm-up Phase: {warm_up} episodes | Early Stopping active after warm-up.")

    for ep in range(cfg.total_episodes):
        buffer = RolloutBuffer() 
        obs, _ = env.reset(seed=1000 + ep)
        hidden = {a: torch.zeros(1, 1, cfg.agent.hidden_dim).to(device) for a in env.agents}
        msg = {a: torch.zeros(1, 1).to(device) for a in env.agents}
        ep_cost = 0.0
        
        ep_agent_costs = {a: 0.0 for a in env.agents}
        
        while True:
            acts, next_msg = {}, {}
            sorted_agents = sorted(env.agents)
            state = np.concatenate([obs[a] for a in sorted_agents])
            
            for i, a in enumerate(env.agents):
                o_t = torch.tensor(obs[a], dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    if algo == "comm_mappo":
                        dist, dist_comm, next_h = actor(o_t, msg[a], hidden[a])
                        
                        comm_idx = dist_comm.sample()
                        # Map categorical choice to vocabulary: -1.0, 0.0, 1.0
                        vocab = torch.tensor([-1.0, 0.0, 1.0], device=device)
                        comm_val = vocab[comm_idx].view(1, 1) 
                        
                        # --- FIX B: 10% LATENT CHANNEL DROPOUT ---
                        # Simulates packet loss. Forces the receiver to stay adaptable 
                        # and prevents the sender from building a "Dial Tone" bridge.
                        if torch.rand(1).item() < 0.10:
                            comm_val = torch.tensor([[0.0]], dtype=torch.float32, device=device) # Force Silence
                        
                        if i < len(env.agents) - 1: next_msg[env.agents[i+1]] = comm_val
                    else:
                        dist, next_h = actor(o_t, hidden[a])
                
                action_val = dist.sample()
                acts[a] = [action_val.cpu().item()]
                
                buffer.local_obs.append(o_t)
                buffer.hidden_states.append(hidden[a])
                buffer.actions.append(action_val.detach())
                buffer.global_states.append(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
                
                if algo == "comm_mappo": 
                    buffer.comm_in.append(msg[a])
                    buffer.comm_actions.append(comm_idx.detach())
                    combined_log_prob = dist.log_prob(action_val) + dist_comm.log_prob(comm_idx)
                    buffer.log_probs.append(combined_log_prob.detach())
                else:
                    buffer.log_probs.append(dist.log_prob(action_val).detach())
                    
                hidden[a] = next_h
            
            msg = next_msg
            msg["retailer"] = torch.zeros(1, 1).to(device)
            
            obs, rewards, terms, truncs, infos = env.step(acts)
            
            true_step_cost = sum(infos[a]["local_cost"] for a in env.agents)
            ep_cost += true_step_cost
            
            for a in env.agents: 
                local_cost = infos.get(a, {}).get("local_cost", 0.0)
                ep_agent_costs[a] += local_cost 
                
                if algo == "ippo":
                    raw_cost = abs(local_cost)
                else:
                    raw_cost = abs(rewards[a])
                
                scaled_reward = -raw_cost / 100.0 
                buffer.rewards.append(scaled_reward)
                buffer.is_terminals.append(terms[a])
                
            if any(terms.values()): break
            
        actor_loss, critic_loss = trainer.update(buffer, ep)
        
        actor_scheduler.step()
        critic_scheduler.step()
        
        log_dict = {
            "Cost": ep_cost, 
            "Actor_Loss": actor_loss, 
            "Critic_Loss": critic_loss,
            "Actor_LR": actor_scheduler.get_last_lr()[0]
        }
        for a, cost in ep_agent_costs.items():
            log_dict[f"Cost/{a}"] = cost
            
        wandb.log(log_dict)
        
        cost_history.append(ep_cost)
        avg_cost = sum(cost_history) / len(cost_history)
        
        if ep == warm_up:
            print(f"--- Ep {ep}: Warm-up complete! Resetting early stopping baseline. ---")
            best_avg_cost = float('inf')
            since_imp = 0
            
        if avg_cost < best_avg_cost and len(cost_history) == 50: 
            best_avg_cost = avg_cost
            since_imp = 0
            if ep >= warm_up:
                torch.save(actor.state_dict(), f"{algo}_best.pth")
        else: 
            if ep >= warm_up: 
                since_imp += 1
        
        if ep > warm_up and since_imp >= patience: 
            print(f"Stopping early at Ep {ep}: No improvement in 50-Ep Avg Cost for {patience} episodes.")
            break
            
        if ep % 10 == 0: 
            best_display = best_avg_cost if best_avg_cost != float('inf') else 0.0
            print(f"Ep {ep} | Cost: {ep_cost:.2f} | 50-Ep Avg: {avg_cost:.2f} | Best Avg: {best_display:.2f}")
    
    wandb.finish()

if __name__ == "__main__": main()