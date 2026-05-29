import sys, os, torch, wandb, numpy as np
from omegaconf import DictConfig
import hydra

# Add project root
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
    
    actor = CommMAPPOActor(local_dim, cfg.agent.hidden_dim) if algo == "comm_mappo" else MAPPOActor(local_dim, cfg.agent.hidden_dim)
    actor, critic = actor.to(device), MAPPOCritic(critic_in, cfg.agent.hidden_dim).to(device)
    
    trainer = MAPPOTrainer(actor, critic, cfg.agent, device, algo)
    
    # --- STEP 1: SCHEDULER INITIALIZATION ---
    actor_scheduler = torch.optim.lr_scheduler.StepLR(trainer.actor_optimizer, step_size=2000, gamma=0.5)
    critic_scheduler = torch.optim.lr_scheduler.StepLR(trainer.critic_optimizer, step_size=2000, gamma=0.5)
    
    best_cost, patience, since_imp = float('inf'), 500, 0
    warm_up = cfg.agent.get("warm_up_episodes", 1000)

    print(f"--- Starting {algo.upper()} Training Marathon ---")
    print(f"Warm-up Phase: {warm_up} episodes | Early Stopping active after warm-up.")

    for ep in range(cfg.total_episodes):
        buffer = RolloutBuffer() 
        obs, _ = env.reset(seed=1000 + ep)
        hidden = {a: torch.zeros(1, 1, cfg.agent.hidden_dim).to(device) for a in env.agents}
        msg = {a: torch.zeros(1, 1).to(device) for a in env.agents}
        ep_cost = 0.0
        
        while True:
            acts, next_msg = {}, {}
            sorted_agents = sorted(env.agents)
            state = np.concatenate([obs[a] for a in sorted_agents])
            
            for i, a in enumerate(env.agents):
                o_t = torch.tensor(obs[a], dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    if algo == "comm_mappo":
                        dist, comm, next_h = actor(o_t, msg[a], hidden[a])
                        if i < len(env.agents)-1: next_msg[env.agents[i+1]] = comm
                    else:
                        dist, next_h = actor(o_t, hidden[a])
                
                action_val = dist.sample()
                acts[a] = [action_val.cpu().item()]
                
                buffer.local_obs.append(o_t)
                buffer.hidden_states.append(hidden[a])
                buffer.actions.append(action_val.detach())
                buffer.log_probs.append(dist.log_prob(action_val).detach())
                buffer.global_states.append(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
                if algo == "comm_mappo": buffer.comm_in.append(msg[a])
                hidden[a] = next_h
            
            msg = next_msg; msg["retailer"] = torch.zeros(1, 1).to(device)
            obs, rewards, terms, _, _ = env.step(acts)
            ep_cost -= rewards["retailer"]
            for a in env.agents: 
                buffer.rewards.append(rewards[a])
                buffer.is_terminals.append(terms[a])
            if any(terms.values()): break
            
        # --- STEP 2: UPDATE TRAINER (Pass 'ep' for Warm-Up Logic) & SCHEDULER ---
        actor_loss, critic_loss = trainer.update(buffer, ep)
        
        # Step the schedulers
        actor_scheduler.step()
        critic_scheduler.step()
        
        wandb.log({
            "Cost": ep_cost, 
            "Actor_Loss": actor_loss, 
            "Critic_Loss": critic_loss,
            "Actor_LR": actor_scheduler.get_last_lr()[0]
        })
        
        if ep_cost < best_cost: 
            best_cost = ep_cost; since_imp = 0; torch.save(actor.state_dict(), f"{algo}_best.pth")
        else: since_imp += 1
        
        # --- FIX: EARLY STOPPING ONLY AFTER WARM-UP ---
        if ep > warm_up and since_imp >= patience: 
            print(f"Stopping early at Ep {ep}: No improvement for {patience} episodes after warm-up.")
            break
            
        if ep % 10 == 0: print(f"Ep {ep} | Cost: {ep_cost:.2f} | Best: {best_cost:.2f}")
    
    wandb.finish()

if __name__ == "__main__": main()