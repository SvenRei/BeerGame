import sys, os, torch, wandb, numpy as np
from omegaconf import DictConfig
import hydra
from collections import deque 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.beer_game_env import BeerGameParallelEnv
from agents.rl.mappo import MAPPOActor, CommMAPPOActor, MAPPOCommMAC, MAPPOCritic, MAPPOTrainer, RolloutBuffer

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    run = wandb.init(project="BeerGame_Research", config=dict(cfg), name=cfg.agent.algorithm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.env.demand_type = "poisson"
    env = BeerGameParallelEnv(cfg.env)
    algo = cfg.agent.algorithm.lower()
    
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
    critic_in = local_dim if algo == "ippo" else local_dim * len(env.agents)
    
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
        
        while True:
            obs_array = np.stack([obs[a] for a in env.agents])
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0).to(device)
            state = np.concatenate([obs[a] for a in sorted(env.agents)])
            
            with torch.no_grad():
                if algo == "comm_mappo":
                    dist_action, dist_comm, comm_actions_raw, next_hiddens, masked_msg_in, safe_logs = actor(obs_tensor, hiddens_tensor)
                    actions_raw = dist_action.sample()
                    
                    actions_val = actions_raw.view(1, len(env.agents), 1)
                    comm_actions = comm_actions_raw.view(1, len(env.agents))
                    episode_messages.append(safe_logs)
                else:
                    dist_action, next_hiddens_raw = actor(obs_tensor.view(-1, local_dim), hiddens_tensor.view(-1, cfg.agent.hidden_dim))
                    actions_raw = dist_action.sample()
                    
                    actions_val = actions_raw.view(1, len(env.agents), 1)
                    next_hiddens = next_hiddens_raw.view(1, len(env.agents), -1)
            
            acts = {a: [actions_val[0, i].cpu().item()] for i, a in enumerate(env.agents)}
            
            # THE FIX: ALL Buffer Appends Must Occur Inside This Loop
            for i, a in enumerate(env.agents):
                buffer.local_obs.append(obs_tensor[0, i].unsqueeze(0))
                buffer.hidden_states.append(hiddens_tensor[0, i].unsqueeze(0))
                buffer.actions.append(actions_val[0, i].unsqueeze(0))
                
                if algo == "comm_mappo":
                    buffer.comm_in.append(masked_msg_in[0, i].unsqueeze(0))
                    buffer.comm_actions.append(comm_actions[0, i].view(1))
                    
                    lp_act = dist_action.log_prob(actions_raw)[i]
                    lp_comm = dist_comm.log_prob(comm_actions_raw)[i].view(1)
                    buffer.log_probs.append((lp_act + lp_comm).view(1, 1))
                else:
                    lp_act = dist_action.log_prob(actions_raw)[i].view(1, 1)
                    buffer.log_probs.append(lp_act)

                # FIX: Align global_states to 200 elements by appending inside the agent loop
                buffer.global_states.append(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))

            hiddens_tensor = next_hiddens
            
            next_obs, rewards, terms, truncs, infos = env.step(acts)
            
            true_step_cost = sum(infos[a]["local_cost"] for a in env.agents)
            ep_cost += true_step_cost
            
            for i, a in enumerate(env.agents): 
                local_cost = infos.get(a, {}).get("local_cost", 0.0)
                ep_agent_costs[a] += local_cost 
                raw_cost = abs(local_cost) if algo == "ippo" else abs(rewards[a])
                scaled_reward = -raw_cost / 100.0 
                buffer.rewards.append(scaled_reward)
                buffer.is_terminals.append(terms[a])
                
            obs = next_obs
            if any(terms.values()) or any(truncs.values()): break
            
        actor_loss, critic_loss = trainer.update(buffer, ep)
        actor_scheduler.step()
        critic_scheduler.step()
        
        log_dict = {"Cost": ep_cost, "Actor_Loss": actor_loss, "Critic_Loss": critic_loss, "Actor_LR": actor_scheduler.get_last_lr()[0]}
        for a, cost in ep_agent_costs.items(): log_dict[f"Cost/{a}"] = cost
        wandb.log(log_dict)
        
        cost_history.append(ep_cost)
        avg_cost = sum(cost_history) / len(cost_history)
        
        if ep == warm_up: 
            best_avg_cost, since_imp = float('inf'), 0
            
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