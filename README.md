# BeerGame

This repository contains a research-grade framework evaluating cooperative Multi-Agent Reinforcement Learning (MARL) agents tasked with mitigating the bullwhip effect in a classic four-stage supply chain simulation (The Beer Game). The project introduces and benchmarks categorical communication protocols against traditional baseline MARL algorithms under Centralized Training, Decentralized Execution (CTDE) constraints.

## Core Methodology

The environment models a classic decentralized supply chain consisting of four distinct stages: Retailer, Wholesaler, Distributor, and Manufacturer. Each stage operates under partial observability (Dec-POMDP), managing local inventory levels, incoming orders, backlogs, and shipment delays while minimizing cumulative holding and backlog costs.

The repository evaluates two core algorithmic families across communicating and non-communicating variants:

    Value-Based Architectures (QMIX): Employs an explicit centralized mixing network to factor joint action-values (Qtot​) monotonically, enforcing structural stability under structural non-stationarity.

    Policy-Based Architectures (MAPPO): Imploys a centralized critic sharing global environmental states to guide decentralized policy updates across cooperative actor networks.

## Communication Protocols

To address information asymmetry across independent supply chain entities, communicating variants (comm_qmix and comm_mappo) incorporate a differentiable, continuous-to-categorical communication channel via discrete vocabulary layers.

    Channel Capacity Controls (vocab_size): Evaluates structural bottlenecks across variable codebooks (1, 3, or 5 tokens). A vocabulary size of 1 explicitly maps to a zero-information channel, acting as an architectural ablation control.

    Regularization (L2​ Penalty): Employs an internal differentiable message magnitude penalty (λ∑m2) to counteract token collapse and discourage spamming uninformative signals across the channel.

## File Structure

```
├── agents/
│   └── rl/
│       ├── comm_utils.py          # Unified discrete vocabulary mapping definitions
│       ├── mappo.py               # Actor-Critic network variants and MAPPOTrainer loops
│       ├── qmix.py                # Value network definitions and centralized mixing modules
│       ├── train_comm_qmix.py     # Training wrapper for communicating QMIX networks
│       ├── train_ppo.py           # Unified training wrapper for MAPPO and IPPO variants
│       └── train_qmix.py          # Training wrapper for baseline non-communicating QMIX
├── configs/                       # Hydra environment configuration mappings
├── envs/
│   └── beer_game_env.py       # Custom PettingZoo/Gymnasium multi-agent Beer Game environment
├── sweep_comm_mappo.yaml          # W&B Bayesian Sweep configuration for Comm-MAPPO
├── sweep_comm_qmix.yaml           # W&B Bayesian Sweep configuration for Comm-QMIX
├── sweep_mappo.yaml               # W&B Bayesian Sweep configuration for baseline MAPPO
├── sweep_qmix.yaml                # W&B Bayesian Sweep configuration for baseline QMIX
└── requirements.txt               # Complete Python system and training dependencies
```

## Dependencies & Environment Installation

The codebase relies on PyTorch configured with CUDA 12.1 for tensor processing, alongside PettingZoo and Gymnasium for multi-agent environment compliance.

An execution sandbox environment can be compiled directly via the following pipeline:

```
# 1. Ensure system utilities are installed
sudo apt update && sudo apt install -y git tmux python3-pip python3-venv

# 2. Set up an isolated Python Virtual Environment
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade local installer and pull pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Automated Hyperparameter Optimization via W&B

Experiment telemetry and hyperparameter tracking are natively driven via Weights & Biases (W&B) Bayesian sweeps. The command configurations rely entirely on Hydra command-line overrides to dynamically mutate target configs without mutating on-disk script architecture.
Launching a Master Sweep Controller

To generate an active Sweep ID on the W&B master cluster, initialize the targeted configuration file from your deployment node:

```
# For communicating value-based sweeps
wandb sweep sweep_comm_qmix.yaml

# For communicating policy-based sweeps
wandb sweep sweep_comm_mappo.yaml
```

## Running Parallel Compute Workers

To run multiple agents concurrently across your available CPU cores or distributed Spot instances without shell termination errors, wrap execution blocks within detached tmux sessions:

```
# 1. Create a persistent window manager session
tmux new -s marl_sweeps

# 2. Inside your tmux window, activate the virtual sandbox
source venv/bin/activate

# 3. Spin up an autonomous agent listener pointing to your W&B Sweep ID
./venv/bin/python -m wandb agent <YOUR_ENTITY_OR_USERNAME>/BeerGame_Research/<SWEEP_ID>
```

To scale up processing throughput horizontally, open a new tmux window (Ctrl+B, then C) inside the session and repeat step 3 to attach up to 4 concurrent parallel execution agents within the same instance.

To safely drop out of the terminal session while allowing background training sweeps to run continuously, detach the container using Ctrl+B, followed by D. To resume standard visual monitoring upon re-establishing a cloud connection, use:

```
tmux attach -t marl_sweeps
```