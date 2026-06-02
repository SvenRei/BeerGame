# Import the core PyTorch deep learning library
import torch
# Import the neural network module from PyTorch to build our layers
import torch.nn as nn
# Import the functional module for activation functions (like ReLU)
import torch.nn.functional as F

# ==============================================================================
# 1. THE LOCAL AGENT NETWORK (Used for both Training and Evaluation)
# ==============================================================================
# This class defines the "brain" of each individual supply chain agent.
class QMixLocalAgent(nn.Module):
    # Initialize the network with the size of the observation, hidden memory, and number of possible actions
    def __init__(self, local_dim, hidden_dim, n_actions):
        # Call the initialization method of the parent nn.Module class
        super(QMixLocalAgent, self).__init__()
        
        # Save the hidden dimension size for use in tensor reshaping later
        self.hidden_dim = hidden_dim
        # Save the number of discrete actions (e.g., 11 bins: order 0, 10, 20... 100)
        self.n_actions = n_actions
        
        # Define the first Fully Connected (Linear) layer to process the raw observation
        self.fc1 = nn.Linear(local_dim, hidden_dim)
        # Define the Gated Recurrent Unit (GRU) to give the agent memory of the 4-week delay
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        # Define the final output layer that maps the memory to a Q-value for each possible action
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    # Define the forward pass (how data moves through the network at each time step)
    def forward(self, obs, hidden_state):
        # Pass the observation through the first layer and apply a ReLU activation function
        x = F.relu(self.fc1(obs))
        
        # Reshape the incoming hidden state to ensure it matches the GRU's expected 2D shape (Batch, Hidden_Dim)
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        
        # Pass the processed observation and previous memory into the GRU to generate the new memory state
        h = self.gru(x, h_in)
        
        # Pass the new memory state through the final layer to get the Q-values for all discrete actions
        q_values = self.fc2(h)
        
        # Return the calculated Q-values and the updated memory state (reshaped to maintain dimensions)
        return q_values, h.unsqueeze(1)


# ==============================================================================
# 2. THE MIXING HYPERNETWORK (Used ONLY during Training)
# ==============================================================================
# This class enforces the Monotonic Value Decomposition constraint.
# It takes the local Q-values from all 4 agents and combines them into one global Q-total.
class QMixer(nn.Module):
    # Initialize the mixer with the number of agents, global state size, and internal embedding sizes
    def __init__(self, n_agents, state_dim, mixing_embed_dim=256, hypernet_embed=64):
        # Call the initialization method of the parent nn.Module class
        super(QMixer, self).__init__()
        
        # Save the number of agents in the environment (4 for the Beer Game)
        self.n_agents = n_agents
        # Save the size of the global 24-dimensional supply chain state
        self.state_dim = state_dim
        # Save the size of the mixing layer's internal dimension
        self.mixing_embed_dim = mixing_embed_dim
        
        # --- HYPERNETWORK 1: Generates weights for the first mixing layer ---
        # A Sequential block that takes the global state and outputs weights
        self.hyper_w_1 = nn.Sequential(
            # First linear layer of the hypernetwork
            nn.Linear(state_dim, hypernet_embed),
            # ReLU activation for non-linear processing
            nn.ReLU(),
            # Output layer generating enough weights for (n_agents * mixing_embed_dim) matrix
            nn.Linear(hypernet_embed, n_agents * mixing_embed_dim)
        )
        
        # --- HYPERNETWORK 2: Generates weights for the second (final) mixing layer ---
        # A Sequential block taking the global state and outputting the final weights
        self.hyper_w_2 = nn.Sequential(
            # First linear layer of the second hypernetwork
            nn.Linear(state_dim, hypernet_embed),
            # ReLU activation
            nn.ReLU(),
            # Output layer generating weights for a (mixing_embed_dim * 1) matrix
            nn.Linear(hypernet_embed, mixing_embed_dim)
        )
        
        # --- BIAS GENERATORS ---
        # Generates the bias for the first mixing layer (simply a linear transformation)
        self.hyper_b_1 = nn.Linear(state_dim, mixing_embed_dim)
        
        # Generates the final global bias (V-state) using a 2-layer network with ReLU
        self.hyper_b_2 = nn.Sequential(
            # First linear layer for the final bias
            nn.Linear(state_dim, mixing_embed_dim),
            # ReLU activation
            nn.ReLU(),
            # Output layer collapsing the bias down to a single global value (1)
            nn.Linear(mixing_embed_dim, 1)
        )

    # Define the forward pass for the mixer (combining the local Q-values)
    def forward(self, agent_qs, states):
        # Check the batch size from the incoming agent Q-values tensor
        batch_size = agent_qs.size(0)
        # Reshape the global states to ensure it aligns with the batch size
        states = states.reshape(-1, self.state_dim)
        # Reshape the agent Q-values into a column vector: (Batch, 1, 4 agents)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        
        # --- LAYER 1 MIXING ---
        # Pass the global state through Hypernetwork 1 to generate raw weights
        w1 = self.hyper_w_1(states)
        # CRITICAL QMIX STEP: Apply absolute value to enforce strictly positive weights (Monotonicity)
        w1 = torch.abs(w1)
        # Reshape the positive weights into the correct matrix shape: (Batch, 4 agents, 256)
        w1 = w1.view(-1, self.n_agents, self.mixing_embed_dim)
        
        # Pass the global state through the bias generator
        b1 = self.hyper_b_1(states)
        # Reshape the bias to match the matrix multiplication shape: (Batch, 1, 256)
        b1 = b1.view(-1, 1, self.mixing_embed_dim)
        
        # Perform batched matrix multiplication of the Local Q-values by the positive weights
        hidden = torch.bmm(agent_qs, w1)
        # Add the generated bias to the mixed hidden state
        hidden = hidden + b1
        # Apply an Exponential Linear Unit (ELU) activation function to the hidden layer
        hidden = F.elu(hidden)
        
        # --- LAYER 2 MIXING ---
        # Pass the global state through Hypernetwork 2 to generate raw weights for the final layer
        w2 = self.hyper_w_2(states)
        # CRITICAL QMIX STEP: Apply absolute value again to ensure monotonic gradients
        w2 = torch.abs(w2)
        # Reshape the weights into the final matrix shape: (Batch, 256, 1)
        w2 = w2.view(-1, self.mixing_embed_dim, 1)
        
        # Pass the global state through the final bias generator (the V-state)
        b2 = self.hyper_b_2(states)
        # Reshape the final bias: (Batch, 1, 1)
        b2 = b2.view(-1, 1, 1)
        
        # Multiply the hidden layer by the final positive weights
        q_tot = torch.bmm(hidden, w2)
        # Add the final global bias
        q_tot = q_tot + b2
        
        # Reshape the final global Q-total back to the standard Batch size dimensions
        q_tot = q_tot.view(batch_size, -1, 1)
        
        # Return the mathematically guaranteed monotonic global Q-value
        return q_tot