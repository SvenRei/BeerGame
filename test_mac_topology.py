import unittest
import torch
import numpy as np

# Import both architectures
from agents.rl.mappo import CommMAPPOActor, MAPPOCommMAC
from agents.rl.qmix import CommQMixLocalAgent, QMixCommMAC

# ==============================================================================
# 1. MAPPO TOPOLOGY TESTS
# ==============================================================================
class TestMAPPOCommunicationTopology(unittest.TestCase):
    
    def setUp(self):
        self.obs_dim = 7 # standard obs size
        self.hidden_dim = 64
        self.vocab_size = 3
        
        base_actor = CommMAPPOActor(self.obs_dim, self.hidden_dim, vocab_size=self.vocab_size)
        self.mac = MAPPOCommMAC(base_actor, vocab_size=self.vocab_size, num_agents=4)
        
        # Initialize the temporal buffer for a batch size of 1
        self.mac.init_buffer(batch_size=1, device="cpu")

    def test_01_mappo_strict_tridiagonal_routing(self):
        """
        LITERATURE TEST: Verifies that a message sent by the Retailer ONLY reaches 
        the Wholesaler, and mathematically cannot jump to the Distributor or Manufacturer.
        """
        self.mac.msg_buffer[0, 0, 0] = 1.0 # Retailer distress signal
        self.mac.msg_buffer[0, 1, 0] = 0.0 # Wholesaler
        self.mac.msg_buffer[0, 2, 0] = 0.0 # Distributor
        self.mac.msg_buffer[0, 3, 0] = 0.0 # Manufacturer
        
        dummy_obs = torch.zeros(1, 4, self.obs_dim)
        dummy_hiddens = torch.zeros(1, 4, self.hidden_dim)
        
        _, _, _, _, masked_msgs, _ = self.mac(dummy_obs, dummy_hiddens, test_mode=True)
        
        wholesaler_received = masked_msgs[0, 1, 0].item()
        distributor_received = masked_msgs[0, 2, 0].item()
        manufacturer_received = masked_msgs[0, 3, 0].item()
        
        self.assertEqual(wholesaler_received, 1.0, "MAPPO Topology Flaw: Wholesaler did not receive the Retailer's signal.")
        self.assertEqual(distributor_received, 0.0, "MAPPO Topology Flaw: Telepathy detected! Distributor heard the Retailer.")
        self.assertEqual(manufacturer_received, 0.0, "MAPPO Topology Flaw: Telepathy detected! Manufacturer heard the Retailer.")

    def test_02_mappo_vocabulary_constraint(self):
        """
        ENGINEERING TEST: Verifies the MAC strictly enforces the discrete vocabulary 
        output from the continuous network distribution.
        """
        dummy_obs = torch.zeros(1, 4, self.obs_dim)
        dummy_hiddens = torch.zeros(1, 4, self.hidden_dim)
        
        _, _, _, _, _, safe_logs = self.mac(dummy_obs, dummy_hiddens, test_mode=False)
        
        valid_vocab = {-1.0, 0.0, 1.0}
        
        for i in range(4):
            agent_msg = safe_logs[0, i, 0]
            self.assertIn(agent_msg, valid_vocab, f"MAPPO Vocabulary Flaw: Agent emitted illegal float {agent_msg}")

# ==============================================================================
# 2. QMIX TOPOLOGY TESTS
# ==============================================================================
class TestQMixCommunicationTopology(unittest.TestCase):
    
    def setUp(self):
        self.obs_dim = 7 
        self.hidden_dim = 64
        self.vocab_size = 3
        self.n_actions = 51 # QMIX requires discrete physical action bins
        
        base_agent = CommQMixLocalAgent(self.obs_dim, self.hidden_dim, self.n_actions, vocab_size=self.vocab_size)
        self.mac = QMixCommMAC(base_agent, num_agents=4)
        
        self.mac.init_buffer(batch_size=1, device="cpu")

    def test_03_qmix_strict_tridiagonal_routing(self):
        """
        LITERATURE TEST: Verifies strict adjacent routing within the QMIX DIAL architecture.
        """
        self.mac.msg_buffer[0, 0, 0] = 1.0 # Retailer distress signal
        self.mac.msg_buffer[0, 1, 0] = 0.0 # Wholesaler
        self.mac.msg_buffer[0, 2, 0] = 0.0 # Distributor
        self.mac.msg_buffer[0, 3, 0] = 0.0 # Manufacturer
        
        # In QMixCommMAC, the mask is applied directly at the start of the forward pass.
        # We explicitly evaluate the matrix multiplication that guards the inputs:
        masked_msgs = torch.matmul(self.mac.adj_mask, self.mac.msg_buffer)
        
        wholesaler_received = masked_msgs[0, 1, 0].item()
        distributor_received = masked_msgs[0, 2, 0].item()
        manufacturer_received = masked_msgs[0, 3, 0].item()
        
        self.assertEqual(wholesaler_received, 1.0, "QMIX Topology Flaw: Wholesaler did not receive the Retailer's signal.")
        self.assertEqual(distributor_received, 0.0, "QMIX Topology Flaw: Telepathy detected! Distributor heard the Retailer.")
        self.assertEqual(manufacturer_received, 0.0, "QMIX Topology Flaw: Telepathy detected! Manufacturer heard the Retailer.")

    def test_04_qmix_vocabulary_constraint(self):
        """
        ENGINEERING TEST: Verifies the QMIX MAC strictly enforces the discrete vocabulary 
        via the Gumbel-Softmax bottleneck layer.
        """
        dummy_obs = torch.zeros(1, 4, self.obs_dim)
        dummy_hiddens = torch.zeros(1, 4, self.hidden_dim)
        
        # Run forward pass. QMIX returns: q_vals, next_hiddens, safe_logs
        _, _, safe_logs = self.mac(dummy_obs, dummy_hiddens)
        
        valid_vocab = {-1.0, 0.0, 1.0}
        
        for i in range(4):
            agent_msg = safe_logs[0, i, 0]
            self.assertIn(agent_msg, valid_vocab, f"QMIX Vocabulary Flaw: Agent emitted illegal float {agent_msg}")

if __name__ == '__main__':
    unittest.main(verbosity=2)