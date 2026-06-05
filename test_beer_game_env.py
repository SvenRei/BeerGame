import unittest
import numpy as np

from envs.beer_game_env import BeerGameParallelEnv

AGENTS = ["retailer", "wholesaler", "distributor", "manufacturer"]
DOWNSTREAM_ORDERING_AGENTS = ["retailer", "wholesaler", "distributor"]

class RecordingRNG:
    def __init__(self, integer_value=7):
        self.integer_value = integer_value
        self.poisson_lams = []
        self.integer_calls = []
    def poisson(self, lam):
        self.poisson_lams.append(lam)
        return int(lam)
    def integers(self, low, high=None):
        self.integer_calls.append((low, high))
        if high is None: return min(self.integer_value, low - 1)
        return min(max(self.integer_value, low), high - 1)

class TestBeerGameEnvironment(unittest.TestCase):
    def setUp(self):
        self.config = {
            "horizon": 50,
            "max_order": 100,
            "holding_cost": 0.5,
            "backorder_cost": 1.0,
            "lookahead": 4,
            "demand_type": "step",
            "reward_alpha": 0.5,
            "jittery_lead_time": False,
        }
        self.env = BeerGameParallelEnv(self.config)
        self.obs, _ = self.env.reset(seed=42)
        self.agents = AGENTS[:]

    def _make_env(self, overrides=None, seed=42):
        config = self.config.copy()
        if overrides: config.update(overrides)
        env = BeerGameParallelEnv(config)
        obs, infos = env.reset(seed=seed)
        return env, obs, infos

    def _actions_for_order(self, env, order_qty: int):
        return {agent: np.array([order_qty / env.max_order], dtype=np.float32) for agent in env.agents}

    def _actions_for_orders(self, env, orders_by_agent):
        actions = self._actions_for_order(env, 0)
        for agent, qty in orders_by_agent.items():
            actions[agent] = np.array([qty / env.max_order], dtype=np.float32)
        return actions

    def _clear_all_pipelines(self, env):
        for agent in env.possible_agents:
            env.shipment_pipelines[agent].pipeline = {}
            env.order_pipelines[agent].pipeline = {}
            env.unfulfilled_orders[agent] = 0

    def _pipeline_snapshot(self, env):
        return {agent: dict(env.shipment_pipelines[agent].pipeline) for agent in env.possible_agents}

    def _state_snapshot(self, env):
        return {
            "current_step": getattr(env, "current_step", None),
            "inventory": dict(getattr(env, "inventory", {})),
            "backlog": dict(getattr(env, "backlog", {})),
            "unfulfilled_orders": dict(getattr(env, "unfulfilled_orders", {})),
            "current_incoming_order": dict(getattr(env, "current_incoming_order", {})),
            "shipment_pipelines": {a: dict(env.shipment_pipelines[a].pipeline) for a in env.possible_agents},
            "order_pipelines": {a: dict(env.order_pipelines[a].pipeline) for a in env.possible_agents},
        }

    # ==============================================================================
    # MIT / STERMAN INITIALIZATION TESTS
    # ==============================================================================
    def test_01_sterman_steady_state_inventory_and_backlog(self):
        for agent in self.env.agents:
            self.assertEqual(self.env.inventory[agent], 12)
            self.assertEqual(self.env.backlog[agent], 0)

    def test_02_initial_delay_boxes_are_loaded_with_four_cases(self):
        for agent in self.env.possible_agents:
            self.assertEqual(self.env.shipment_pipelines[agent].pipeline.get(1), 4)
            self.assertEqual(self.env.shipment_pipelines[agent].pipeline.get(2), 4)
        for agent in DOWNSTREAM_ORDERING_AGENTS:
            self.assertEqual(self.env.order_pipelines[agent].pipeline.get(1), 4)
            self.assertEqual(self.env.order_pipelines[agent].pipeline.get(2), 4)

    def test_03_initial_open_order_ledger_includes_all_known_supply_line(self):
        expected = {"retailer": 16, "wholesaler": 16, "distributor": 16, "manufacturer": 12}
        self.assertEqual(self.env.unfulfilled_orders, expected)

    def test_04_initial_observation_layout_and_space_match(self):
        expected_dim = 4 + self.env.lookahead
        for agent, obs in self.obs.items():
            self.assertEqual(obs.shape, (expected_dim,))
            self.assertEqual(self.env.observation_space(agent).shape, (expected_dim,))
            self.assertEqual(obs[0], 12.0)
            self.assertEqual(obs[1], 0.0)
            self.assertEqual(obs[3], 4.0)

    def test_05_zero_demand_reset_observation_does_not_lie_about_customer_order(self):
        env, obs, _ = self._make_env({"demand_type": "zero"}, seed=0)
        self.assertEqual(obs["retailer"][3], 0.0)

    # ==============================================================================
    # PIPELINE TIMING TESTS
    # ==============================================================================
    def test_06_strict_order_delay_to_upstream_supplier(self):
        actions = {a: np.array([0.9], dtype=np.float32) for a in self.env.agents}
        self.env.step(actions)
        pipeline = self.env.order_pipelines["retailer"].pipeline
        self.assertIn(3, pipeline)
        self.assertEqual(pipeline[3], 90)

    def test_07_strict_shipment_delay_to_downstream_customer(self):
        actions = self._actions_for_order(self.env, 0)
        self.env.step(actions)
        pipeline = self.env.shipment_pipelines["retailer"].pipeline
        self.assertEqual(pipeline.get(3, 0), 4)

    def test_08_factory_initial_production_request_from_reset_arrives(self):
        env, _, _ = self._make_env({"demand_type": "zero", "horizon": 10}, seed=0)
        for _ in range(3):
            env.step(self._actions_for_order(env, 0))
        self.assertEqual(env.inventory["manufacturer"], 16)

    def test_09_factory_new_production_request_has_mit_request_delay_plus_production_delay(self):
        env, _, _ = self._make_env({"demand_type": "zero", "horizon": 10}, seed=0)
        self._clear_all_pipelines(env)
        env.inventory["manufacturer"] = 0
        env.backlog["manufacturer"] = 0
        actions = self._actions_for_orders(env, {"manufacturer": 100})
        env.step(actions)
        actions = self._actions_for_order(env, 0)
        env.step(actions)
        self.assertEqual(env.inventory["manufacturer"], 0)
        env.step(actions)
        self.assertEqual(env.inventory["manufacturer"], 0)
        env.step(actions)
        self.assertEqual(env.inventory["manufacturer"], 100)

    # ==============================================================================
    # ACTION TRANSLATION TESTS
    # ==============================================================================
    def test_10_continuous_action_rounds_to_discrete_order_quantity(self):
        actions = {
            "retailer": np.array([0.457], dtype=np.float32),
            "wholesaler": np.array([0.10], dtype=np.float32),
            "distributor": np.array([0.0], dtype=np.float32),
            "manufacturer": np.array([0.0], dtype=np.float32),
        }
        self.env.step(actions)
        self.assertEqual(self.env.order_pipelines["retailer"].pipeline[3], 46)
        self.assertEqual(self.env.order_pipelines["wholesaler"].pipeline[3], 10)

    def test_11_action_clipping_enforces_box_bounds(self):
        actions = {
            "retailer": np.array([-1.0], dtype=np.float32),
            "wholesaler": np.array([2.0], dtype=np.float32),
            "distributor": np.array([0.5], dtype=np.float32),
            "manufacturer": np.array([1.5], dtype=np.float32),
        }
        self.env.step(actions)
        self.assertEqual(self.env.order_pipelines["retailer"].pipeline.get(3, 0), 0)
        self.assertEqual(self.env.order_pipelines["wholesaler"].pipeline.get(3, 0), 100)
        self.assertEqual(self.env.order_pipelines["distributor"].pipeline.get(3, 0), 50)
        self.assertEqual(self.env.order_pipelines["manufacturer"].pipeline.get(2, 0), 100)

    def test_12_absolute_zero_action_places_no_new_order(self):
        self.env.inventory["retailer"] = 0
        self.env.backlog["retailer"] = 50
        actions = self._actions_for_order(self.env, 0)
        self.env.step(actions)
        self.assertEqual(self.env.order_pipelines["retailer"].pipeline.get(3, 0), 0)

    # ==============================================================================
    # PHYSICAL INVENTORY & BACKLOG TESTS
    # ==============================================================================
    def test_13_retailer_demand_depletion_in_initial_equilibrium(self):
        actions = self._actions_for_order(self.env, 0)
        self.env.step(actions)
        self.assertEqual(self.env.inventory["retailer"], 12)
        self.assertEqual(self.env.backlog["retailer"], 0)

    def test_14_backlog_accumulates_when_inventory_is_insufficient(self):
        self.env.inventory["retailer"] = 0
        self.env.shipment_pipelines["retailer"].pipeline = {}
        self.env.unfulfilled_orders["retailer"] = 0
        actions = self._actions_for_order(self.env, 0)
        self.env.step(actions)
        self.assertEqual(self.env.inventory["retailer"], 0)
        self.assertEqual(self.env.backlog["retailer"], 4)

    def test_15_backlog_recovery_uses_incoming_goods_before_fulfillment(self):
        env, _, _ = self._make_env({"demand_type": "zero"}, seed=0)
        env.inventory["retailer"] = 0
        env.backlog["retailer"] = 10
        env.shipment_pipelines["retailer"].pipeline = {1: 20}
        env.unfulfilled_orders["retailer"] = 20
        actions = self._actions_for_order(env, 0)
        env.step(actions)
        self.assertEqual(env.backlog["retailer"], 0)
        self.assertEqual(env.inventory["retailer"], 10)
        self.assertEqual(env.unfulfilled_orders["retailer"], 0)

    def test_16_backlog_is_cumulative_not_replaced(self):
        self.env.inventory["retailer"] = 0
        self.env.backlog["retailer"] = 10
        self.env.shipment_pipelines["retailer"].pipeline = {}
        self.env.unfulfilled_orders["retailer"] = 0
        actions = self._actions_for_order(self.env, 0)
        self.env.step(actions)
        self.assertEqual(self.env.backlog["retailer"], 14)

    def test_17_retailer_backlog_does_not_automatically_create_orders(self):
        self.env.inventory["retailer"] = 0
        self.env.backlog["retailer"] = 500
        actions = self._actions_for_order(self.env, 0)
        self.env.step(actions)
        self.assertEqual(self.env.order_pipelines["retailer"].pipeline.get(3, 0), 0)

    def test_18_upstream_supplier_backlog_does_not_make_customer_forget_open_order(self):
        self.env.order_pipelines["wholesaler"].pipeline = {}
        self.env.shipment_pipelines["wholesaler"].pipeline = {}
        self.env.inventory["distributor"] = 0
        self.env.shipment_pipelines["distributor"].pipeline = {}
        self.env.unfulfilled_orders["wholesaler"] = 0
        actions = self._actions_for_order(self.env, 0)
        actions["wholesaler"] = np.array([1.0], dtype=np.float32)
        self.env.step(actions) 
        actions = self._actions_for_order(self.env, 0)
        self.env.step(actions) 
        self.env.shipment_pipelines["distributor"].pipeline = {}
        self.env.step(actions) 
        wholesaler_obs = self.env._build_obs("wholesaler")
        self.assertEqual(wholesaler_obs[2], 100.0)

    def test_19_open_order_ledger_never_goes_negative_for_downstream_nodes(self):
        env, _, _ = self._make_env({"horizon": 10, "demand_type": "zero"}, seed=0)
        for _ in range(5):
            env.step(self._actions_for_order(env, 0))
            for agent in DOWNSTREAM_ORDERING_AGENTS:
                self.assertGreaterEqual(env.unfulfilled_orders[agent], 0)

    def test_20_open_order_ledger_drains_to_zero_without_new_orders(self):
        env, _, _ = self._make_env({"horizon": 10, "demand_type": "zero"}, seed=0)
        for _ in range(5):
            env.step(self._actions_for_order(env, 0))
        for agent in AGENTS:
            self.assertEqual(env.unfulfilled_orders[agent], 0)

    # ==============================================================================
    # MIT DEMAND DECK & DECISION-INFORMATION TESTS
    # ==============================================================================
    def test_21_constant_order_4_preserves_equilibrium_until_week_4(self):
        self.env.reset(seed=0)
        for _ in range(4):
            self.env.step(self._actions_for_order(self.env, 4))
            for agent in AGENTS:
                self.assertEqual(self.env.inventory[agent], 12)
                self.assertEqual(self.env.backlog[agent], 0)

    def test_22_mit_customer_demand_jumps_in_week_5(self):
        self.env.reset(seed=0)
        for _ in range(4):
            self.env.step(self._actions_for_order(self.env, 4))
        self.assertEqual(self.env.inventory["retailer"], 12)
        self.env.step(self._actions_for_order(self.env, 4))
        self.assertEqual(self.env.inventory["retailer"], 8)
        self.assertEqual(self.env.backlog["retailer"], 0)

    def test_23_incoming_order_observation_changes_after_mail_is_opened(self):
        env_low, _, _ = self._make_env({"horizon": 10, "demand_type": "zero", "lookahead": 2}, seed=0)
        env_high, _, _ = self._make_env({"horizon": 10, "demand_type": "zero", "lookahead": 2}, seed=0)
        env_low.order_pipelines["retailer"].pipeline.clear()
        env_high.order_pipelines["retailer"].pipeline.clear()
        env_low.order_pipelines["retailer"].add_shipment(1, 1, 1)
        env_high.order_pipelines["retailer"].add_shipment(1, 99, 1)
        obs_low, _, _, _, _ = env_low.step(self._actions_for_order(env_low, 0))
        obs_high, _, _, _, _ = env_high.step(self._actions_for_order(env_high, 0))
        self.assertNotEqual(obs_low["wholesaler"][3], obs_high["wholesaler"][3])
        self.assertEqual(obs_low["wholesaler"][3], 1)
        self.assertEqual(obs_high["wholesaler"][3], 99)

    def test_24_decision_observation_exposes_week5_demand_before_week5_order(self):
        env, obs, _ = self._make_env({"demand_type": "step", "horizon": 10}, seed=0)
        self.assertEqual(obs["retailer"][3], 4)
        for _ in range(4):
            obs, _, _, _, _ = env.step(self._actions_for_order(env, 4))
        self.assertEqual(obs["retailer"][3], 8)

    # ==============================================================================
    # FINANCIAL ACCOUNTING & REWARD TESTS
    # ==============================================================================
    def test_25_financial_accounting_uses_true_inventory_and_backlog(self):
        env, _, _ = self._make_env({"demand_type": "zero"}, seed=0)
        env.inventory["retailer"] = 10
        env.backlog["retailer"] = 0
        env.inventory["wholesaler"] = 0
        env.backlog["wholesaler"] = 20
        actions = self._actions_for_order(env, 0)
        _, _, _, _, infos = env.step(actions)
        self.assertEqual(infos["retailer"]["local_cost"], 7.0)
        self.assertEqual(infos["wholesaler"]["local_cost"], 20.0)

    def test_26_observation_clipping_does_not_clip_physics_or_accounting(self):
        env, _, _ = self._make_env({"demand_type": "zero"}, seed=0)
        env.inventory["retailer"] = 5000
        actions = self._actions_for_order(env, 0)
        obs, _, _, _, infos = env.step(actions)
        self.assertEqual(obs["retailer"][0], 2000.0)
        self.assertEqual(env.inventory["retailer"], 5004)
        self.assertEqual(infos["retailer"]["local_cost"], 2502.0)

    def test_27_default_reward_is_mit_team_total_cost(self):
        env = BeerGameParallelEnv({"horizon": 10, "max_order": 100, "holding_cost": 0.5, "backorder_cost": 1.0, "demand_type": "zero", "jittery_lead_time": False})
        env.reset(seed=0)
        self._clear_all_pipelines(env)
        for agent in AGENTS:
            env.inventory[agent] = 0
            env.backlog[agent] = 0
        env.inventory["retailer"] = 10 
        env.backlog["wholesaler"] = 20 
        _, rewards, _, _, infos = env.step(self._actions_for_order(env, 0))
        total_cost = sum(info["local_cost"] for info in infos.values())
        self.assertEqual(total_cost, 25.0)
        for agent in AGENTS:
            self.assertEqual(rewards[agent], -25.0)

    def test_28_explicit_marl_reward_alpha_blending_is_supported(self):
        env, _, _ = self._make_env({"demand_type": "zero", "reward_alpha": 0.5}, seed=0)
        self._clear_all_pipelines(env)
        for agent in AGENTS:
            env.inventory[agent] = 0
            env.backlog[agent] = 0
        env.inventory["retailer"] = 10 
        env.backlog["wholesaler"] = 20 
        _, rewards, _, _, _ = env.step(self._actions_for_order(env, 0))
        self.assertEqual(rewards["retailer"], -15.0)
        self.assertEqual(rewards["wholesaler"], -22.5)
        self.assertEqual(rewards["distributor"], -12.5)
        self.assertEqual(rewards["manufacturer"], -12.5)

    # ==============================================================================
    # LOCAL OBSERVABILITY / POMDP TESTS
    # ==============================================================================
    def test_29_local_observation_has_no_telepathic_supplier_state(self):
        base_obs = self.env._build_obs("retailer").copy()
        self.env.inventory["wholesaler"] = 999
        self.env.backlog["wholesaler"] = 999
        new_obs = self.env._build_obs("retailer")
        np.testing.assert_array_equal(base_obs, new_obs)

    def test_30_retailer_customer_demand_is_not_revealed_to_upstream_agents(self):
        env, obs, _ = self._make_env({"demand_type": "step", "horizon": 10}, seed=0)
        for _ in range(4):
            obs, _, _, _, _ = env.step(self._actions_for_order(env, 4))
        self.assertNotEqual(obs["wholesaler"][3], 8)

    # ==============================================================================
    # STOCHASTIC DEMAND & RNG TESTS
    # ==============================================================================
    def test_31_seeded_envs_are_independent_with_jittery_lead_time(self):
        cfg = {"horizon": 10, "demand_type": "zero", "max_order": 100, "jittery_lead_time": True}
        env_a = BeerGameParallelEnv(cfg)
        env_b = BeerGameParallelEnv(cfg)
        env_a.reset(seed=123)
        env_b.reset(seed=123)
        env_a.step(self._actions_for_order(env_a, 0))
        env_b.step(self._actions_for_order(env_b, 0))
        self.assertEqual(self._pipeline_snapshot(env_a), self._pipeline_snapshot(env_b))

    def test_32_reset_seed_does_not_pollute_global_numpy_rng(self):
        np.random.seed(999)
        expected = np.random.random(5)
        np.random.seed(999)
        self.env.reset(seed=12345)
        actual = np.random.random(5)
        np.testing.assert_allclose(actual, expected)

    def test_33_stochastic_demand_is_actually_realized_in_fulfillment(self):
        env, _, _ = self._make_env({"demand_type": "black_swan", "horizon": 30}, seed=0)
        for _ in range(23):
            env.step(self._actions_for_order(env, 0))
            
        env.np_random = RecordingRNG() 
        env.step(self._actions_for_order(env, 0)) 
        
        black_swan_demand = env._build_obs("retailer")[3]
        self.assertEqual(black_swan_demand, 20)
        
        retailer_backlog_before = env.backlog["retailer"]
        env.step(self._actions_for_order(env, 0)) 
        retailer_backlog_after = env.backlog["retailer"]
        
        self.assertEqual(retailer_backlog_after - retailer_backlog_before, 20)

    def test_34_extreme_chaos_late_random_base_uses_env_rng(self):
        env, _, _ = self._make_env({"demand_type": "extreme_chaos", "horizon": 50}, seed=0)
        for _ in range(32):
            env.step(self._actions_for_order(env, 0))
        
        rng = RecordingRNG(integer_value=7)
        env.np_random = rng
        env.step(self._actions_for_order(env, 0))
        self.assertIn((5, 25), rng.integer_calls)
        self.assertEqual(rng.poisson_lams[-1], 7)

    def test_35_extreme_chaos_non_negative_inventory_in_active_episode(self):
        env, _, _ = self._make_env({"demand_type": "extreme_chaos", "horizon": 200}, seed=0)
        for _ in range(100):
            if not env.agents: break
            env.step(self._actions_for_order(env, 0))
            self.assertGreaterEqual(env.inventory["retailer"], 0)

    # ==============================================================================
    # PETTINGZOO / RL API TESTS
    # ==============================================================================
    def test_36_horizon_truncates_not_terminates(self):
        env, _, _ = self._make_env({"horizon": 2}, seed=0)
        env.step(self._actions_for_order(env, 4))
        _, _, terms, truncs, _ = env.step(self._actions_for_order(env, 4))
        self.assertTrue(all(truncs.values()))
        self.assertFalse(any(terms.values()))

    def test_37_parallel_env_removes_agents_after_horizon(self):
        env, _, _ = self._make_env({"horizon": 1}, seed=0)
        _, _, _, truncs, _ = env.step(self._actions_for_order(env, 4))
        self.assertTrue(all(truncs.values()))
        self.assertEqual(env.agents, [])

    def test_38_reset_restores_agents_after_truncation(self):
        env, _, _ = self._make_env({"horizon": 1}, seed=0)
        env.step(self._actions_for_order(env, 4))
        self.assertEqual(env.agents, [])
        obs, infos = env.reset(seed=1)
        self.assertEqual(env.agents, AGENTS)
        self.assertEqual(set(obs.keys()), set(AGENTS))
        self.assertEqual(set(infos.keys()), set(AGENTS))

    def test_39_step_after_done_raises_runtime_error(self):
        env, _, _ = self._make_env({"horizon": 1}, seed=0)
        env.step(self._actions_for_order(env, 4))
        with self.assertRaises(RuntimeError):
            env.step(self._actions_for_order(env, 4))

    def test_40_action_type_safety_for_common_box_formats(self):
        actions_float64 = {a: np.array([0.5], dtype=np.float64) for a in self.env.agents}
        actions_list = {a: [0.5] for a in self.env.agents}
        try:
            self.env.step(actions_float64)
            self.env.step(actions_list)
        except Exception as exc:
            self.fail(f"Environment crashed on common Gymnasium Box action formats: {exc}")

    def test_41_pettingzoo_parallel_api_contract(self):
        from pettingzoo.test import parallel_api_test
        env = BeerGameParallelEnv({"horizon": 3, "demand_type": "step", "max_order": 100})
        parallel_api_test(env, num_cycles=10)

    # ==============================================================================
    # CENTRALIZED-TRAINING API REGRESSION TESTS
    # ==============================================================================
    def test_42_get_global_state_api_exists_for_ctde(self):
        self.assertTrue(hasattr(self.env, "get_global_state"))

    def test_43_get_global_state_is_true_unclipped_physical_state(self):
        env, _, _ = self._make_env()
        # Set extreme physical state
        env.inventory["retailer"] = 15000 
        global_state = env.get_global_state()
        
        # Verify it returns unclipped raw values rather than just clipped concatenated obs
        self.assertIn(15000.0, global_state)
        expected_len = 4 * (3 + env.config["lookahead"])
        self.assertEqual(global_state.shape, (expected_len,))

    # ==============================================================================
    # ENGINE INTEGRITY & API CONTRACT TESTS (44 - 68)
    # ==============================================================================
    def test_44_missing_demand_type_defaults_to_step(self):
        env, obs, _ = self._make_env({"demand_type": None}, seed=0) 
        self.assertEqual(env.config.get("demand_type", "step"), "step")
        self.assertEqual(obs["retailer"][3], 4.0)

    def test_45_invalid_demand_type_raises_error(self):
        with self.assertRaises(ValueError):
            self._make_env({"demand_type": "stepp"}, seed=0)

    def test_46_config_is_strictly_immutable_after_init(self):
        env, _, _ = self._make_env({"demand_type": "zero"}, seed=0)
        # Verify the @property returns a copy, meaning external mutation is blocked
        env.config["demand_type"] = "black_swan"
        self.assertEqual(env._config["demand_type"], "zero")

    def test_47_malformed_action_aborts_before_mutation(self):
        env, _, _ = self._make_env({"horizon": 10}, seed=0)
        snapshot = self._state_snapshot(env)
        bad_actions = {"retailer": np.array([0.5], dtype=np.float32)} 
        with self.assertRaises(ValueError):
            env.step(bad_actions)
        self.assertEqual(self._state_snapshot(env), snapshot)

    def test_48_order_slips_are_consumed_when_processed(self):
        env, _, _ = self._make_env({"horizon": 10, "demand_type": "zero"}, seed=0)
        self.assertEqual(env.order_pipelines["retailer"].pipeline.get(1), 4)
        env.step(self._actions_for_order(env, 0)) 
        self.assertNotIn(1, env.order_pipelines["retailer"].pipeline)

    def test_49_transit_pipeline_rejects_negative_or_zero_lead_time(self):
        env, _, _ = self._make_env()
        with self.assertRaises(ValueError):
            env.shipment_pipelines["retailer"].add_shipment(1, 10, 0)
        with self.assertRaises(ValueError):
            env.shipment_pipelines["retailer"].add_shipment(1, 10, -1)

    def test_50_transit_pipeline_rejects_negative_quantities(self):
        env, _, _ = self._make_env()
        with self.assertRaises(ValueError):
            env.shipment_pipelines["retailer"].add_shipment(1, -5, 2)

    def test_51_transit_pipeline_ignores_zero_quantity_shipments(self):
        env, _, _ = self._make_env()
        env.shipment_pipelines["retailer"].pipeline.clear()
        env.shipment_pipelines["retailer"].add_shipment(1, 0, 2)
        self.assertEqual(len(env.shipment_pipelines["retailer"].pipeline), 0)

    def test_52_env_initialization_rejects_impossible_configs_types(self):
        with self.assertRaises(ValueError): self._make_env({"max_order": 0})
        with self.assertRaises(ValueError): self._make_env({"max_order": 10.5}) 
        with self.assertRaises(ValueError): self._make_env({"max_order": True}) 
        with self.assertRaises(ValueError): self._make_env({"horizon": 0})
        with self.assertRaises(ValueError): self._make_env({"horizon": 1.5})
        with self.assertRaises(ValueError): self._make_env({"horizon": True})
        with self.assertRaises(ValueError): self._make_env({"lookahead": -1})
        with self.assertRaises(ValueError): self._make_env({"lookahead": 2.5}) 
        with self.assertRaises(ValueError): self._make_env({"lookahead": True})
        with self.assertRaises(ValueError): self._make_env({"holding_cost": -0.5})
        with self.assertRaises(ValueError): self._make_env({"holding_cost": np.nan})
        with self.assertRaises(ValueError): self._make_env({"backorder_cost": np.inf})

    def test_53_manufacturer_reset_strictly_enforces_single_delay_box(self):
        env, _, _ = self._make_env({"demand_type": "step"}, seed=0)
        self.assertEqual(env.order_pipelines["manufacturer"].pipeline, {1: 4})
        self.assertNotIn(2, env.order_pipelines["manufacturer"].pipeline)

    def test_54_observation_is_pure_and_does_not_consume_rng(self):
        env, _, _ = self._make_env({"demand_type": "black_swan", "horizon": 30}, seed=0)
        rng_state_before = env.np_random.bit_generator.state
        
        # This used to roll demand and mutate the RNG. It should now be pure.
        obs = env._build_obs("retailer")
        
        rng_state_after = env.np_random.bit_generator.state
        self.assertEqual(rng_state_before, rng_state_after)

    def test_55_malformed_action_shape_raises_error_before_mutation(self):
        env, _, _ = self._make_env({"horizon": 10}, seed=0)
        snapshot = self._state_snapshot(env)
        
        bad_actions_2d = self._actions_for_order(env, 0)
        bad_actions_2d["retailer"] = np.array([[0.5]], dtype=np.float32)
        with self.assertRaises(ValueError):
            env.step(bad_actions_2d)
            
        bad_actions_extra = self._actions_for_order(env, 0)
        bad_actions_extra["ghost_agent"] = np.array([0.5], dtype=np.float32)
        with self.assertRaises(ValueError):
            env.step(bad_actions_extra)
            
        self.assertEqual(self._state_snapshot(env), snapshot)

    def test_56_nan_and_inf_action_raises_error(self):
        env, _, _ = self._make_env({"horizon": 10}, seed=0)
        bad_actions = self._actions_for_order(env, 0)
        bad_actions["retailer"] = np.array([np.nan], dtype=np.float32)
        with self.assertRaises(ValueError): env.step(bad_actions)
        
        bad_actions["retailer"] = np.array([np.inf], dtype=np.float32)
        with self.assertRaises(ValueError): env.step(bad_actions)

    def test_57_transit_pipeline_rejects_non_integer_or_nan_quantity(self):
        env, _, _ = self._make_env()
        with self.assertRaises(ValueError):
            env.shipment_pipelines["retailer"].add_shipment(1, np.nan, 2)
        with self.assertRaises(ValueError):
            env.shipment_pipelines["retailer"].add_shipment(1, 1.5, 2)

    def test_58_transit_pipeline_rejects_fractional_or_negative_step(self):
        env, _, _ = self._make_env()
        with self.assertRaises(ValueError):
            env.shipment_pipelines["retailer"].add_shipment(-1, 10, 2)
        with self.assertRaises(ValueError):
            env.shipment_pipelines["retailer"].add_shipment(1.5, 10, 2)

    def test_59_env_rejects_invalid_jittery_config(self):
        with self.assertRaises(ValueError):
            self._make_env({"jittery_lead_time": "False"})

    def test_60_env_rejects_invalid_reward_alpha(self):
        with self.assertRaises(ValueError):
            self._make_env({"reward_alpha": 1.5})
            
    def test_61_final_step_does_not_pollute_future_pipelines(self):
        env, _, _ = self._make_env({"horizon": 1}, seed=0)
        env.step(self._actions_for_order(env, 10))
        self.assertEqual(env.order_pipelines["retailer"].pipeline.get(3, 0), 0)

    def test_62_build_obs_dynamically_recomputes_incoming_order(self):
        env, _, _ = self._make_env({"horizon": 10, "lookahead": 2}, seed=0)
        env.order_pipelines["retailer"].add_shipment(0, 99, 1) 
        obs = env._build_obs("wholesaler")
        self.assertEqual(obs[3], 103.0)

    def test_63_string_and_object_actions_raise_error(self):
        env, _, _ = self._make_env({"horizon": 10}, seed=0)
        bad_actions = self._actions_for_order(env, 0)
        
        bad_actions["retailer"] = "0.5" # String
        with self.assertRaises(ValueError): env.step(bad_actions)
            
        bad_actions["retailer"] = [object()] # Object
        with self.assertRaises(ValueError): env.step(bad_actions)

    def test_64_transit_pipeline_rejects_bools_and_numpy_floats(self):
        env, _, _ = self._make_env()
        with self.assertRaises(ValueError):
            env.shipment_pipelines["retailer"].add_shipment(True, 10, 2)
        with self.assertRaises(ValueError):
            env.shipment_pipelines["retailer"].add_shipment(1, True, 2)
        with self.assertRaises(ValueError):
            env.shipment_pipelines["retailer"].add_shipment(1, 10, True)
            
        with self.assertRaises(ValueError):
            env.shipment_pipelines["retailer"].add_shipment(1, np.float32(1.5), 2)
            
        with self.assertRaises(ValueError):
            env.shipment_pipelines["retailer"].receive_shipment(True)

    def test_65_config_is_strictly_immutable_after_init(self):
        env, _, _ = self._make_env({"demand_type": "zero"}, seed=0)
        env.config["demand_type"] = "black_swan"
        self.assertEqual(env._config["demand_type"], "zero")
        
    def test_66_bankers_rounding_avoided(self):
        env, _, _ = self._make_env({"max_order": 2}, seed=0)
        # ENGINEER FIX: float32 precision requires an exact binary fraction
        # raw_action = 0.25 * 2 = 0.5. NumPy Banker's rounding yields 0. Math Floor+0.5 yields 1.
        actions = self._actions_for_order(env, 0)
        actions["retailer"] = np.array([0.25], dtype=np.float32) 
        env.step(actions)
        # Verify the order placed into the pipeline was exactly 1
        self.assertEqual(env.order_pipelines["retailer"].pipeline.get(3, 0), 1)

    def test_67_current_incoming_order_is_diagnostic_only(self):
        env, _, _ = self._make_env({"demand_type": "step", "horizon": 10}, seed=0)
        obs, _, _, _, _ = env.step(self._actions_for_order(env, 4))
        # After step 1 is done, it processed week 1 demand (4.0).
        self.assertEqual(env.current_incoming_order["retailer"], 4.0)
        # But the returned observation is peaking at week 2 (4.0). 
        self.assertEqual(obs["retailer"][3], 4.0)

    def test_68_long_rollout_invariants_fuzzing(self):
        env, _, _ = self._make_env({"horizon": 100, "demand_type": "extreme_chaos", "jittery_lead_time": True}, seed=42)
        for step in range(100):
            actions = {a: np.array([env.action_space(a).sample()[0]], dtype=np.float32) for a in env.agents}
            obs, rewards, _, _, infos = env.step(actions)
            
            for agent in env.possible_agents:
                self.assertGreaterEqual(env.inventory[agent], 0)
                self.assertGreaterEqual(env.backlog[agent], 0)
                self.assertGreaterEqual(env.unfulfilled_orders[agent], 0)
                
                if agent in env.agents:
                    self.assertTrue(np.isfinite(rewards[agent]))
                    self.assertTrue(np.isfinite(infos[agent]["local_cost"]))
                    self.assertTrue(np.all(np.isfinite(obs[agent])))
                    
                for pipe in [env.order_pipelines[agent], env.shipment_pipelines[agent]]:
                    for arr_step, qty in pipe.pipeline.items():
                        self.assertTrue(isinstance(arr_step, (int, np.integer)))
                        self.assertTrue(isinstance(qty, (int, np.integer)))
                        self.assertGreaterEqual(arr_step, env.current_step)
                        self.assertGreater(qty, 0)

if __name__ == "__main__":
    unittest.main(verbosity=2)