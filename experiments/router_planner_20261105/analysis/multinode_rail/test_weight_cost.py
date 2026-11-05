"""Small independently calculable cases for replica-transfer accounting."""
import unittest
import numpy as np

from simulate import Config
from weight_cost import WeightConfig, evaluate_weights, compose_forward_cost


class WeightCostTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[[0]], [[0]], [[0]], [[1]]], np.int16)
        self.owner = np.array([[[2]], [[3]], [[1]], [[3]]], np.int16)
        self.home = np.array([0, 3], np.int16)
        self.config = Config(hidden=2, rdma_gbytes_per_second=1,
                             nvlink_gbytes_per_second=2, rdma_startup_us=0,
                             nvlink_startup_us=0)
        self.weights = WeightConfig(intermediate=2, matrices=3, bytes_per_element=2,
                                    gradient_bytes_per_element=4, chunk_bytes=10)

    def evaluate(self, **kwargs):
        return evaluate_weights(self.x, self.owner, self.home, 2, 2,
                                self.config, self.weights, **kwargs)

    def test_remote_node_dedup_and_local_rank_fanout(self):
        result = self.evaluate()
        forward = result["forward"]["traffic"]
        # Expert 0 is 24 bytes: once to node 1; NVLink to rank 1 and rank 3.
        self.assertEqual(result["weight_bytes_per_expert"], 24)
        self.assertEqual(result["active_extra_copies"], 3)
        self.assertEqual(forward["rdma_payload_bytes"], 24)
        self.assertEqual(forward["nvlink_payload_bytes"], 48)
        self.assertEqual(forward["nic_tx_bytes_node_rail"], [[24, 0], [0, 0]])
        self.assertEqual(forward["nic_rx_bytes_node_rail"], [[0, 0], [24, 0]])
        self.assertEqual(result["forward"]["schedule"]["rdma_chunks"], 3)
        self.assertEqual(result["forward"]["schedule"]["nvlink_chunks"], 6)

    def test_gradient_reverse_and_fp32(self):
        result = self.evaluate()
        grad = result["gradient"]["traffic"]
        self.assertEqual(grad["rdma_payload_bytes"], 48)
        self.assertEqual(grad["nvlink_payload_bytes"], 96)
        self.assertEqual(grad["nic_tx_bytes_node_rail"], [[0, 0], [48, 0]])
        self.assertEqual(grad["nvlink_bytes_node_src_local_dst_local"],
                         [[[0, 0], [48, 0]], [[0, 0], [48, 0]]])

    def test_warm_weights_do_not_remove_gradient(self):
        warm = np.array([[True, False], [False, False], [True, False], [True, True]])
        result = self.evaluate(initial_resident=warm)
        self.assertEqual(result["missing_active_copies"], 1)
        self.assertEqual(result["forward"]["traffic"]["rdma_payload_bytes"], 0)
        self.assertEqual(result["forward"]["traffic"]["nvlink_payload_bytes"], 24)
        self.assertEqual(result["gradient"]["traffic"]["rdma_payload_bytes"], 48)
        self.assertEqual(result["gradient"]["traffic"]["nvlink_payload_bytes"], 96)

    def test_home_only_and_disabled_gradient(self):
        owner = self.home[self.x]
        result = evaluate_weights(self.x, owner, self.home, 2, 2, self.config, self.weights)
        self.assertEqual(result["forward"]["schedule"]["makespan_us"], 0)
        self.assertEqual(result["gradient"]["schedule"]["makespan_us"], 0)
        weights = WeightConfig(gradient_bytes_per_element=0)
        result = evaluate_weights(self.x, self.owner, self.home, 2, 2, self.config, weights)
        self.assertEqual(result["gradient"]["traffic"]["rdma_payload_bytes"], 0)
        self.assertEqual(result["gradient"]["schedule"]["makespan_us"], 0)

    def test_invalid_warm_home_and_topology(self):
        with self.assertRaises(ValueError):
            self.evaluate(initial_resident=np.zeros((4, 2), bool))
        with self.assertRaises(ValueError):
            self.evaluate(initial_resident=np.ones((4, 2), int))
        with self.assertRaises(ValueError):
            evaluate_weights(self.x, self.owner, self.home, 4, 2)

    def test_schedule_counter_match_and_lower_bound_random(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            x = rng.integers(0, 8, (8, 3, 2))
            owner = rng.integers(0, 8, x.shape)
            result = evaluate_weights(x, owner, np.arange(8), 2, 4,
                                      self.config, self.weights)
            for phase in ("forward", "gradient"):
                p = result[phase]
                self.assertGreaterEqual(p["schedule"]["makespan_us"] + 1e-12,
                                        p["traffic"]["chunk_service_port_lower_bound_us"])
                self.assertGreaterEqual(p["traffic"]["chunk_service_port_lower_bound_us"] + 1e-12,
                                        p["traffic"]["payload_port_lower_bound_us"])

    def test_overlap_sums_same_port_service(self):
        weights = self.evaluate()
        token = {"latency_us": {"planner": 3, "dispatch": 5, "compute": 7, "combine": 11},
                 "dispatch": {"ports": {"rdma_tx/0/0": {"busy_us": 5}}}}
        result = compose_forward_cost(token, weights)
        self.assertAlmostEqual(result["relaxed_weight_dispatch_joint_port_bound_us"], 5.024)
        self.assertAlmostEqual(result["serialized_forward_weight_gradient_us"],
                               3 + weights["forward"]["schedule"]["makespan_us"] + 5 + 7 + 11 +
                               weights["gradient"]["schedule"]["makespan_us"])


if __name__ == "__main__":
    unittest.main()
