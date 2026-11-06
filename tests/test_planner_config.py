"""Host configuration and model reuse tests; no distributed/CUDA allocation."""
from types import SimpleNamespace
import pytest
from moonep import Buffer, MoEModelConfig, SampledPlannerConfig
from moonep.planner_config import resolve_planner_config
import moonep.api as api


def test_defaults_and_small_batch_cap():
    cfg = SampledPlannerConfig()
    assert cfg.to_dict() == dict(sample_size=256, groups_per_class=2, rounds=4, sample_seed=20261026, candidate_limit=0)
    assert cfg.effective_sample_size(8) == 8
    assert cfg.effective_sample_size(4096) == 256
    assert resolve_planner_config({'sample_size': 37}).sample_size == 37
    assert resolve_planner_config(rounds=2, sample_seed=7).rounds == 2
    assert resolve_planner_config(cfg, rounds=4) == cfg


@pytest.mark.parametrize('values', [dict(sample_size=0), dict(sample_size=4097),
    dict(sample_size=True), dict(groups_per_class=0), dict(groups_per_class=9),
    dict(rounds=0), dict(sample_seed=-1), dict(rounds=1.5),
    dict(candidate_limit=-1), dict(candidate_limit=True), dict(candidate_limit=1.5),
    dict(candidate_limit=2**31)])
def test_invalid_settings(values):
    with pytest.raises(ValueError):
        SampledPlannerConfig(**values)


def test_unknown_and_conflicting_settings():
    with pytest.raises(ValueError, match='unknown'):
        resolve_planner_config({'E': 352})
    with pytest.raises(ValueError, match='conflicts'):
        resolve_planner_config({'rounds': 3}, rounds=4)
    with pytest.raises(TypeError):
        resolve_planner_config('sampled')


def test_model_aliases_and_custom_fields():
    expected = MoEModelConfig(7168, 512, 16)
    assert MoEModelConfig.from_model_config(SimpleNamespace(
        hidden_size=7168, n_routed_experts=512, num_experts_per_tok=16)) == expected
    assert MoEModelConfig.from_model_config({'text_config': {
        'd_model': 7168, 'moe_num_experts': 512, 'moe_top_k': 16}}) == expected
    assert MoEModelConfig.from_model_config(expected) is expected
    assert MoEModelConfig.from_model_config({'model': {'width': 7168, 'count': 512, 'k': 16}},
        hidden_field='model.width', expert_field='model.count', topk_field='model.k') == expected
    # Generation top_k must never silently become the router's topK.
    assert MoEModelConfig.from_model_config({'hidden_size': 7168, 'num_experts': 512,
        'num_experts_per_tok': 16, 'top_k': 50}) == expected
    with pytest.raises(ValueError, match='cannot infer'):
        MoEModelConfig.from_model_config({'hidden_size': 7168, 'num_experts': 512, 'top_k': 16})
    with pytest.raises(ValueError, match='conflicting'):
        MoEModelConfig.from_model_config({'hidden_size': 7168, 'num_experts': 512,
            'n_routed_experts': 256, 'num_experts_per_tok': 16})
    with pytest.raises(ValueError):
        MoEModelConfig.from_model_config({'hidden_size': 7168, 'num_experts': [256, 512],
            'num_experts_per_tok': 16})


def test_model_to_buffer_has_one_source_of_dimensions(monkeypatch):
    captured = {}
    class Probe(Buffer):
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def __del__(self):
            pass
    cfg = SampledPlannerConfig(sample_size=512, groups_per_class=3)
    model = {'hidden_size': 256, 'num_experts': 1024, 'num_experts_per_tok': 32}
    Probe.from_model_config(model, S=33, num_ep_ranks=8,
        planning_backend='sampled_group', planner_config=cfg)
    assert (captured['H'], captured['E'], captured['K']) == (256, 1024, 32)
    assert captured['planner_config'] is cfg
    with pytest.raises(ValueError, match='overridden'):
        Probe.from_model_config(model, S=33, num_ep_ranks=8, E=352)


def test_new_shapes_reach_allocation_and_bad_shapes_do_not(monkeypatch):
    class AllocationReached(Exception):
        pass
    def allocation(*args, **kwargs):
        raise AllocationReached
    monkeypatch.setattr(api, '_create_context', allocation)
    monkeypatch.setattr(Buffer, '__del__', lambda self: None)
    for E, K, S, B in ((64, 1, 1, 8), (128, 3, 33, 16), (520, 7, 65, 65),
                       (512, 16, 65, 64), (1024, 32, 33, 128), (4096, 1, 1, 512)):
        with pytest.raises(AllocationReached):
            Buffer(S, 256, K, E, 8, B=B, planning_backend='sampled_group')
    for E, K, S, B in ((520, 33, 65, 65), (513, 8, 65, 64), (4097, 8, 65, 8),
                       (64, 1, 0, 8), (64, 1, 33, 57)):
        with pytest.raises(ValueError):
            Buffer(S, 256, K, E, 8, B=B, planning_backend='sampled_group')


def test_candidate_limit_delivery_config_and_model_reuse():
    cfg = resolve_planner_config({'candidate_limit': 128, 'rounds': 4})
    assert cfg.candidate_limit == 128 and cfg.rounds == 4
    assert resolve_planner_config(cfg) == cfg
    assert SampledPlannerConfig(candidate_limit=2**31-1).candidate_limit == 2**31-1
    # The same requested cap can be reused for smaller model configurations;
    # the selector clips it to its K/G-specific capacity after shape resolution.
    cfg.validate_shape(R=8, E=64, K=1, S=17, B=8)
    cfg.validate_shape(R=8, E=352, K=8, S=257, B=8)
