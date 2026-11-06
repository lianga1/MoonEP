"""Host-side planner settings and model-dimension reuse; no CUDA imports."""
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from numbers import Integral


def _integer(name, value, minimum, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        suffix = f"..{maximum}" if maximum is not None else " or greater"
        raise ValueError(f"{name} must be {minimum}{suffix}")
    return value


@dataclass(frozen=True)
class SampledPlannerConfig:
    """Planner-only controls; E/K/H come from the Buffer/model, never here.

    sample_size is the requested global sample count, capped at R*S when the
    batch is smaller. Settings are fixed for the lifetime of a Buffer. Create
    and warm a new Buffer when changing them, before CUDA Graph capture.
    candidate_limit=0 retains unlimited package selection. A positive value
    keeps the best initial packages and updates only their dependent caches;
    the effective limit is clipped to the current K/G package capacity.
    """
    sample_size: int = 256
    groups_per_class: int = 2
    rounds: int = 4
    sample_seed: int = 20261026
    candidate_limit: int = 0

    def __post_init__(self):
        bounds = {"sample_size": (1, 4096), "groups_per_class": (1, 8),
                  "rounds": (1, None), "sample_seed": (0, None),
                  "candidate_limit": (0, 2**31-1)}
        for name, (lo, hi) in bounds.items():
            object.__setattr__(self, name, _integer(name, getattr(self, name), lo, hi))

    def to_dict(self):
        return asdict(self)

    def effective_sample_size(self, total_tokens):
        return min(self.sample_size, _integer("total_tokens", total_tokens, 1))

    def validate_shape(self, *, R, E, K, S, B):
        for name, value in (("R", R), ("E", E), ("K", K), ("S", S), ("B", B)):
            _integer(name, value, 1)
        if R != 8 or not 8 <= E <= 4096 or E % R or not 1 <= K <= min(32, E):
            raise ValueError("sampled_group requires R=8, 8<=E<=4096, E%R=0, 1<=K<=min(32,E)")
        if not 1 <= B <= E-E//R:
            raise ValueError("sampled_group requires 1<=B<=E-E/R")


def resolve_planner_config(config=None, *, rounds=None, sample_seed=None):
    """Accept a dataclass or JSON-style mapping, retaining the old flat options.

    Explicit settings may not conflict. Omitted legacy arguments do not
    override a supplied configuration object's defaults or chosen values.
    """
    supplied = config is not None
    if config is None:
        config = SampledPlannerConfig()
    elif isinstance(config, Mapping):
        unknown = set(config)-{f.name for f in fields(SampledPlannerConfig)}
        if unknown:
            raise ValueError(f"unknown sampled planner settings: {sorted(unknown)}")
        config = SampledPlannerConfig(**dict(config))
    elif not isinstance(config, SampledPlannerConfig):
        raise TypeError("planner_config must be SampledPlannerConfig, a mapping, or None")
    values = config.to_dict()
    for name, value in (("rounds", rounds), ("sample_seed", sample_seed)):
        if value is not None:
            value = _integer(name, value, 1 if name == "rounds" else 0)
            if supplied and value != values[name]:
                raise ValueError(f"planning_{name} conflicts with planner_config.{name}")
            values[name] = value
    return SampledPlannerConfig(**values)


def _read(obj, path):
    for name in path.split('.'):
        if isinstance(obj, Mapping):
            obj = obj.get(name)
        else:
            obj = getattr(obj, name, None)
        if obj is None:
            return None
    return obj


@dataclass(frozen=True)
class MoEModelConfig:
    """Canonical global routed-expert dimensions, not per-rank shard sizes."""
    hidden_size: int
    num_experts: int
    top_k: int

    def __post_init__(self):
        for name in ("hidden_size", "num_experts", "top_k"):
            object.__setattr__(self, name, _integer(name, getattr(self, name), 1))
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed global num_experts")

    @classmethod
    def from_model_config(cls, config, *, expert_field=None, topk_field=None, hidden_field=None):
        """Read a config object or mapping, including an optional text_config.

        Ambiguous fields with different values are rejected. Explicit field
        paths override auto-detection for that dimension. num_local_experts
        and generation top_k are deliberately NOT auto-detected: their meaning
        varies; pass expert_field/topk_field explicitly when appropriate.
        """
        if isinstance(config, cls) and all(v is None for v in (expert_field, topk_field, hidden_field)):
            return config
        def resolve(label, aliases, explicit):
            if explicit is not None:
                if not isinstance(explicit, str) or not explicit:
                    raise ValueError(f"{label} field path must be a non-empty string")
                value = _read(config, explicit)
                if value is None:
                    raise ValueError(f"model config is missing {explicit}")
                return _integer(explicit, value, 1)
            matches = []
            for prefix in ("", "text_config."):
                for alias in aliases:
                    path = prefix+alias
                    value = _read(config, path)
                    if value is not None:
                        matches.append((path, _integer(path, value, 1)))
            if not matches:
                raise ValueError(f"cannot infer {label}; provide its explicit field path")
            if len({v for _, v in matches}) != 1:
                raise ValueError(f"conflicting model fields for {label}: {matches}")
            return matches[0][1]
        return cls(
            hidden_size=resolve("hidden_size", ("hidden_size", "d_model", "n_embd"), hidden_field),
            num_experts=resolve("global routed experts", ("num_experts", "n_routed_experts", "moe_num_experts"), expert_field),
            top_k=resolve("experts per token", ("num_experts_per_tok", "num_experts_per_token", "moe_top_k"), topk_field),
        )
