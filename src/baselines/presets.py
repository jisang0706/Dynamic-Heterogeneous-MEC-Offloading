from __future__ import annotations

from dataclasses import dataclass, replace

from src.config import ExperimentConfig


@dataclass(frozen=True, slots=True)
class ExperimentVariant:
    variant_id: str
    name: str
    category: str
    runner_kind: str
    description: str


VARIANT_REGISTRY: dict[str, ExperimentVariant] = {
    "B0": ExperimentVariant("B0", "Li et al. Exact", "baseline", "external", "Original static Li et al. code path."),
    "B1": ExperimentVariant("B1", "Li-Arch Dynamic", "baseline", "ppo", "Dynamic env with Li-style individual actors and MLP critic."),
    "B2": ExperimentVariant("B2", "Shared MLP", "baseline", "ppo", "Shared actor without role and fair MLP critic."),
    "B3": ExperimentVariant("B3", "P-GCN Only", "baseline", "ppo", "Shared actor without role and P-GCN star critic."),
    "B4": ExperimentVariant("B4", "Role Only", "baseline", "ppo", "Role-conditioned actor with fair MLP critic."),
    "B5": ExperimentVariant("B5", "Deterministic Context", "baseline", "det_context", "Deterministic context encoder with P-GCN critic."),
    "B6": ExperimentVariant("B6", "Set Critic", "baseline", "ppo", "Shared actor without role and DeepSets critic."),
    "B7": ExperimentVariant("B7", "IPPO", "baseline", "ippo", "Independent PPO with local critics and no centralized info."),
    "B8": ExperimentVariant("B8", "MADDPG", "baseline", "unsupported", "Off-policy continuous-control baseline."),
    "LOCAL_ONLY": ExperimentVariant("LOCAL_ONLY", "Local Only", "fixed", "fixed", "All computation remains local."),
    "EDGE_ONLY": ExperimentVariant("EDGE_ONLY", "Edge Computing", "fixed", "fixed", "All tasks fully offloaded with max power."),
    "RANDOM": ExperimentVariant("RANDOM", "Random", "fixed", "fixed", "Uniform random offloading and power ratios."),
    "QAG": ExperimentVariant("QAG", "Queue-Aware Greedy", "fixed", "fixed", "Queue-aware heuristic using current queues, channel, CPU, and deadlines."),
    "A1": ExperimentVariant("A1", "Proposed", "ablation", "ppo", "P-GCN star critic + role + L_I."),
    "A2": ExperimentVariant("A2", "No L_I", "ablation", "ppo", "P-GCN + role without identifiability loss."),
    "A3": ExperimentVariant("A3", "Static Full Model", "ablation", "ppo", "Full model on static environment."),
    "A4": ExperimentVariant("A4", "Star+Proximity", "ablation", "ppo", "Full model with star plus proximity edges."),
    "A5_ROLE2": ExperimentVariant("A5_ROLE2", "Role Dim 2", "ablation", "ppo", "Full model with role_dim=2."),
    "A5_ROLE5": ExperimentVariant("A5_ROLE5", "Role Dim 5", "ablation", "ppo", "Full model with role_dim=5."),
    "A6A": ExperimentVariant("A6A", "Mobility Only", "ablation", "ppo", "Full model with mobility only."),
    "A6B": ExperimentVariant("A6B", "CPU Variation Only", "ablation", "ppo", "Full model with CPU dynamics only."),
    "A7_100": ExperimentVariant("A7_100", "Threshold 100", "ablation", "ppo", "A4 with distance threshold 100m."),
    "A7_200": ExperimentVariant("A7_200", "Threshold 200", "ablation", "ppo", "A4 with distance threshold 200m."),
    "A8": ExperimentVariant("A8", "Simple L_D", "ablation", "ppo", "Full model plus simple diversity loss."),
}


def list_experiment_variants() -> list[ExperimentVariant]:
    return list(VARIANT_REGISTRY.values())


def get_experiment_variant(variant_id: str | None) -> ExperimentVariant | None:
    if variant_id is None:
        return None
    normalized = variant_id.upper()
    return VARIANT_REGISTRY.get(normalized)


def apply_experiment_variant(config: ExperimentConfig, variant_id: str | None) -> tuple[ExperimentConfig, ExperimentVariant | None]:
    variant = get_experiment_variant(variant_id)
    if variant is None:
        return config, None

    environment = config.environment
    model = config.model
    training = replace(config.training, variant_id=variant.variant_id)

    if variant.variant_id == "B0":
        environment = replace(environment, use_mobility=False, use_cpu_dynamics=False)
        model = replace(model, critic_type="mlp", actor_type="individual", actor_hidden_dim=200, use_role=False, use_l_i=False)
    elif variant.variant_id == "B1":
        model = replace(model, critic_type="mlp", actor_type="individual", actor_hidden_dim=200, use_role=False, use_l_i=False)
    elif variant.variant_id == "B2":
        model = replace(model, critic_type="mlp", actor_type="shared", actor_hidden_dim=128, use_role=False, use_l_i=False)
    elif variant.variant_id == "B3":
        model = replace(model, critic_type="pgcn", actor_type="shared", actor_hidden_dim=128, use_role=False, use_l_i=False)
    elif variant.variant_id == "B4":
        model = replace(model, critic_type="mlp", actor_type="shared", actor_hidden_dim=128, use_role=True, use_l_i=True)
    elif variant.variant_id == "B5":
        model = replace(model, critic_type="pgcn", actor_type="shared", actor_hidden_dim=128, use_role=True, use_l_i=False)
    elif variant.variant_id == "B6":
        model = replace(model, critic_type="set", actor_type="shared", actor_hidden_dim=128, use_role=False, use_l_i=False)
    elif variant.variant_id == "B7":
        model = replace(model, critic_type="mlp", actor_type="individual", actor_hidden_dim=200, use_role=False, use_l_i=False)
    elif variant.variant_id == "B8":
        model = replace(model, use_role=False, use_l_i=False)
    elif variant.variant_id == "A1":
        model = replace(model, critic_type="pgcn", actor_type="individual", use_role=True, use_l_i=True, use_l_d_simple=False)
        environment = replace(environment, graph_type="star", use_mobility=True, use_cpu_dynamics=True)
    elif variant.variant_id == "A2":
        model = replace(model, critic_type="pgcn", actor_type="individual", use_role=True, use_l_i=False, use_l_d_simple=False)
    elif variant.variant_id == "A3":
        model = replace(model, critic_type="pgcn", actor_type="individual", use_role=True, use_l_i=True, use_l_d_simple=False)
        environment = replace(environment, use_mobility=False, use_cpu_dynamics=False, graph_type="star")
    elif variant.variant_id == "A4":
        model = replace(model, critic_type="pgcn", actor_type="individual", use_role=True, use_l_i=True, use_l_d_simple=False)
        environment = replace(environment, graph_type="star_proximity")
    elif variant.variant_id == "A5_ROLE2":
        model = replace(model, critic_type="pgcn", actor_type="individual", use_role=True, use_l_i=True, role_dim=2, use_l_d_simple=False)
    elif variant.variant_id == "A5_ROLE5":
        model = replace(model, critic_type="pgcn", actor_type="individual", use_role=True, use_l_i=True, role_dim=5, use_l_d_simple=False)
    elif variant.variant_id == "A6A":
        model = replace(model, critic_type="pgcn", actor_type="individual", use_role=True, use_l_i=True, use_l_d_simple=False)
        environment = replace(environment, use_mobility=True, use_cpu_dynamics=False)
    elif variant.variant_id == "A6B":
        model = replace(model, critic_type="pgcn", actor_type="individual", use_role=True, use_l_i=True, use_l_d_simple=False)
        environment = replace(environment, use_mobility=False, use_cpu_dynamics=True)
    elif variant.variant_id == "A7_100":
        model = replace(model, critic_type="pgcn", actor_type="individual", use_role=True, use_l_i=True, use_l_d_simple=False)
        environment = replace(environment, graph_type="star_proximity", distance_threshold_m=100.0)
    elif variant.variant_id == "A7_200":
        model = replace(model, critic_type="pgcn", actor_type="individual", use_role=True, use_l_i=True, use_l_d_simple=False)
        environment = replace(environment, graph_type="star_proximity", distance_threshold_m=200.0)
    elif variant.variant_id == "A8":
        model = replace(model, critic_type="pgcn", actor_type="individual", use_role=True, use_l_i=True, use_l_d_simple=True)

    return replace(config, environment=environment, model=model, training=training), variant
