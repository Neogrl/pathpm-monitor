import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config
from trainer import Trainer
from worker import RolloutWorker


def representation_metrics(features: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    relative_variations = []
    cosine_similarities = []
    for batch_id in range(features.shape[0]):
        for uav_id in range(features.shape[1]):
            selected = features[batch_id, uav_id][valid[batch_id, uav_id]]
            if selected.shape[0] < 2:
                continue
            centered = selected - selected.mean(dim=0, keepdim=True)
            relative_variations.append(
                centered.norm(dim=-1).mean() / selected.norm(dim=-1).mean().clamp(min=1e-8)
            )
            normalized = torch.nn.functional.normalize(selected, dim=-1)
            cosine = normalized @ normalized.transpose(0, 1)
            off_diagonal = ~torch.eye(selected.shape[0], dtype=torch.bool, device=selected.device)
            cosine_similarities.append(cosine[off_diagonal].mean())
    return {
        "relative_variation": float(torch.stack(relative_variations).mean().item()),
        "mean_cosine_similarity": float(torch.stack(cosine_similarities).mean().item()),
    }


@torch.no_grad()
def encoder_stage_metrics(trainer: Trainer, obs: dict[str, torch.Tensor], use_pe: bool) -> dict:
    actor = trainer.actor
    global_inputs = obs["global_node_inputs"]
    b, n, g, _ = global_inputs.shape
    encoder_mask, padding = actor._global_masks(
        obs["global_edge_mask"], obs["global_node_padding_mask"], b, n, g
    )
    flat_nodes = global_inputs.reshape(b * n, g, -1)
    node_embedding = actor.actor_initial_embedding(flat_nodes)
    position_embedding = actor.actor_spatio_pos_embedding(
        obs["spatio_pos_encoding"].reshape(b * n, g, -1)
    )
    current = node_embedding + position_embedding if use_pe else node_embedding
    valid_global = ~padding
    stages = {
        "node_feature_embedding": representation_metrics(
            node_embedding.reshape(b, n, g, -1), valid_global
        )
    }
    if use_pe:
        stages["position_embedding"] = representation_metrics(
            position_embedding.reshape(b, n, g, -1), valid_global
        )
        stages["fused_embedding"] = representation_metrics(
            current.reshape(b, n, g, -1), valid_global
        )
    for layer_id, layer in enumerate(actor.actor_encoder.layers, start=1):
        current = layer(current, encoder_mask)
        stages[f"attention_layer_{layer_id}"] = representation_metrics(
            current.reshape(b, n, g, -1), valid_global
        )
    return stages


def build_observation(worker: RolloutWorker, seed: int) -> tuple[dict, np.ndarray]:
    env, target, search, tracks, prev_option = worker.reset_stack(seed)
    candidates = worker.node_builder.build(
        env.uav_positions,
        target,
        search,
        tracks,
        step=env.step_count,
    )
    graph = worker.node_builder.global_batch_from_candidates(
        env.uav_positions,
        target,
        search,
        tracks,
        candidates.candidate_node_indices,
        candidates.node_padding_mask,
        candidates.action_mask,
        step=env.step_count,
    )
    obs = worker._obs_dict_from_arrays(
        candidates.node_inputs,
        candidates.node_padding_mask,
        candidates.action_mask,
        env,
        target,
        search,
        prev_option,
        global_batch=graph,
    )
    return worker._to_torch(obs, batch_dim=True), graph.spatio_pos_encoding.copy()


@torch.no_grad()
def compare_outputs(trainer: Trainer, obs: dict[str, torch.Tensor]) -> dict[str, float]:
    actor = trainer.actor
    actor.eval()
    encoded_with, _, _ = actor._encode_actor_global(
        obs["global_node_inputs"],
        obs["spatio_pos_encoding"],
        obs["global_edge_mask"],
        obs["global_node_padding_mask"],
    )
    zero_pe = torch.zeros_like(obs["spatio_pos_encoding"])
    encoded_without, _, _ = actor._encode_actor_global(
        obs["global_node_inputs"],
        zero_pe,
        obs["global_edge_mask"],
        obs["global_node_padding_mask"],
    )

    candidates_with = actor._gather_candidates(encoded_with, obs["candidate_node_indices"])
    candidates_without = actor._gather_candidates(encoded_without, obs["candidate_node_indices"])
    candidate_valid = ~(
        obs["action_mask"] | obs["candidate_padding_mask"] | (obs["candidate_node_indices"] < 0)
    )

    def relative_variation(features: torch.Tensor) -> float:
        ratios = []
        for batch_id in range(features.shape[0]):
            for uav_id in range(features.shape[1]):
                selected = features[batch_id, uav_id][candidate_valid[batch_id, uav_id]]
                if selected.shape[0] < 2:
                    continue
                centered = selected - selected.mean(dim=0, keepdim=True)
                ratios.append(centered.norm(dim=-1).mean() / selected.norm(dim=-1).mean().clamp(min=1e-8))
        return float(torch.stack(ratios).mean().item()) if ratios else 0.0

    _, logits_with, values_with = actor(**obs)
    obs_without = dict(obs)
    obs_without["spatio_pos_encoding"] = zero_pe
    _, logits_without, values_without = actor(**obs_without)
    valid = ~(obs["action_mask"] | obs["candidate_padding_mask"] | (obs["candidate_node_indices"] < 0))
    logit_delta = (logits_with - logits_without)[valid]

    return {
        "encoder_mean_abs_delta": float((encoded_with - encoded_without).abs().mean().item()),
        "encoder_max_abs_delta": float((encoded_with - encoded_without).abs().max().item()),
        "valid_logit_mean_abs_delta": float(logit_delta.abs().mean().item()),
        "valid_logit_max_abs_delta": float(logit_delta.abs().max().item()),
        "value_mean_abs_delta": float((values_with - values_without).abs().mean().item()),
        "candidate_relative_variation_with_pe": relative_variation(candidates_with),
        "candidate_relative_variation_without_pe": relative_variation(candidates_without),
        "greedy_action_changes": int(
            (torch.argmax(logits_with, dim=-1) != torch.argmax(logits_without, dim=-1)).sum().item()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = Config()
    cfg.device = args.device
    device = torch.device(args.device)
    trainer = Trainer(cfg, device=device)
    worker = RolloutWorker(cfg, actor=trainer.actor, device=device)

    obs, pe_first = build_observation(worker, args.seed)
    _, pe_repeat = build_observation(worker, args.seed)
    _, pe_other = build_observation(worker, args.seed + 1)
    first_uav = pe_first[0]
    pivots = np.argmax(np.abs(first_uav), axis=0)
    pivot_values = first_uav[pivots, np.arange(first_uav.shape[1])]

    result = {
        "seed": args.seed,
        "shape": list(pe_first.shape),
        "finite": bool(np.isfinite(pe_first).all()),
        "nonzero_fraction": float(np.mean(np.abs(pe_first) > 1e-8)),
        "same_seed_max_abs_delta": float(np.max(np.abs(pe_first - pe_repeat))),
        "different_seed_mean_abs_delta": float(np.mean(np.abs(pe_first - pe_other))),
        "all_canonical_pivots_nonnegative": bool(np.all(pivot_values >= 0.0)),
        "shared_across_uavs": bool(np.allclose(pe_first, pe_first[0:1])),
        "global_encoder_stages_with_pe": encoder_stage_metrics(trainer, obs, use_pe=True),
        "global_encoder_stages_without_pe": encoder_stage_metrics(trainer, obs, use_pe=False),
        **compare_outputs(trainer, obs),
    }
    print(json.dumps(result, indent=2))

    assert result["finite"]
    assert result["nonzero_fraction"] > 0.0
    assert result["same_seed_max_abs_delta"] == 0.0
    assert result["different_seed_mean_abs_delta"] > 0.0
    assert result["all_canonical_pivots_nonnegative"]
    assert result["shared_across_uavs"]
    assert result["encoder_mean_abs_delta"] > 0.0
    assert result["valid_logit_mean_abs_delta"] > 0.0
    assert result["value_mean_abs_delta"] > 0.0


if __name__ == "__main__":
    main()
