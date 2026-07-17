import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from model import OptionActor
from nodes import NODE_INPUT_FIELDS, NODE_INPUT_INDEX, NodeBuilder
from worker import RolloutWorker
from utils import write_json


UAV_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728", "#17becf", "#ff7f0e"]


def write_table(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def setup_axis(ax, cfg: Config, title: str) -> None:
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, cfg.map_size)
    ax.set_ylim(0, cfg.map_size)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#dddddd", linewidth=0.45)


def load_actor(cfg: Config, device: torch.device, checkpoint: Optional[str]) -> OptionActor:
    actor = OptionActor(cfg).to(device)
    actor.eval()
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        state = ckpt["actor"] if isinstance(ckpt, dict) and "actor" in ckpt else ckpt
        actor.load_state_dict(state)
    return actor


def build_attention_state(cfg: Config, actor: OptionActor, device: torch.device, seed: int, n_targets: Optional[int], warmup_steps: int):
    worker = RolloutWorker(cfg, actor=actor, device=device)
    env, target, search, tracks, prev_option = worker.reset_stack(seed, n_targets=n_targets, eval_mode=True)
    rng = np.random.default_rng(seed + 303)
    builder = worker.node_builder
    for _ in range(max(warmup_steps, 0)):
        batch = builder.build(env.uav_positions, target, search, tracks, step=env.step_count)
        valid = ~batch.action_mask & ~batch.node_padding_mask
        actions = np.zeros(cfg.n_uavs, dtype=np.int64)
        for i in range(cfg.n_uavs):
            slots = np.flatnonzero(valid[i])
            actions[i] = int(rng.choice(slots)) if len(slots) else 0
        waypoints = batch.waypoints[np.arange(cfg.n_uavs), actions]
        info = env.step(waypoints)
        target.predict(info.step_duration)
        target.update(info.measurements.points, env.uav_positions)
        tracks.update(env.step_count, info.measurements.points, target.peaks())
        search.update(env.uav_positions, info.measurements.points)
    batch = builder.build(env.uav_positions, target, search, tracks, step=env.step_count)
    global_batch = builder.global_batch_from_candidates(
        env.uav_positions,
        target,
        search,
        tracks,
        batch.candidate_node_indices,
        batch.node_padding_mask,
        batch.action_mask,
        step=env.step_count,
    )
    obs = worker._obs_dict_from_arrays(
        batch.node_inputs,
        batch.node_padding_mask,
        batch.action_mask,
        env,
        target,
        search,
        prev_option,
        global_batch=global_batch,
    )
    torch_obs = worker._to_torch(obs, batch_dim=True)
    return worker, env, target, search, tracks, batch, global_batch, obs, torch_obs


def multihead_attention_weights(attention, query: torch.Tensor, key: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if key is None:
        key = query
    batch, n_key, dim = key.shape
    n_query = query.shape[1]
    q = torch.matmul(query.contiguous().view(-1, dim), attention.w_query).view(attention.n_heads, batch, n_query, attention.key_dim)
    k = torch.matmul(key.contiguous().view(-1, dim), attention.w_key).view(attention.n_heads, batch, n_key, attention.key_dim)
    logits = attention.norm_factor * torch.matmul(q, k.transpose(2, 3))
    if mask is not None:
        logits = logits.masked_fill(mask.view(1, batch, n_query, n_key).expand_as(logits).bool(), -1e9)
    weights = torch.softmax(logits, dim=-1)
    if mask is not None:
        weights = weights.masked_fill(mask.view(1, batch, n_query, n_key).expand_as(weights).bool(), 0.0)
    return weights


@torch.no_grad()
def compute_attention(actor: OptionActor, torch_obs: dict[str, torch.Tensor]) -> dict:
    global_node_inputs = torch_obs["global_node_inputs"]
    global_edge_mask = torch_obs["global_edge_mask"]
    global_node_padding_mask = torch_obs["global_node_padding_mask"]
    current_node_indices = torch_obs["current_node_indices"]
    candidate_node_indices = torch_obs["candidate_node_indices"]
    candidate_padding_mask = torch_obs["candidate_padding_mask"]
    action_mask = torch_obs["action_mask"]
    prev_option = torch_obs["prev_option"]

    b, n, g, _ = global_node_inputs.shape
    encoder_mask, global_padding = actor._global_masks(global_edge_mask, global_node_padding_mask, b, n, g)
    flat_nodes = global_node_inputs.reshape(b * n, g, -1)
    src = actor.actor_initial_embedding(flat_nodes)

    encoder_current_attention = []
    layer_src = src
    flat_current = current_node_indices.reshape(b * n)
    for layer in actor.actor_encoder.layers:
        normed = layer.norm1(layer_src)
        weights = multihead_attention_weights(layer.attention, normed, normed, encoder_mask)
        current_weights = weights[:, torch.arange(b * n, device=flat_current.device), flat_current, :]
        encoder_current_attention.append(current_weights.mean(dim=0).reshape(b, n, g).detach().cpu())
        layer_src = layer(layer_src, encoder_mask)
    encoded = layer_src.reshape(b, n, g, -1)
    encoded = encoded.masked_fill(global_padding.reshape(b, n, g, 1), 0.0)
    current_feature = actor._gather_current(encoded, current_node_indices)
    candidate_feature = actor._gather_candidates(encoded, candidate_node_indices)
    current_option = torch.zeros_like(prev_option.long()) if actor.cfg.disable_options else prev_option.long()
    query = current_feature
    if not actor.cfg.disable_options:
        query = query + actor.option_embedding_for_policy(current_option)
    flat_query = query.reshape(b * n, 1, -1)
    flat_candidates = candidate_feature.reshape(b * n, candidate_node_indices.shape[-1], -1)
    decoder_mask = (action_mask | candidate_padding_mask | (candidate_node_indices < 0)).reshape(b * n, 1, -1).bool()

    decoder_layer = actor.actor_decoder.layers[0]
    decoder_weights = multihead_attention_weights(
        decoder_layer.attention,
        decoder_layer.norm1(flat_query),
        decoder_layer.norm1(flat_candidates),
        decoder_mask,
    )
    decoder_attention = decoder_weights.mean(dim=0).reshape(b, n, candidate_node_indices.shape[-1]).detach().cpu()

    _, logits, values = actor.forward(**torch_obs)
    pointer_probs = torch.softmax(logits, dim=-1).detach().cpu()
    greedy_actions = torch.argmax(logits, dim=-1).detach().cpu()

    return {
        "encoder_current_attention": encoder_current_attention,
        "decoder_attention": decoder_attention,
        "pointer_probs": pointer_probs,
        "greedy_actions": greedy_actions,
        "values": values.detach().cpu(),
    }


def draw_entities(ax, cfg: Config, env) -> None:
    for i, pos in enumerate(env.uav_positions):
        color = UAV_COLORS[i % len(UAV_COLORS)]
        ax.scatter(pos[0], pos[1], marker="^", color=color, edgecolor="black", s=46, linewidth=0.25)
        ax.text(pos[0] + 0.8, pos[1] + 0.8, f"U{i}", color=color, fontsize=7)
    ax.scatter(env.target_states[:, 0], env.target_states[:, 1], marker="x", color="#555555", s=28)


def draw_global_self_attention(path: Path, cfg: Config, env, global_batch, attention_layers: List[torch.Tensor], uav_id: int) -> None:
    positions = global_batch.node_positions
    n_layers = len(attention_layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(5.2 * n_layers, 5.0), dpi=145)
    if n_layers == 1:
        axes = [axes]
    for layer_idx, ax in enumerate(axes):
        weights = attention_layers[layer_idx][0, uav_id].numpy()
        setup_axis(ax, cfg, f"Actor self-attention | UAV {uav_id} layer {layer_idx}")
        sc = ax.scatter(positions[:, 0], positions[:, 1], c=weights, cmap="viridis", s=28, vmin=0, vmax=max(float(weights.max()), 1e-9))
        cur_idx = int(global_batch.current_node_indices[uav_id])
        ax.scatter(positions[cur_idx, 0], positions[cur_idx, 1], marker="*", color="white", edgecolor="black", s=130)
        draw_entities(ax, cfg, env)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03, label="mean head attention")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_decoder_attention(path: Path, cfg: Config, env, batch, decoder_attention: torch.Tensor) -> None:
    fig, axes = plt.subplots(1, cfg.n_uavs, figsize=(4.2 * cfg.n_uavs, 4.2), dpi=140)
    if cfg.n_uavs == 1:
        axes = [axes]
    for uav_id, ax in enumerate(axes):
        weights = decoder_attention[0, uav_id].numpy()
        valid = ~batch.node_padding_mask[uav_id] & ~batch.action_mask[uav_id] & (batch.candidate_node_indices[uav_id] >= 0)
        slots = np.flatnonzero(valid)
        setup_axis(ax, cfg, f"Decoder attention over candidates U{uav_id}")
        draw_entities(ax, cfg, env)
        if len(slots):
            points = batch.waypoints[uav_id, slots]
            values = weights[slots]
            sc = ax.scatter(points[:, 0], points[:, 1], c=values, cmap="plasma", s=85, vmin=0, vmax=max(float(values.max()), 1e-9), edgecolor="black", linewidth=0.25)
            for slot, point in zip(slots, points):
                ax.plot([env.uav_positions[uav_id, 0], point[0]], [env.uav_positions[uav_id, 1], point[1]], color=UAV_COLORS[uav_id % len(UAV_COLORS)], alpha=0.18, linewidth=1.1)
                ax.text(point[0] + 0.35, point[1] + 0.35, f"{weights[slot]:.2f}", fontsize=6, color="#111111")
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_pointer_attention(path: Path, cfg: Config, env, batch, pointer_probs: torch.Tensor, greedy_actions: torch.Tensor) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    setup_axis(ax, cfg, "Pointer/action attention over candidates")
    draw_entities(ax, cfg, env)
    probs = pointer_probs[0].numpy()
    actions = greedy_actions[0].numpy()
    for uav_id in range(cfg.n_uavs):
        valid = ~batch.node_padding_mask[uav_id] & ~batch.action_mask[uav_id]
        slots = np.flatnonzero(valid)
        color = UAV_COLORS[uav_id % len(UAV_COLORS)]
        for slot in slots:
            point = batch.waypoints[uav_id, slot]
            prob = float(probs[uav_id, slot])
            ax.plot([env.uav_positions[uav_id, 0], point[0]], [env.uav_positions[uav_id, 1], point[1]], color=color, alpha=0.13 + 0.55 * prob, linewidth=0.8 + 4.0 * prob)
            ax.scatter(point[0], point[1], s=30 + 260 * prob, color=color, alpha=0.55, edgecolor="black", linewidth=0.25)
            ax.text(point[0] + 0.35, point[1] + 0.35, f"{prob:.2f}", fontsize=6, color="#111111")
        chosen = batch.waypoints[uav_id, actions[uav_id]]
        ax.scatter(chosen[0], chosen[1], marker="*", s=150, color=color, edgecolor="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_csv_rows(cfg: Config, global_batch, batch, attention: Dict) -> Tuple[List[Dict], List[Dict]]:
    pointer_rows = []
    probs = attention["pointer_probs"][0].numpy()
    actions = attention["greedy_actions"][0].numpy()
    for uav_id in range(cfg.n_uavs):
        for slot in range(cfg.max_node_candidates):
            node_idx = int(batch.candidate_node_indices[uav_id, slot])
            point = batch.waypoints[uav_id, slot]
            row = {
                "uav": uav_id,
                "slot": slot,
                "node_idx": node_idx,
                "x": float(point[0]),
                "y": float(point[1]),
                "prob": float(probs[uav_id, slot]),
                "is_padding": bool(batch.node_padding_mask[uav_id, slot]),
                "is_action_masked": bool(batch.action_mask[uav_id, slot]),
                "is_greedy_action": bool(actions[uav_id] == slot),
            }
            for name, idx in NODE_INPUT_INDEX.items():
                row[name] = float(batch.node_inputs[uav_id, slot, idx])
            pointer_rows.append(row)

    top_rows = []
    positions = global_batch.node_positions
    for layer_idx, layer_attention in enumerate(attention["encoder_current_attention"]):
        arr = layer_attention[0].numpy()
        for uav_id in range(cfg.n_uavs):
            order = np.argsort(arr[uav_id])[::-1][:10]
            for rank, node_idx in enumerate(order, start=1):
                top_rows.append(
                    {
                        "kind": "actor_self_attention_current_node",
                        "layer": layer_idx,
                        "uav": uav_id,
                        "rank": rank,
                        "node_idx": int(node_idx),
                        "x": float(positions[node_idx, 0]),
                        "y": float(positions[node_idx, 1]),
                        "attention": float(arr[uav_id, node_idx]),
                    }
                )
    decoder = attention["decoder_attention"][0].numpy()
    for uav_id in range(cfg.n_uavs):
        valid = ~batch.node_padding_mask[uav_id] & ~batch.action_mask[uav_id] & (batch.candidate_node_indices[uav_id] >= 0)
        valid_slots = np.flatnonzero(valid)
        order = valid_slots[np.argsort(decoder[uav_id, valid_slots])[::-1][:10]]
        for rank, slot in enumerate(order, start=1):
            node_idx = int(batch.candidate_node_indices[uav_id, slot])
            point = batch.waypoints[uav_id, slot]
            top_rows.append(
                {
                    "kind": "actor_decoder_query_to_candidate",
                    "layer": 0,
                    "uav": uav_id,
                    "rank": rank,
                    "slot": int(slot),
                    "node_idx": node_idx,
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "is_padding": bool(batch.node_padding_mask[uav_id, slot]),
                    "is_action_masked": bool(batch.action_mask[uav_id, slot]),
                    "attention": float(decoder[uav_id, slot]),
                }
            )
    return pointer_rows, top_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize current attention maps without training.")
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--n-targets", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--uav-id", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default="diagnostic_runs/attention_visualization")
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    actor = load_actor(cfg, device, args.checkpoint)
    _, env, target, search, tracks, batch, global_batch, obs, torch_obs = build_attention_state(
        cfg,
        actor,
        device,
        args.seed,
        args.n_targets,
        args.warmup_steps,
    )
    attention = compute_attention(actor, torch_obs)
    uav_id = int(np.clip(args.uav_id, 0, cfg.n_uavs - 1))

    draw_global_self_attention(out / f"actor_self_attention_uav{uav_id}.png", cfg, env, global_batch, attention["encoder_current_attention"], uav_id)
    draw_decoder_attention(out / "actor_decoder_attention_all_uavs.png", cfg, env, batch, attention["decoder_attention"])
    draw_pointer_attention(out / "pointer_candidate_attention.png", cfg, env, batch, attention["pointer_probs"], attention["greedy_actions"])

    pointer_rows, top_rows = build_csv_rows(cfg, global_batch, batch, attention)
    write_table(out / "pointer_candidate_attention.csv", pointer_rows)
    write_table(out / "attention_top_nodes.csv", top_rows)

    summary = {
        "seed": args.seed,
        "warmup_steps": args.warmup_steps,
        "n_uavs": cfg.n_uavs,
        "n_global_nodes": int(global_batch.global_node_inputs.shape[1]),
        "node_input_fields": NODE_INPUT_FIELDS,
        "checkpoint": args.checkpoint,
        "note": "No training is run. If checkpoint is null, attention comes from the current randomly initialized model.",
        "outputs": {
            "actor_self_attention": str(out / f"actor_self_attention_uav{uav_id}.png"),
            "actor_decoder_attention": str(out / "actor_decoder_attention_all_uavs.png"),
            "pointer_candidate_attention": str(out / "pointer_candidate_attention.png"),
            "pointer_csv": str(out / "pointer_candidate_attention.csv"),
            "top_nodes_csv": str(out / "attention_top_nodes.csv"),
        },
        "value_mean": float(attention["values"].mean()),
        "pointer_entropy_mean": float(torch.distributions.Categorical(probs=attention["pointer_probs"]).entropy().mean()),
    }
    write_json(out / "attention_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
