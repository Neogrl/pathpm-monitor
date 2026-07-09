import argparse
import csv
import json
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from config import Config
from model import masked_logits
from nodes import NODE_INPUT_FIELDS
from visualize_attention import build_attention_state, load_actor, multihead_attention_weights


def scalar(x) -> float:
    return float(x.detach().cpu().item() if torch.is_tensor(x) else x)


def tensor_stats(name: str, tensor: torch.Tensor, note: str = "") -> Dict:
    data = tensor.detach()
    out = {
        "name": name,
        "shape": list(data.shape),
        "dtype": str(data.dtype).replace("torch.", ""),
        "note": note,
    }
    if data.dtype == torch.bool:
        total = int(data.numel())
        true_count = int(data.sum().item())
        out.update(
            {
                "true_count": true_count,
                "false_count": total - true_count,
                "true_ratio": true_count / max(total, 1),
            }
        )
        return out
    if data.is_floating_point():
        finite = torch.isfinite(data)
        out["nan_count"] = int(torch.isnan(data).sum().item())
        out["inf_count"] = int(torch.isinf(data).sum().item())
        values = data[finite].float()
    else:
        out["nan_count"] = 0
        out["inf_count"] = 0
        values = data.float().reshape(-1)
    if values.numel() > 0:
        out.update(
            {
                "min": scalar(values.min()),
                "max": scalar(values.max()),
                "mean": scalar(values.mean()),
                "std": scalar(values.std(unbiased=False)) if values.numel() > 1 else 0.0,
            }
        )
    return out


def attention_entropy(weights: torch.Tensor) -> float:
    p = weights.clamp(min=1e-12)
    entropy = -(p * p.log()).sum(dim=-1)
    return scalar(entropy.mean())


def write_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_candidate_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_stat_line(item: Dict) -> str:
    base = f"- `{item['name']}`: shape={item['shape']}, dtype={item['dtype']}"
    if "mean" in item:
        base += f", mean={item['mean']:.5f}, std={item['std']:.5f}, min={item['min']:.5f}, max={item['max']:.5f}"
    if "true_count" in item:
        base += f", true={item['true_count']}, false={item['false_count']}, true_ratio={item['true_ratio']:.4f}"
    if item.get("note"):
        base += f"\n  - {item['note']}"
    return base


def top_global_attention_rows(global_batch, current_node_indices, attention_layers: List[torch.Tensor], top_k: int) -> List[Dict]:
    rows = []
    positions = global_batch.node_positions
    for layer_idx, layer_weights in enumerate(attention_layers):
        arr = layer_weights[0].detach().cpu().numpy()
        for uav_id in range(arr.shape[0]):
            order = np.argsort(arr[uav_id])[::-1][:top_k]
            for rank, node_idx in enumerate(order, start=1):
                rows.append(
                    {
                        "kind": "actor_encoder_current_node_to_global",
                        "layer": layer_idx,
                        "uav": uav_id,
                        "rank": rank,
                        "current_node_idx": int(current_node_indices[0, uav_id].item()),
                        "node_idx": int(node_idx),
                        "x": float(positions[node_idx, 0]),
                        "y": float(positions[node_idx, 1]),
                        "attention": float(arr[uav_id, node_idx]),
                    }
                )
    return rows


@torch.no_grad()
def trace_flow(cfg: Config, actor, torch_obs: Dict[str, torch.Tensor], global_batch, batch, top_k: int) -> Dict:
    rows = []
    top_rows = []
    candidate_rows = []

    global_node_inputs = torch_obs["global_node_inputs"]
    global_edge_mask = torch_obs["global_edge_mask"]
    global_node_padding_mask = torch_obs["global_node_padding_mask"]
    current_node_indices = torch_obs["current_node_indices"]
    candidate_node_indices = torch_obs["candidate_node_indices"]
    candidate_padding_mask = torch_obs["candidate_padding_mask"]
    action_mask = torch_obs["action_mask"]
    uav_state = torch_obs["uav_state"]
    prev_option = torch_obs["prev_option"]

    b, n, g, _ = global_node_inputs.shape
    m = candidate_node_indices.shape[-1]
    rows.extend(
        [
            tensor_stats("input.global_node_inputs", global_node_inputs, "每架 UAV 一份全局节点图特征；维度由当前 PRM 节点数和 NODE_INPUT_FIELDS 决定。"),
            tensor_stats("input.global_edge_mask", global_edge_mask, "保留图结构接口；当前不再限制 encoder attention。"),
            tensor_stats("input.global_node_padding_mask", global_node_padding_mask, "只屏蔽不可通行 / padding 的全局节点。"),
            tensor_stats("input.current_node_indices", current_node_indices, "每架 UAV 当前所在的最近全局节点编号。"),
            tensor_stats("input.candidate_node_indices", candidate_node_indices, "每架 UAV 的邻接候选动作节点编号。"),
            tensor_stats("input.candidate_padding_mask", candidate_padding_mask, "候选动作空槽。"),
            tensor_stats("input.action_mask", action_mask, "候选动作非法槽；与 padding/invalid index 共同约束 pointer。"),
            tensor_stats("input.uav_state", uav_state),
            tensor_stats("input.prev_option", prev_option),
        ]
    )

    encoder_mask, global_padding = actor._global_masks(global_edge_mask, global_node_padding_mask, b, n, g)
    rows.append(tensor_stats("actor.encoder_mask", encoder_mask, "dense encoder 只由 global_node_padding_mask 扩展得到；不再包含 global_edge_mask。"))

    flat_nodes = global_node_inputs.reshape(b * n, g, -1)
    actor_src = actor.actor_initial_embedding(flat_nodes)
    rows.append(tensor_stats("actor.initial_embedding(flat_global_nodes)", actor_src))

    actor_attention_layers = []
    actor_attention_entropy = []
    layer_src = actor_src
    flat_current = current_node_indices.reshape(b * n)
    for layer_idx, layer in enumerate(actor.actor_encoder.layers):
        normed = layer.norm1(layer_src)
        weights = multihead_attention_weights(layer.attention, normed, normed, encoder_mask)
        actor_attention_entropy.append(attention_entropy(weights))
        current_weights = weights[:, torch.arange(b * n, device=flat_current.device), flat_current, :]
        actor_attention_layers.append(current_weights.mean(dim=0).reshape(b, n, g).detach().cpu())
        rows.append(tensor_stats(f"actor.encoder.layer{layer_idx}.attention_weights", weights, "shape 是 [heads, B*n_uavs, query_global_nodes, key_global_nodes]。"))
        layer_src = layer(layer_src, encoder_mask)
        rows.append(tensor_stats(f"actor.encoder.layer{layer_idx}.output", layer_src))

    actor_encoded = layer_src.reshape(b, n, g, -1)
    actor_encoded = actor_encoded.masked_fill(global_padding.reshape(b, n, g, 1), 0.0)
    actor_current = actor._gather_current(actor_encoded, current_node_indices)
    actor_candidates = actor._gather_candidates(actor_encoded, candidate_node_indices)
    rows.extend(
        [
            tensor_stats("actor.encoded_global_nodes", actor_encoded),
            tensor_stats("actor.current_node_feature", actor_current),
            tensor_stats("actor.candidate_node_features", actor_candidates, "从 encoded global nodes 中按 candidate_node_indices gather 得到。"),
        ]
    )

    if cfg.disable_options:
        current_option = torch.zeros_like(prev_option.long())
        termination_logits = torch.full_like(prev_option.float(), -20.0)
    else:
        termination_logits = actor.termination_head(actor_current + actor.option_embedding_for_termination(prev_option.long())).squeeze(-1)
        current_option = prev_option.long()
    query = actor_current
    if not cfg.disable_options:
        query = query + actor.option_embedding_for_policy(current_option.long())
    flat_query = query.reshape(b * n, 1, -1)
    flat_candidates = actor_candidates.reshape(b * n, m, -1)
    pointer_mask = (action_mask | candidate_padding_mask | (candidate_node_indices < 0)).reshape(b * n, 1, m).bool()
    rows.extend(
        [
            tensor_stats("actor.termination_logits", termination_logits),
            tensor_stats("actor.policy_query", query),
            tensor_stats("actor.pointer_mask", pointer_mask, "局部候选动作 mask。True 的候选不会被 decoder/pointer 选择。"),
        ]
    )

    decoder_layer = actor.actor_decoder.layers[0]
    decoder_weights = multihead_attention_weights(
        decoder_layer.attention,
        decoder_layer.norm1(flat_query),
        decoder_layer.norm1(flat_candidates),
        pointer_mask,
    )
    enhanced_query = actor.actor_decoder(flat_query, flat_candidates, pointer_mask)
    logp = actor.pointer(enhanced_query, flat_candidates, pointer_mask).squeeze(1)
    waypoint_logits = masked_logits(logp.reshape(b, n, m), action_mask | candidate_padding_mask | (candidate_node_indices < 0))
    waypoint_probs = torch.softmax(waypoint_logits, dim=-1)
    greedy_actions = torch.argmax(waypoint_logits, dim=-1)
    rows.extend(
        [
            tensor_stats("actor.decoder.attention_weights", decoder_weights, "shape 是 [heads, B*n_uavs, 1, max_node_candidates]，只看邻接候选节点。"),
            tensor_stats("actor.decoder.enhanced_query", enhanced_query),
            tensor_stats("actor.pointer_log_probs_before_final_mask", logp),
            tensor_stats("actor.waypoint_logits", waypoint_logits),
            tensor_stats("actor.waypoint_probs", waypoint_probs),
            tensor_stats("actor.greedy_actions", greedy_actions),
        ]
    )

    critic_mask, critic_padding = actor._global_masks(global_edge_mask, global_node_padding_mask, b, n, g)
    critic_src = actor.critic_initial_embedding(flat_nodes)
    rows.append(tensor_stats("critic.initial_embedding(flat_global_nodes)", critic_src))
    critic_attention_entropy = []
    critic_layer_src = critic_src
    for layer_idx, layer in enumerate(actor.critic_encoder.layers):
        normed = layer.norm1(critic_layer_src)
        weights = multihead_attention_weights(layer.attention, normed, normed, critic_mask)
        critic_attention_entropy.append(attention_entropy(weights))
        rows.append(tensor_stats(f"critic.encoder.layer{layer_idx}.attention_weights", weights))
        critic_layer_src = layer(critic_layer_src, critic_mask)
        rows.append(tensor_stats(f"critic.encoder.layer{layer_idx}.output", critic_layer_src))

    critic_encoded = critic_layer_src.reshape(b, n, g, -1)
    critic_encoded = critic_encoded.masked_fill(critic_padding.reshape(b, n, g, 1), 0.0)
    critic_valid = (~critic_padding).float().unsqueeze(-1)
    critic_pooled = (critic_encoded * critic_valid).sum(dim=2) / critic_valid.sum(dim=2).clamp(min=1.0)
    critic_current = actor._gather_current(critic_encoded, current_node_indices)
    critic_uav = actor.critic_uav_encoder(uav_state.float())
    critic_state = actor.critic_state_embedding(torch.cat([critic_current + critic_pooled, critic_uav], dim=-1))
    team_context = critic_state.mean(dim=1)
    values_manual = actor.value_head(team_context).squeeze(-1).unsqueeze(1).expand(-1, n)
    rows.extend(
        [
            tensor_stats("critic.encoded_global_nodes", critic_encoded),
            tensor_stats("critic.pooled_global_context", critic_pooled),
            tensor_stats("critic.current_node_feature", critic_current),
            tensor_stats("critic.team_context", team_context),
            tensor_stats("critic.values", values_manual),
        ]
    )

    termination_forward, logits_forward, values_forward = actor.forward(**torch_obs)
    rows.extend(
        [
            tensor_stats("forward.output.termination_logits", termination_forward),
            tensor_stats("forward.output.waypoint_logits", logits_forward),
            tensor_stats("forward.output.values", values_forward),
        ]
    )

    top_rows.extend(top_global_attention_rows(global_batch, current_node_indices, actor_attention_layers, top_k))
    decoder_mean = decoder_weights.mean(dim=0).reshape(b, n, m).detach().cpu().numpy()
    probs = waypoint_probs.detach().cpu().numpy()[0]
    logits_np = waypoint_logits.detach().cpu().numpy()[0]
    for uav_id in range(n):
        for slot in range(m):
            node_idx = int(batch.candidate_node_indices[uav_id, slot])
            point = batch.waypoints[uav_id, slot]
            valid_slot = bool((not batch.node_padding_mask[uav_id, slot]) and (not batch.action_mask[uav_id, slot]) and node_idx >= 0)
            candidate_rows.append(
                {
                    "uav": uav_id,
                    "slot": slot,
                    "node_idx": node_idx,
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "valid": valid_slot,
                    "is_padding": bool(batch.node_padding_mask[uav_id, slot]),
                    "is_action_masked": bool(batch.action_mask[uav_id, slot]),
                    "decoder_attention": float(decoder_mean[0, uav_id, slot]),
                    "logit": float(logits_np[uav_id, slot]),
                    "prob": float(probs[uav_id, slot]),
                    "is_greedy_action": bool(int(greedy_actions[0, uav_id].item()) == slot),
                }
            )

    checks = {
        "manual_vs_forward_max_abs_logits": scalar((waypoint_logits - logits_forward).abs().max()),
        "manual_vs_forward_max_abs_values": scalar((values_manual - values_forward).abs().max()),
        "manual_vs_forward_max_abs_termination": scalar((termination_logits - termination_forward).abs().max()),
        "actor_attention_entropy_by_layer": actor_attention_entropy,
        "critic_attention_entropy_by_layer": critic_attention_entropy,
        "decoder_attention_entropy": attention_entropy(decoder_weights),
        "valid_candidate_counts_per_uav": [
            int((~batch.node_padding_mask[i] & ~batch.action_mask[i] & (batch.candidate_node_indices[i] >= 0)).sum())
            for i in range(n)
        ],
        "greedy_actions": greedy_actions.detach().cpu().numpy().tolist(),
        "greedy_node_indices": [
            int(batch.candidate_node_indices[i, int(greedy_actions[0, i].item())])
            for i in range(n)
        ],
    }

    return {
        "stats": rows,
        "checks": checks,
        "top_global_attention": top_rows,
        "candidate_outputs": candidate_rows,
        "node_input_fields": NODE_INPUT_FIELDS,
    }


def write_markdown(path: Path, cfg: Config, args, trace: Dict) -> None:
    lines = []
    lines.append("# 网络输出流诊断报告")
    lines.append("")
    lines.append("本脚本不训练，只构造一帧 observation，并按当前模型真实 forward 路径记录从输入到输出的张量流。")
    lines.append("")
    lines.append("## 运行配置")
    lines.append("")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- warmup_steps: `{args.warmup_steps}`")
    lines.append(f"- checkpoint: `{args.checkpoint}`")
    lines.append(f"- device: `{args.device}`")
    lines.append(f"- n_uavs: `{cfg.n_uavs}`")
    lines.append(f"- global nodes: `{int((cfg.map_size / cfg.graph_node_spacing + 1) ** 2)}`")
    lines.append(f"- max_node_candidates: `{cfg.max_node_candidates}`")
    lines.append("")
    lines.append("## 关键检查")
    lines.append("")
    for key, value in trace["checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## 张量流")
    lines.append("")
    for item in trace["stats"]:
        lines.append(format_stat_line(item))
    lines.append("")
    lines.append("## Actor Encoder 当前节点 Top Attention")
    lines.append("")
    for row in trace["top_global_attention"][: min(len(trace["top_global_attention"]), 80)]:
        lines.append(
            f"- layer {row['layer']} UAV{row['uav']} rank {row['rank']}: "
            f"node={row['node_idx']} pos=({row['x']:.1f},{row['y']:.1f}) attention={row['attention']:.6f}"
        )
    lines.append("")
    lines.append("## 候选动作输出")
    lines.append("")
    shown = 0
    for row in trace["candidate_outputs"]:
        if not row["valid"]:
            continue
        lines.append(
            f"- UAV{row['uav']} slot {row['slot']} node={row['node_idx']} "
            f"pos=({row['x']:.1f},{row['y']:.1f}) decoder_attn={row['decoder_attention']:.4f} "
            f"logit={row['logit']:.4f} prob={row['prob']:.4f} greedy={row['is_greedy_action']}"
        )
        shown += 1
        if shown >= 80:
            break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace OptionActor network flow from observation inputs to action/value outputs.")
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--n-targets", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="diagnostic_runs/network_flow_trace")
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    actor = load_actor(cfg, device, args.checkpoint)
    _, _, _, _, _, batch, global_batch, _, torch_obs = build_attention_state(
        cfg,
        actor,
        device,
        args.seed,
        args.n_targets,
        args.warmup_steps,
    )
    trace = trace_flow(cfg, actor, torch_obs, global_batch, batch, args.top_k)

    write_json(out / "network_flow_trace.json", trace)
    write_markdown(out / "network_flow_trace.md", cfg, args, trace)
    write_candidate_csv(out / "candidate_outputs.csv", trace["candidate_outputs"])
    write_candidate_csv(out / "top_global_attention.csv", trace["top_global_attention"])
    print(
        {
            "report": str(out / "network_flow_trace.md"),
            "json": str(out / "network_flow_trace.json"),
            "candidate_csv": str(out / "candidate_outputs.csv"),
            "top_attention_csv": str(out / "top_global_attention.csv"),
            "checks": trace["checks"],
        }
    )


if __name__ == "__main__":
    main()
