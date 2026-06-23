import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


def masked_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(action_mask.bool(), -1e9)


class KNNMessagePassing(nn.Module):
    def __init__(self, embed_dim: int, k_neighbors: int):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.message = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, emb: torch.Tensor, coords: torch.Tensor, invalid_mask: torch.Tensor) -> torch.Tensor:
        b, m, e = emb.shape
        k = min(max(self.k_neighbors, 1), m)
        dist = torch.cdist(coords, coords)
        eye = torch.eye(m, dtype=torch.bool, device=emb.device).unsqueeze(0)
        invalid_neighbors = invalid_mask.bool().unsqueeze(1).expand(-1, m, -1)
        dist = dist.masked_fill(eye | invalid_neighbors, 1e6)
        nn_idx = torch.topk(dist, k=k, dim=-1, largest=False).indices
        gather_idx = nn_idx.unsqueeze(-1).expand(-1, -1, -1, e)
        neigh = torch.gather(emb.unsqueeze(1).expand(-1, m, -1, -1), 2, gather_idx)
        neigh_valid = torch.gather((~invalid_mask).float().unsqueeze(1).expand(-1, m, -1), 2, nn_idx)
        denom = neigh_valid.sum(dim=-1, keepdim=True).clamp(min=1.0)
        msg = (neigh * neigh_valid.unsqueeze(-1)).sum(dim=2) / denom
        out = self.norm(emb + self.message(torch.cat([emb, msg], dim=-1)))
        return out.masked_fill(invalid_mask.bool().unsqueeze(-1), 0.0)


class OptionActor(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        e = cfg.embed_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(16, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, e),
            nn.ReLU(),
        )
        self.graph_encoder = KNNMessagePassing(e, cfg.k_neighbors)
        enc_layer = nn.TransformerEncoderLayer(d_model=e, nhead=4, dim_feedforward=cfg.hidden_dim * 2, batch_first=True)
        self.candidate_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.option_embedding = nn.Embedding(2, e)
        self.uav_encoder = nn.Sequential(nn.Linear(4, e), nn.ReLU(), nn.Linear(e, e), nn.ReLU())
        self.team_encoder = nn.Sequential(nn.Linear(12, e), nn.ReLU(), nn.Linear(e, e), nn.ReLU())
        self.context = nn.Sequential(nn.Linear(e * 3, e), nn.ReLU(), nn.Linear(e, e))
        self.termination_head = nn.Sequential(nn.Linear(e, 64), nn.ReLU(), nn.Linear(64, 1))
        self.query = nn.Linear(e, e)

    def forward(
        self,
        node_inputs: torch.Tensor,
        node_padding_mask: torch.Tensor,
        action_mask: torch.Tensor,
        uav_state: torch.Tensor,
        prev_option: torch.Tensor,
        team_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, m, _ = node_inputs.shape
        flat_nodes = node_inputs.reshape(b * n, m, -1)
        flat_padding = node_padding_mask.reshape(b * n, m).bool()
        emb = self.node_encoder(flat_nodes)
        emb = self.graph_encoder(emb, flat_nodes[:, :, :2], flat_padding)
        encoded = self.candidate_encoder(emb, src_key_padding_mask=flat_padding)
        encoded = encoded.reshape(b, n, m, -1)
        uav_emb = self.uav_encoder(uav_state)
        opt_emb = self.option_embedding(prev_option.long())
        team_emb = self.team_encoder(team_summary).unsqueeze(1).expand(-1, n, -1)
        ctx = self.context(torch.cat([uav_emb, opt_emb, team_emb], dim=-1))
        termination_logits = self.termination_head(ctx).squeeze(-1)
        q = self.query(ctx).unsqueeze(2)
        waypoint_logits = torch.sum(q * encoded, dim=-1) / (self.cfg.embed_dim ** 0.5)
        waypoint_logits = masked_logits(waypoint_logits, action_mask)
        return termination_logits, waypoint_logits

    @torch.no_grad()
    def act(
        self,
        node_inputs: torch.Tensor,
        node_padding_mask: torch.Tensor,
        action_mask: torch.Tensor,
        uav_state: torch.Tensor,
        prev_option: torch.Tensor,
        team_summary: torch.Tensor,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        termination_logits, logits = self.forward(node_inputs, node_padding_mask, action_mask, uav_state, prev_option, team_summary)
        beta = torch.sigmoid(termination_logits)
        if greedy:
            terminate = beta > 0.5
        else:
            terminate = torch.bernoulli(beta).bool()
        option = torch.where(terminate, 1 - prev_option.long(), prev_option.long())
        probs = F.softmax(logits, dim=-1)
        if greedy:
            action = torch.argmax(probs, dim=-1)
        else:
            action = torch.distributions.Categorical(probs=probs).sample()
        return action, option, terminate


class CentralizedCritic(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        e = cfg.embed_dim
        self.node_encoder = nn.Sequential(nn.Linear(16, e), nn.ReLU(), nn.Linear(e, e), nn.ReLU())
        self.graph_encoder = KNNMessagePassing(e, cfg.k_neighbors)
        self.option_embedding = nn.Embedding(2, e)
        self.global_encoder = nn.Sequential(
            nn.Linear(5 + 4 + cfg.max_true_targets * 5 + cfg.max_true_targets * 4, e),
            nn.ReLU(),
            nn.Linear(e, e),
            nn.ReLU(),
        )
        self.uav_encoder = nn.Sequential(nn.Linear(4, e), nn.ReLU(), nn.Linear(e, e), nn.ReLU())
        self.q_head = nn.Sequential(nn.Linear(e * 4, e), nn.ReLU(), nn.Linear(e, 1))

    def forward(
        self,
        node_inputs: torch.Tensor,
        action_mask: torch.Tensor,
        uav_state: torch.Tensor,
        current_options: torch.Tensor,
        global_phd: torch.Tensor,
        global_search: torch.Tensor,
        true_target_states: torch.Tensor,
        discovered_memory: torch.Tensor,
    ) -> torch.Tensor:
        b, n, m, _ = node_inputs.shape
        node_emb = self.node_encoder(node_inputs)
        flat_nodes = node_inputs.reshape(b * n, m, -1)
        flat_emb = node_emb.reshape(b * n, m, -1)
        flat_invalid = action_mask.reshape(b * n, m).bool()
        node_emb = self.graph_encoder(flat_emb, flat_nodes[:, :, :2], flat_invalid).reshape(b, n, m, -1)
        uav_emb = self.uav_encoder(uav_state).unsqueeze(2).expand(-1, -1, m, -1)
        opt_emb = self.option_embedding(current_options.long()).unsqueeze(2).expand(-1, -1, m, -1)
        global_raw = torch.cat(
            [
                global_phd,
                global_search,
                true_target_states.reshape(b, -1),
                discovered_memory.reshape(b, -1),
            ],
            dim=-1,
        )
        glob = self.global_encoder(global_raw).view(b, 1, 1, -1).expand(-1, n, m, -1)
        q = self.q_head(torch.cat([node_emb, uav_emb, opt_emb, glob], dim=-1)).squeeze(-1)
        return q.masked_fill(action_mask.bool(), -1e9)


class TwinCritic(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.q1 = CentralizedCritic(cfg)
        self.q2 = CentralizedCritic(cfg)

    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(*args, **kwargs), self.q2(*args, **kwargs)
