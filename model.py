import math
from typing import Optional

import torch
import torch.nn as nn

from config import Config


def masked_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(action_mask.bool(), -1e9)


class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.norm_factor = 1.0 / math.sqrt(embedding_dim)
        self.tanh_clipping = 10.0
        self.w_query = nn.Parameter(torch.empty(embedding_dim, embedding_dim))
        self.w_key = nn.Parameter(torch.empty(embedding_dim, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for param in self.parameters():
            bound = 1.0 / math.sqrt(param.size(-1))
            nn.init.uniform_(param, -bound, bound)

    def forward(self, query: torch.Tensor, key: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, n_key, dim = key.shape
        n_query = query.shape[1]
        q = torch.matmul(query.reshape(-1, dim), self.w_query).view(batch, n_query, dim)
        k = torch.matmul(key.reshape(-1, dim), self.w_key).view(batch, n_key, dim)
        logits = self.norm_factor * torch.matmul(q, k.transpose(1, 2))
        logits = self.tanh_clipping * torch.tanh(logits)
        if mask is not None:
            logits = logits.masked_fill(mask.view(batch, n_query, n_key).bool(), -1e9)
        return torch.log_softmax(logits, dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim // n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1.0 / math.sqrt(self.key_dim)
        self.w_query = nn.Parameter(torch.empty(n_heads, embedding_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.empty(n_heads, embedding_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.empty(n_heads, embedding_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.empty(n_heads, self.value_dim, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for param in self.parameters():
            bound = 1.0 / math.sqrt(param.size(-1))
            nn.init.uniform_(param, -bound, bound)

    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if key is None:
            key = query
        value = key
        batch, n_key, dim = key.shape
        n_query = query.shape[1]
        q = torch.matmul(query.contiguous().view(-1, dim), self.w_query).view(self.n_heads, batch, n_query, self.key_dim)
        k = torch.matmul(key.contiguous().view(-1, dim), self.w_key).view(self.n_heads, batch, n_key, self.key_dim)
        v = torch.matmul(value.contiguous().view(-1, dim), self.w_value).view(self.n_heads, batch, n_key, self.value_dim)
        logits = self.norm_factor * torch.matmul(q, k.transpose(2, 3))
        if mask is not None:
            logits = logits.masked_fill(mask.view(1, batch, n_query, n_key).expand_as(logits).bool(), -1e9)
        attention = torch.softmax(logits, dim=-1)
        if mask is not None:
            attention = attention.masked_fill(mask.view(1, batch, n_query, n_key).expand_as(attention).bool(), 0.0)
        heads = torch.matmul(attention, v)
        return torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            self.w_out.reshape(-1, self.embedding_dim),
        ).view(batch, n_query, self.embedding_dim)


class Normalization(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.reshape(-1, x.size(-1))).view(*x.size())


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim, n_heads)
        self.norm1 = Normalization(embedding_dim)
        self.ff = nn.Sequential(nn.Linear(embedding_dim, 512), nn.ReLU(inplace=True), nn.Linear(512, embedding_dim))
        self.norm2 = Normalization(embedding_dim)

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = src + self.attention(self.norm1(src), mask=mask)
        return h + self.ff(self.norm2(h))


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim, n_heads)
        self.norm1 = Normalization(embedding_dim)
        self.ff = nn.Sequential(nn.Linear(embedding_dim, 512), nn.ReLU(inplace=True), nn.Linear(512, embedding_dim))
        self.norm2 = Normalization(embedding_dim)

    def forward(self, target: torch.Tensor, memory: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = target + self.attention(self.norm1(target), self.norm1(memory), mask=mask)
        return h + self.ff(self.norm2(h))


class Encoder(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int = 8, n_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(embedding_dim, n_heads) for _ in range(n_layers))

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src, mask)
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int = 8, n_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer(embedding_dim, n_heads) for _ in range(n_layers))

    def forward(self, target: torch.Tensor, memory: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            target = layer(target, memory, mask)
        return target


class OptionActor(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        e = cfg.embed_dim
        self.actor_initial_embedding = nn.Linear(cfg.node_input_dim, e)
        self.actor_spatio_pos_embedding = nn.Linear(cfg.graph_laplacian_pe_dim, e, bias=False)
        self.actor_encoder = Encoder(embedding_dim=e, n_heads=8, n_layers=cfg.graph_encoder_layers)
        self.actor_decoder = Decoder(embedding_dim=e, n_heads=8, n_layers=1)
        self.pointer = SingleHeadAttention(e)
        self.actor_agent_embedding = nn.Embedding(cfg.n_uavs, e)
        nn.init.normal_(self.actor_agent_embedding.weight, mean=0.0, std=0.02)
        self.termination_head = nn.Sequential(nn.Linear(e, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        self.option_embedding_for_termination = nn.Embedding(2, e)
        self.option_embedding_for_policy = nn.Embedding(2, e)

        self.critic_initial_embedding = nn.Linear(cfg.node_input_dim, e)
        self.critic_spatio_pos_embedding = nn.Linear(cfg.graph_laplacian_pe_dim, e, bias=False)
        self.critic_encoder = Encoder(embedding_dim=e, n_heads=8, n_layers=cfg.graph_encoder_layers)
        self.critic_uav_encoder = nn.Sequential(nn.Linear(cfg.uav_state_dim, e), nn.ReLU(inplace=True), nn.Linear(e, e))
        self.critic_state_embedding = nn.Linear(e * 2, e)
        self.value_head = nn.Sequential(nn.Linear(e, e), nn.ReLU(inplace=True), nn.Linear(e, 1))

    def _global_masks(
        self,
        global_edge_mask: torch.Tensor,
        global_node_padding_mask: torch.Tensor,
        b: int,
        n: int,
        g: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if global_edge_mask.dim() not in (2, 3, 4):
            raise ValueError(f"global_edge_mask must have 2, 3, or 4 dims, got {global_edge_mask.shape}")
        if global_node_padding_mask.dim() == 1:
            padding = global_node_padding_mask.view(1, 1, g).expand(b, n, g)
        elif global_node_padding_mask.dim() == 2:
            padding = global_node_padding_mask.view(b, 1, g).expand(b, n, g)
        elif global_node_padding_mask.dim() == 3:
            padding = global_node_padding_mask
        else:
            raise ValueError(f"global_node_padding_mask must have 1, 2, or 3 dims, got {global_node_padding_mask.shape}")
        # CAtNIPP-style global encoder: dense self-attention over all valid global
        # nodes. The graph edge mask is kept in the observation interface for
        # candidate/action construction, but it no longer limits encoder attention.
        encoder_mask = padding.unsqueeze(2).expand(b, n, g, g).bool()
        return encoder_mask.reshape(b * n, g, g), padding.bool()

    def _encode_global_with(
        self,
        global_node_inputs: torch.Tensor,
        spatio_pos_encoding: torch.Tensor,
        global_edge_mask: torch.Tensor,
        global_node_padding_mask: torch.Tensor,
        initial_embedding: nn.Linear,
        positional_embedding: nn.Linear,
        encoder: Encoder,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, g, _ = global_node_inputs.shape
        encoder_mask, padding = self._global_masks(global_edge_mask, global_node_padding_mask, b, n, g)
        flat_nodes = global_node_inputs.reshape(b * n, g, -1)
        embedded = initial_embedding(flat_nodes)
        if self.cfg.graph_laplacian_pe_enabled:
            expected = (b, n, g, self.cfg.graph_laplacian_pe_dim)
            if tuple(spatio_pos_encoding.shape) != expected:
                raise ValueError(
                    f"spatio_pos_encoding must have shape {expected}, got {tuple(spatio_pos_encoding.shape)}"
                )
            flat_position = spatio_pos_encoding.reshape(b * n, g, -1).float()
            embedded = embedded + positional_embedding(flat_position)
        encoded = encoder(embedded, encoder_mask)
        encoded = encoded.masked_fill(padding.reshape(b * n, g, 1), 0.0)
        valid = (~padding).float().unsqueeze(-1)
        pooled = (encoded.reshape(b, n, g, -1) * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1.0)
        return encoded.reshape(b, n, g, -1), padding, pooled

    def _encode_actor_global(
        self,
        global_node_inputs: torch.Tensor,
        spatio_pos_encoding: torch.Tensor,
        global_edge_mask: torch.Tensor,
        global_node_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._encode_global_with(
            global_node_inputs,
            spatio_pos_encoding,
            global_edge_mask,
            global_node_padding_mask,
            self.actor_initial_embedding,
            self.actor_spatio_pos_embedding,
            self.actor_encoder,
        )

    def _encode_critic_global(
        self,
        global_node_inputs: torch.Tensor,
        spatio_pos_encoding: torch.Tensor,
        global_edge_mask: torch.Tensor,
        global_node_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._encode_global_with(
            global_node_inputs,
            spatio_pos_encoding,
            global_edge_mask,
            global_node_padding_mask,
            self.critic_initial_embedding,
            self.critic_spatio_pos_embedding,
            self.critic_encoder,
        )

    @staticmethod
    def _gather_current(encoded: torch.Tensor, current_node_indices: torch.Tensor) -> torch.Tensor:
        b, n, _, e = encoded.shape
        idx = current_node_indices.long().clamp(min=0).view(b, n, 1, 1).expand(-1, -1, 1, e)
        return torch.gather(encoded, dim=2, index=idx).squeeze(2)

    @staticmethod
    def _gather_candidates(encoded: torch.Tensor, candidate_node_indices: torch.Tensor) -> torch.Tensor:
        b, n, _, e = encoded.shape
        m = candidate_node_indices.shape[-1]
        idx = candidate_node_indices.long().clamp(min=0).view(b, n, m, 1).expand(-1, -1, -1, e)
        return torch.gather(encoded, dim=2, index=idx)

    def _critic_value(
        self,
        global_node_inputs: torch.Tensor,
        spatio_pos_encoding: torch.Tensor,
        global_edge_mask: torch.Tensor,
        global_node_padding_mask: torch.Tensor,
        current_node_indices: torch.Tensor,
        uav_state: torch.Tensor,
    ) -> torch.Tensor:
        critic_encoded, _, critic_pooled = self._encode_critic_global(
            global_node_inputs,
            spatio_pos_encoding,
            global_edge_mask,
            global_node_padding_mask,
        )
        critic_current = self._gather_current(critic_encoded, current_node_indices)
        critic_uav = self.critic_uav_encoder(uav_state.float())
        critic_state = self.critic_state_embedding(torch.cat([critic_current + critic_pooled, critic_uav], dim=-1))
        team_context = critic_state.mean(dim=1)
        return self.value_head(team_context).squeeze(-1).unsqueeze(1).expand(-1, uav_state.shape[1])

    def termination_logits(
        self,
        global_node_inputs: torch.Tensor,
        spatio_pos_encoding: torch.Tensor,
        global_edge_mask: torch.Tensor,
        global_node_padding_mask: torch.Tensor,
        current_node_indices: torch.Tensor,
        candidate_node_indices: torch.Tensor,
        candidate_padding_mask: torch.Tensor,
        action_mask: torch.Tensor,
        uav_state: torch.Tensor,
        prev_option: torch.Tensor,
    ) -> torch.Tensor:
        del candidate_node_indices, candidate_padding_mask, action_mask, uav_state
        encoded, _, _ = self._encode_actor_global(
            global_node_inputs, spatio_pos_encoding, global_edge_mask, global_node_padding_mask
        )
        current_feature = self._gather_current(encoded, current_node_indices)
        b, n, _ = current_feature.shape
        agent_ids = torch.arange(n, device=current_feature.device).view(1, n).expand(b, n)
        agent_feature = self.actor_agent_embedding(agent_ids)
        return self.termination_head(
            current_feature + agent_feature + self.option_embedding_for_termination(prev_option.long())
        ).squeeze(-1)

    def forward(
        self,
        global_node_inputs: torch.Tensor,
        spatio_pos_encoding: torch.Tensor,
        global_edge_mask: torch.Tensor,
        global_node_padding_mask: torch.Tensor,
        current_node_indices: torch.Tensor,
        candidate_node_indices: torch.Tensor,
        candidate_padding_mask: torch.Tensor,
        action_mask: torch.Tensor,
        uav_state: torch.Tensor,
        prev_option: torch.Tensor,
        current_option: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, g, _ = global_node_inputs.shape
        m = candidate_node_indices.shape[-1]
        encoded, _, _ = self._encode_actor_global(
            global_node_inputs, spatio_pos_encoding, global_edge_mask, global_node_padding_mask
        )
        current_feature = self._gather_current(encoded, current_node_indices)
        candidate_feature = self._gather_candidates(encoded, candidate_node_indices)
        agent_ids = torch.arange(n, device=current_feature.device).view(1, n).expand(b, n)
        agent_feature = self.actor_agent_embedding(agent_ids)
        if self.cfg.disable_options:
            current_option = torch.zeros_like(prev_option.long())
            termination_logits = torch.full_like(prev_option.float(), -20.0)
        else:
            prev_feature = current_feature + agent_feature + self.option_embedding_for_termination(prev_option.long())
            termination_logits = self.termination_head(prev_feature).squeeze(-1)
            if current_option is None:
                current_option = prev_option.long()
        query = current_feature + agent_feature
        if not self.cfg.disable_options:
            query = query + self.option_embedding_for_policy(current_option.long())
        flat_query = query.reshape(b * n, 1, -1)
        flat_candidates = candidate_feature.reshape(b * n, m, -1)
        pointer_mask = (action_mask | candidate_padding_mask | (candidate_node_indices < 0)).reshape(b * n, 1, m).bool()
        enhanced_query = self.actor_decoder(flat_query, flat_candidates, pointer_mask)
        logp = self.pointer(enhanced_query, flat_candidates, pointer_mask).squeeze(1)
        waypoint_logits = masked_logits(logp.reshape(b, n, m), action_mask | candidate_padding_mask | (candidate_node_indices < 0))
        values = self._critic_value(
            global_node_inputs,
            spatio_pos_encoding,
            global_edge_mask,
            global_node_padding_mask,
            current_node_indices,
            uav_state,
        )
        return termination_logits, waypoint_logits, values

    @torch.no_grad()
    def act(
        self,
        global_node_inputs: torch.Tensor,
        spatio_pos_encoding: torch.Tensor,
        global_edge_mask: torch.Tensor,
        global_node_padding_mask: torch.Tensor,
        current_node_indices: torch.Tensor,
        candidate_node_indices: torch.Tensor,
        candidate_padding_mask: torch.Tensor,
        action_mask: torch.Tensor,
        uav_state: torch.Tensor,
        prev_option: torch.Tensor,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action, option, terminate, _, _, _ = self.act_with_info(
            global_node_inputs,
            spatio_pos_encoding,
            global_edge_mask,
            global_node_padding_mask,
            current_node_indices,
            candidate_node_indices,
            candidate_padding_mask,
            action_mask,
            uav_state,
            prev_option,
            greedy=greedy,
        )
        return action, option, terminate

    def evaluate_actions(
        self,
        global_node_inputs: torch.Tensor,
        spatio_pos_encoding: torch.Tensor,
        global_edge_mask: torch.Tensor,
        global_node_padding_mask: torch.Tensor,
        current_node_indices: torch.Tensor,
        candidate_node_indices: torch.Tensor,
        candidate_padding_mask: torch.Tensor,
        action_mask: torch.Tensor,
        uav_state: torch.Tensor,
        prev_option: torch.Tensor,
        actions: torch.Tensor,
        terminations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cfg.disable_options or self.cfg.disable_termination:
            terminations = torch.zeros_like(terminations)
        current_option = torch.where(terminations.bool(), 1 - prev_option.long(), prev_option.long())
        termination_logits, logits, values = self.forward(
            global_node_inputs,
            spatio_pos_encoding,
            global_edge_mask,
            global_node_padding_mask,
            current_node_indices,
            candidate_node_indices,
            candidate_padding_mask,
            action_mask,
            uav_state,
            prev_option,
            current_option=current_option,
        )
        action_dist = torch.distributions.Categorical(logits=logits)
        action_log_prob = action_dist.log_prob(actions.long())
        action_entropy = action_dist.entropy()
        if self.cfg.disable_options or self.cfg.disable_termination:
            beta = torch.zeros_like(termination_logits)
            term_log_prob = torch.zeros_like(action_log_prob)
            term_entropy = torch.zeros_like(action_entropy)
        else:
            term_dist = torch.distributions.Bernoulli(logits=termination_logits)
            beta = torch.sigmoid(termination_logits)
            term_log_prob = term_dist.log_prob(terminations.float())
            term_entropy = term_dist.entropy()
        return action_log_prob + term_log_prob, action_entropy + term_entropy, values, beta

    @torch.no_grad()
    def act_with_info(
        self,
        global_node_inputs: torch.Tensor,
        spatio_pos_encoding: torch.Tensor,
        global_edge_mask: torch.Tensor,
        global_node_padding_mask: torch.Tensor,
        current_node_indices: torch.Tensor,
        candidate_node_indices: torch.Tensor,
        candidate_padding_mask: torch.Tensor,
        action_mask: torch.Tensor,
        uav_state: torch.Tensor,
        prev_option: torch.Tensor,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cfg.disable_options or self.cfg.disable_termination:
            termination_logits = torch.zeros_like(prev_option.float())
            beta = torch.zeros_like(prev_option.float())
            terminate = torch.zeros_like(prev_option, dtype=torch.bool)
        else:
            termination_logits = self.termination_logits(
                global_node_inputs,
                spatio_pos_encoding,
                global_edge_mask,
                global_node_padding_mask,
                current_node_indices,
                candidate_node_indices,
                candidate_padding_mask,
                action_mask,
                uav_state,
                prev_option,
            )
            beta = torch.sigmoid(termination_logits)
            terminate = beta > 0.5 if greedy else torch.distributions.Bernoulli(probs=beta).sample().bool()
        option = torch.where(terminate, 1 - prev_option.long(), prev_option.long())
        if self.cfg.disable_options:
            option = torch.zeros_like(prev_option.long())
        _, logits, values = self.forward(
            global_node_inputs,
            spatio_pos_encoding,
            global_edge_mask,
            global_node_padding_mask,
            current_node_indices,
            candidate_node_indices,
            candidate_padding_mask,
            action_mask,
            uav_state,
            prev_option,
            current_option=option,
        )
        action_dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if greedy else action_dist.sample()
        action_log_prob = action_dist.log_prob(action)
        if self.cfg.disable_options or self.cfg.disable_termination:
            term_log_prob = torch.zeros_like(action_log_prob)
        else:
            term_log_prob = torch.distributions.Bernoulli(logits=termination_logits).log_prob(terminate.float())
        log_prob = action_log_prob + term_log_prob
        return action, option, terminate, log_prob, values, beta
