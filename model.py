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
        self.initial_embedding = nn.Linear(16, e)
        self.encoder = Encoder(embedding_dim=e, n_heads=8, n_layers=3)
        self.decoder = Decoder(embedding_dim=e, n_heads=8, n_layers=1)
        self.pointer = SingleHeadAttention(e)
        self.uav_encoder = nn.Sequential(nn.Linear(4, e), nn.ReLU(inplace=True), nn.Linear(e, e))
        self.team_encoder = nn.Sequential(nn.Linear(12, e), nn.ReLU(inplace=True), nn.Linear(e, e))
        self.state_embedding = nn.Linear(e * 3, e)
        self.current_embedding = nn.Linear(e * 2, e)
        self.termination_head = nn.Sequential(nn.Linear(e, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        self.option_embedding_for_termination = nn.Embedding(2, e)
        self.option_embedding_for_policy = nn.Embedding(2, e)
        self.value_head = nn.Sequential(nn.Linear(e * 2, e), nn.ReLU(inplace=True), nn.Linear(e, 1))

    def _flat_masks(self, node_padding_mask: torch.Tensor, action_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, m = node_padding_mask.shape
        padding = node_padding_mask.reshape(b * n, m).bool()
        action = (action_mask | node_padding_mask).reshape(b * n, m).bool()
        encoder_mask = padding.unsqueeze(1).expand(-1, m, -1)
        pointer_mask = action.unsqueeze(1)
        return encoder_mask, pointer_mask

    def _encode(
        self,
        node_inputs: torch.Tensor,
        node_padding_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, m, _ = node_inputs.shape
        flat_nodes = node_inputs.reshape(b * n, m, -1)
        encoder_mask, pointer_mask = self._flat_masks(node_padding_mask, action_mask)
        embedded = self.initial_embedding(flat_nodes)
        encoded = self.encoder(embedded, encoder_mask)
        encoded = encoded.masked_fill(node_padding_mask.reshape(b * n, m, 1).bool(), 0.0)
        return encoded.reshape(b, n, m, -1), encoder_mask, pointer_mask

    def _base_state(
        self,
        encoded: torch.Tensor,
        node_padding_mask: torch.Tensor,
        uav_state: torch.Tensor,
        team_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = (~node_padding_mask.bool()).float().unsqueeze(-1)
        pooled = (encoded * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1.0)
        uav_emb = self.uav_encoder(uav_state.float())
        team_emb = self.team_encoder(team_summary.float()).unsqueeze(1).expand_as(uav_emb)
        base = self.state_embedding(torch.cat([pooled, uav_emb, team_emb], dim=-1))
        return base, pooled

    def termination_logits(
        self,
        node_inputs: torch.Tensor,
        node_padding_mask: torch.Tensor,
        action_mask: torch.Tensor,
        uav_state: torch.Tensor,
        prev_option: torch.Tensor,
        team_summary: torch.Tensor,
    ) -> torch.Tensor:
        encoded, _, _ = self._encode(node_inputs, node_padding_mask, action_mask)
        base, _ = self._base_state(encoded, node_padding_mask, uav_state, team_summary)
        prev_feature = base + self.option_embedding_for_termination(prev_option.long())
        return self.termination_head(prev_feature).squeeze(-1)

    def forward(
        self,
        node_inputs: torch.Tensor,
        node_padding_mask: torch.Tensor,
        action_mask: torch.Tensor,
        uav_state: torch.Tensor,
        prev_option: torch.Tensor,
        team_summary: torch.Tensor,
        current_option: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, m, _ = node_inputs.shape
        encoded, _, pointer_mask = self._encode(node_inputs, node_padding_mask, action_mask)
        base, pooled = self._base_state(encoded, node_padding_mask, uav_state, team_summary)
        if self.cfg.disable_options:
            current_option = torch.zeros_like(prev_option.long())
            termination_logits = torch.full_like(prev_option.float(), -20.0)
        else:
            prev_feature = base + self.option_embedding_for_termination(prev_option.long())
            termination_logits = self.termination_head(prev_feature).squeeze(-1)
            if current_option is None:
                current_option = prev_option.long()
        query = base + self.option_embedding_for_policy(current_option.long())
        flat_query = query.reshape(b * n, 1, -1)
        flat_encoded = encoded.reshape(b * n, m, -1)
        enhanced_query = self.decoder(flat_query, flat_encoded, node_padding_mask.reshape(b * n, 1, m).bool())
        state = self.current_embedding(torch.cat([enhanced_query, flat_query], dim=-1))
        logp = self.pointer(state, flat_encoded, pointer_mask).squeeze(1)
        waypoint_logits = masked_logits(logp.reshape(b, n, m), action_mask | node_padding_mask)
        flat_state = state.reshape(b, n, -1)
        values = self.value_head(torch.cat([flat_state, pooled], dim=-1)).squeeze(-1)
        return termination_logits, waypoint_logits, values

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
        action, option, terminate, _, _, _ = self.act_with_info(
            node_inputs, node_padding_mask, action_mask, uav_state, prev_option, team_summary, greedy=greedy
        )
        return action, option, terminate

    def evaluate_actions(
        self,
        node_inputs: torch.Tensor,
        node_padding_mask: torch.Tensor,
        action_mask: torch.Tensor,
        uav_state: torch.Tensor,
        prev_option: torch.Tensor,
        team_summary: torch.Tensor,
        actions: torch.Tensor,
        terminations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cfg.disable_options or self.cfg.disable_termination:
            terminations = torch.zeros_like(terminations)
        current_option = torch.where(terminations.bool(), 1 - prev_option.long(), prev_option.long())
        termination_logits, logits, values = self.forward(
            node_inputs,
            node_padding_mask,
            action_mask,
            uav_state,
            prev_option,
            team_summary,
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
        node_inputs: torch.Tensor,
        node_padding_mask: torch.Tensor,
        action_mask: torch.Tensor,
        uav_state: torch.Tensor,
        prev_option: torch.Tensor,
        team_summary: torch.Tensor,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        termination_logits = self.termination_logits(node_inputs, node_padding_mask, action_mask, uav_state, prev_option, team_summary)
        if self.cfg.disable_options or self.cfg.disable_termination:
            beta = torch.zeros_like(termination_logits)
            terminate = torch.zeros_like(prev_option, dtype=torch.bool)
        else:
            beta = torch.sigmoid(termination_logits)
            terminate = beta > 0.5 if greedy else torch.distributions.Bernoulli(probs=beta).sample().bool()
        option = torch.where(terminate, 1 - prev_option.long(), prev_option.long())
        if self.cfg.disable_options:
            option = torch.zeros_like(prev_option.long())
        _, logits, values = self.forward(
            node_inputs,
            node_padding_mask,
            action_mask,
            uav_state,
            prev_option,
            team_summary,
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
