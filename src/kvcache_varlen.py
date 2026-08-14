"""Padding-free batched forward for the FastSD cloud target.

Replaces the pad-to-max dense path of ``KVCacheModel_batching.generate`` with a
variable-length forward that never feeds padding tokens into the model:

- linear layers (layernorm, projections, MLP) run once over the concatenation
  of all real tokens;
- attention runs per sequence with its own q/k lengths against its own KV
  cache, so no KV zero-padding and no per-step full-KV copy alignment;
- per-proc KV caches and prob history keep exactly the semantics of
  ``KVCacheModel_batching``: ``_past_key_values`` and ``_prob_history`` are
  updated identically and the returned tuple matches ``generate``.

Callers (``engine.handle_request_batch``) are responsible for slicing each
sequence to its *residual* (the tokens not yet covered by that proc's cache):

- prefill fresh        -> full prompt (cache empty / reset)
- prefill continuation -> the new chunk (``x[:, cached_len:]``)
- verify tail-only     -> the received tail segment as-is
- verify full-prefix   -> tokens beyond the cached length (``x[:, cached_len:]``)

The padded implementation remains available in ``kvcache_batching.py`` and is
selected with ``--kv_batch_mode padded``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from .util import norm_logits, sample


def supports_varlen_model(model) -> bool:
    """Return whether the hand-fused path matches the model implementation.

    The current implementation explicitly follows Qwen3's q/k RMS norms,
    rotary embedding and decoder-layer layout.  Unsupported models must use
    the legacy Transformers path instead of failing at runtime.
    """
    config = getattr(model, "config", None)
    return bool(
        getattr(config, "model_type", None) == "qwen3"
        and hasattr(model, "model")
        and hasattr(model.model, "layers")
        and bool(model.model.layers)
        and all(
            hasattr(layer.self_attn, "q_norm")
            and hasattr(layer.self_attn, "k_norm")
            for layer in model.model.layers[:1]
        )
    )


def varlen_generate(
    kv_manager,
    model,
    residuals,      # list[Tensor]: each (1, r_i) on the model device
    proc_ids,       # list[Any] aligned with residuals
    pad_token_id,   # signature parity with KVCacheModel_batching.generate; unused
    is_prefill: bool = False,
    input_lens=None,  # signature parity; unused (no padding in this path)
):
    """Padding-free forward for one mixed batch.

    Mirrors ``KVCacheModel_batching.generate``: appends per-proc logits to
    ``_prob_history`` (with per-position ``norm_logits``), appends the new K/V
    to each proc's ``DynamicCache``, and returns the per-proc extended
    sequences (input + sampled token), identical in structure to ``generate``.
    """
    del pad_token_id, input_lens, is_prefill  # no padding / no flag split in this path
    if len(residuals) != len(proc_ids):
        raise ValueError("residuals and proc_ids must be aligned")
    if not residuals:
        return []

    vocab_size = kv_manager.vocab_size
    temperature = kv_manager._temperature
    top_k = kv_manager._top_k
    top_p = kv_manager._top_p
    device = model.device

    # ---- 1. per-seq cached lengths & flattened input ----------------------
    cached_lens = []
    flat_parts = []
    for pid, res in zip(proc_ids, residuals):
        cache = kv_manager._past_key_values.get(pid)
        # A missing cache means fresh prefill: the engine resets (pops) the
        # entry before admission, exactly like the padded path.
        cached_lens.append(0 if cache is None else int(cache.get_seq_length()))
        flat_parts.append(res.to(device).reshape(-1))
    flat_ids = torch.cat(flat_parts, dim=0)  # (total,)

    seq_ranges = []
    start = 0
    for res, cached_len in zip(residuals, cached_lens):
        length = int(res.numel())
        seq_ranges.append((start, start + length, cached_len))
        start += length

    # ---- 2. layer-by-layer forward -----------------------------------------
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    head_dim = model.model.layers[0].self_attn.head_dim

    hidden_states = model.model.embed_tokens(flat_ids)

    # [layer_idx][seq_idx] -> (k_new, v_new), each (1, n_kv_heads, Lq, head_dim)
    new_kv_per_layer = []

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        normed = layer.input_layernorm(hidden_states)

        q = attn.q_proj(normed).view(-1, n_heads, head_dim)
        k = attn.k_proj(normed).view(-1, n_kv_heads, head_dim)
        v = attn.v_proj(normed).view(-1, n_kv_heads, head_dim)
        q = attn.q_norm(q)
        k = attn.k_norm(k)

        attn_out = torch.zeros_like(normed)
        new_kv_layer = []
        for (s, e, cached_len), pid in zip(seq_ranges, proc_ids):
            seq_len = e - s
            qi = q[s:e].unsqueeze(0).transpose(1, 2)   # (1, n_heads, Lq, hd)
            ki = k[s:e].unsqueeze(0).transpose(1, 2)   # (1, n_kv, Lq, hd)
            vi = v[s:e].unsqueeze(0).transpose(1, 2)

            # Per-sequence RoPE: both q and new k carry their global positions,
            # starting right after the cached prefix (the cached K/V were
            # already rotated at their own positions when stored).
            pos = torch.arange(cached_len, cached_len + seq_len, dtype=torch.long, device=device).unsqueeze(0)
            cos, sin = model.model.rotary_emb(qi, pos)
            qi, _ = apply_rotary_pos_emb(qi, qi, cos, sin)
            ki, _ = apply_rotary_pos_emb(ki, ki, cos, sin)

            cache = kv_manager._past_key_values.get(pid)
            k_full, v_full = ki, vi
            if cache is not None and cache.get_seq_length() > 0:
                k_old, v_old = cache.to_legacy_cache()[layer_idx]
                k_full = torch.cat([k_old.to(ki.dtype), ki], dim=2)
                v_full = torch.cat([v_old.to(vi.dtype), vi], dim=2)

            # GQA: SDPA in torch 2.2 does not broadcast kv heads, so expand
            # them to the query head count for the attention call only. The
            # per-proc cache keeps the original kv-head tensors.
            n_groups = n_heads // n_kv_heads
            k_exp = k_full.repeat_interleave(n_groups, dim=1)
            v_exp = v_full.repeat_interleave(n_groups, dim=1)

            # Causal masking by *global* position: query global position
            # (cached_len + i) may attend key global positions <= itself.
            # ``is_causal`` uses local indices and is only correct for a
            # fresh prefill; cached continuations need an explicit mask.
            Lq, Lk = seq_len, cached_len + seq_len
            if Lq == Lk and cached_len == 0 and Lq > 1:
                o = F.scaled_dot_product_attention(qi, k_exp, v_exp, is_causal=True)
            else:
                visible = (
                    torch.arange(Lq, device=device)[:, None] + cached_len
                    >= torch.arange(Lk, device=device)[None, :]
                )
                attn_bias = torch.zeros((1, 1, Lq, Lk), device=device, dtype=qi.dtype)
                attn_bias.masked_fill_(~visible, float("-inf"))
                o = F.scaled_dot_product_attention(qi, k_exp, v_exp, attn_mask=attn_bias)
            o = o.transpose(1, 2).reshape(seq_len, -1)
            attn_out[s:e] = attn.o_proj(o)
            new_kv_layer.append((ki, vi))

        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))
        new_kv_per_layer.append(new_kv_layer)

    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)[:, :vocab_size]  # (total, V)

    # ---- 3. write back per-proc prob history / KV cache --------------------
    new_x = []
    for i, pid in enumerate(proc_ids):
        s, e, cached_len = seq_ranges[i]
        seq_len = e - s

        logits_i = logits[s:e].unsqueeze(0)  # (1, Lq, V)
        # Per-position normalization matches the padded path (temp==0 branch is
        # row-unsafe for batched inputs, so normalize position by position).
        for t in range(logits_i.shape[1]):
            logits_i[0:1, t, :] = norm_logits(logits_i[0:1, t, :], temperature, top_k, top_p)

        prev = kv_manager._prob_history.get(pid)
        if prev is not None and prev.shape[1] > 0:
            kv_manager._prob_history[pid] = torch.cat([prev, logits_i], dim=1)
        else:
            kv_manager._prob_history[pid] = logits_i

        # Append new K/V per layer to the persistent cache.
        old_cache = kv_manager._past_key_values.get(pid)
        if old_cache is not None and old_cache.get_seq_length() > 0:
            old_legacy = old_cache.to_legacy_cache()
            new_legacy = [
                (
                    torch.cat([old_legacy[layer_idx][0].to(new_kv_per_layer[layer_idx][i][0].dtype),
                               new_kv_per_layer[layer_idx][i][0]], dim=2),
                    torch.cat([old_legacy[layer_idx][1].to(new_kv_per_layer[layer_idx][i][1].dtype),
                               new_kv_per_layer[layer_idx][i][1]], dim=2),
                )
                for layer_idx in range(len(new_kv_per_layer))
            ]
            kv_manager._past_key_values[pid] = DynamicCache.from_legacy_cache(new_legacy)
        else:
            new_legacy = [
                (new_kv_per_layer[layer_idx][i][0], new_kv_per_layer[layer_idx][i][1])
                for layer_idx in range(len(new_kv_per_layer))
            ]
            kv_manager._past_key_values[pid] = DynamicCache.from_legacy_cache(new_legacy)

        sampled = sample(logits_i[0, -1:, :])
        next_tokens_1d = sampled.reshape(-1)
        new_x.append(torch.cat([flat_parts[i], next_tokens_1d], dim=0))

    return new_x
