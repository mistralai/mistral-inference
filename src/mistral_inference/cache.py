from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from xformers.ops.fmha.attn_bias import (  # type: ignore
    AttentionBias,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalMask,
)


def get_cache_sizes(n_layers: int, max_seq_len: int, sliding_window: Optional[int] | Optional[List[int]]) -> List[int]:
    if sliding_window is None:
        return n_layers * [max_seq_len]
    elif isinstance(sliding_window, int):
        return n_layers * [sliding_window]
    else:
        assert isinstance(sliding_window, list), f"Expected list, got {type(sliding_window)}"
        assert (
            n_layers % len(sliding_window) == 0
        ), f"Expected n_layers % len(sliding_window) == 0, got {n_layers} % {len(sliding_window)}"
        num_repeats = n_layers // len(sliding_window)
        return num_repeats * [w if w is not None else max_seq_len for w in sliding_window]


@dataclass
class CacheInputMetadata:
    # # rope absolute positions
    # positions: torch.Tensor
    # # where tokens should go in the cache
    # cache_positions: torch.Tensor

    # # if prefill, use block diagonal causal mask
    # # else use causal with padded key mask
    # prefill: bool
    # mask: AttentionBias
    # seqlens: List[int]
    # rope absolute positions
    positions: torch.Tensor
    # which elements in the sequences need to be cached
    to_cache_mask: torch.Tensor
    # how many elements are cached per sequence
    cached_elements: torch.Tensor
    # where tokens should go in the cache
    cache_positions: torch.Tensor
    # if prefill, use block diagonal causal mask
    # else use causal with padded key mask
    prefill: bool
    mask: AttentionBias
    seqlens: List[int]


def interleave_list(l1: List[torch.Tensor], l2: List[torch.Tensor]) -> List[torch.Tensor]:
    assert len(l1) == len(l2)
    return [v for pair in zip(l1, l2) for v in pair]


def unrotate(cache: torch.Tensor, seqlen: int) -> torch.Tensor:
    assert cache.ndim == 3  # (W, H, D)
    position = seqlen % cache.shape[0]
    if seqlen < cache.shape[0]:
        return cache[:seqlen]
    elif position == 0:
        return cache
    else:
        return torch.cat([cache[position:], cache[:position]], dim=0)


class CacheView:
    def __init__(
        self,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        metadata: CacheInputMetadata,
        kv_seqlens: torch.Tensor,
    ):
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.kv_seqlens = kv_seqlens
        self.metadata = metadata

    def update(self, xk: torch.Tensor, xv: torch.Tensor) -> None:
        """
        to_cache_mask masks the last [max_seq_len] tokens in each sequence
        """
        n_kv_heads, head_dim = self.cache_k.shape[-2:]
        flat_cache_k = self.cache_k.view(-1, n_kv_heads, head_dim)
        flat_cache_v = self.cache_v.view(-1, n_kv_heads, head_dim)

        flat_cache_k.index_copy_(0, self.metadata.cache_positions, xk[self.metadata.to_cache_mask])
        flat_cache_v.index_copy_(0, self.metadata.cache_positions, xv[self.metadata.to_cache_mask])

    def interleave_kv(self, xk: torch.Tensor, xv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is a naive implementation and not optimized for speed.
        """
        assert xk.ndim == xv.ndim == 3  # (B * T, H, D)
        assert xk.shape == xv.shape

        if all([s == 0 for s in self.metadata.seqlens]):
            # No cache to interleave
            return xk, xv

        # Make it a list of [(T, H, D)]
        xk: Tuple[torch.Tensor] = torch.split(xk, self.metadata.seqlens)  # type: ignore
        xv: Tuple[torch.Tensor] = torch.split(xv, self.metadata.seqlens)  # type: ignore
        assert len(xk) == len(self.kv_seqlens), f"Batch size is {len(self.kv_seqlens)}, got {len(xk)}"

        # Order elements in cache by position by unrotating
        cache_k = [unrotate(t, s) for t, s in zip(self.cache_k, self.kv_seqlens)]
        cache_v = [unrotate(t, s) for t, s in zip(self.cache_v, self.kv_seqlens)]

        interleaved_k = interleave_list(cache_k, list(xk))
        interleaved_v = interleave_list(cache_v, list(xv))

        return torch.cat(interleaved_k, dim=0), torch.cat(interleaved_v, dim=0)

    @property
    def max_seq_len(self) -> int:
        return self.cache_k.shape[1]

    @property
    def key(self) -> torch.Tensor:
        return self.cache_k[: len(self.kv_seqlens)]

    @property
    def value(self) -> torch.Tensor:
        return self.cache_v[: len(self.kv_seqlens)]

    @property
    def prefill(self) -> bool:
        return self.metadata.prefill

    @property
    def mask(self) -> AttentionBias:
        return self.metadata.mask


class BufferCache:
    """
    This is an example that implements a buffer cache, allowing for variable length sequences.
    Allocated cache is rectangular which is wasteful (see PagedAttention for better mechanisms)
    """

    def __init__(
        self,
        n_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        sliding_window: Optional[int] | Optional[List[int]] = None,
    ):
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_layers = n_layers

        self.cache_sizes: List[int] = get_cache_sizes(n_layers, max_seq_len, sliding_window)
        assert len(self.cache_sizes) == n_layers, f"Expected {n_layers} cache sizes, got {len(self.cache_sizes)}"

        self.cache_k = {}
        self.cache_v = {}
        for i, cache_size in enumerate(self.cache_sizes):
            self.cache_k[i] = torch.empty((max_batch_size, cache_size, n_kv_heads, head_dim))
            self.cache_v[i] = torch.empty((max_batch_size, cache_size, n_kv_heads, head_dim))

        # holds the valid length for each batch element in the cache
        self.kv_seqlens: Optional[torch.Tensor] = None

    def get_view(self, layer_id: int, metadata: CacheInputMetadata) -> CacheView:
        assert self.kv_seqlens is not None
        return CacheView(self.cache_k[layer_id], self.cache_v[layer_id], metadata, self.kv_seqlens)

    def reset(self) -> None:
        self.kv_seqlens = None

    def init_kvseqlens(self, batch_size: int) -> None:
        self.kv_seqlens = torch.zeros((batch_size,), device=self.device, dtype=torch.long)

    @property
    def device(self) -> torch.device:
        return self.cache_k[0].device

    def to(self, device: torch.device, dtype: torch.dtype) -> "BufferCache":
        for i in range(self.n_layers):
            self.cache_k[i] = self.cache_k[i].to(device=device, dtype=dtype)
            self.cache_v[i] = self.cache_v[i].to(device=device, dtype=dtype)

        return self

    def update_seqlens(self, seqlens: List[int]) -> None:
        assert self.kv_seqlens is not None
        self.kv_seqlens += torch.tensor(seqlens, device=self.device, dtype=torch.long)

    def get_input_metadata(self, seqlens: List[int]) -> List[CacheInputMetadata]:
        """
        input = seqlens [5,7,2] // seqpos [0, 1, 3] // sliding_window 3
        --> only cache last 3 tokens in each sequence
        - to_cache_mask = [0 0 1 1 1 | 0 0 0 0 1 1 1 | 1 1]
        - cached_elements = [3 | 3 | 2]
        --> absolute positions are used for rope
        - positions = [0 1 2 3 4 | 1 2 3 4 5 6 7 | 3 4]
        --> cache positions are positions cache_masked, modulo sliding_window + batch_idx * sliding_window
        - cache_positions = [2 0 1 | 5 3 4 | 6 7]
        """
        metadata: List[CacheInputMetadata] = []

        if self.kv_seqlens is None:
            self.init_kvseqlens(len(seqlens))

        assert self.kv_seqlens is not None
        assert len(seqlens) == len(
            self.kv_seqlens
        ), f"Batch size is {len(self.kv_seqlens)}, got {len(seqlens)}, did you forget to reset cache?"
        seqpos = self.kv_seqlens.tolist()
        assert len(seqlens) > 0, seqlens

        for cache_size in self.cache_sizes:
            metadata.append(self._get_input_metadata_layer(cache_size, seqlens, seqpos))

        return metadata

    def _get_input_metadata_layer(self, cache_size: int, seqlens: List[int], seqpos: List[int]) -> CacheInputMetadata:
        masks = [[x >= seqlen - cache_size for x in range(seqlen)] for seqlen in seqlens]
        to_cache_mask = torch.tensor(sum(masks, []), device=self.device, dtype=torch.bool)
        cached_elements = torch.tensor([sum(mask) for mask in masks], device=self.device, dtype=torch.long)
        positions = torch.cat([torch.arange(pos, pos + seqlen) for pos, seqlen in zip(seqpos, seqlens)]).to(
            device=self.device, dtype=torch.long
        )
        batch_idx = torch.tensor(
            sum([[i] * seqlen for i, seqlen in enumerate(seqlens)], []), device=self.device, dtype=torch.long
        )
        cache_positions = positions % cache_size + batch_idx * cache_size
        first_prefill = seqpos[0] == 0
        subsequent_prefill = any(seqlen > 1 for seqlen in seqlens)
        if first_prefill:
            assert all([pos == 0 for pos in seqpos]), seqpos
            mask = BlockDiagonalCausalMask.from_seqlens(seqlens).make_local_attention(cache_size)
        elif subsequent_prefill:
            assert self.kv_seqlens is not None
            mask = BlockDiagonalMask.from_seqlens(
                q_seqlen=seqlens,
                kv_seqlen=[
                    s + cached_s.clamp(max=cache_size).item() for (s, cached_s) in zip(seqlens, self.kv_seqlens)
                ],
            ).make_local_attention_from_bottomright(cache_size)
        else:
            mask = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=seqlens,
                kv_padding=cache_size,
                kv_seqlen=(self.kv_seqlens + cached_elements).clamp(max=cache_size).tolist(),
            )
        return CacheInputMetadata(
            positions=positions,
            to_cache_mask=to_cache_mask,
            cached_elements=cached_elements,
            cache_positions=cache_positions[to_cache_mask],
            prefill=first_prefill or subsequent_prefill,
            mask=mask,
            seqlens=seqlens,
        )
