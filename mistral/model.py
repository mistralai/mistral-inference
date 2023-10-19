import torch
from torch import nn
from dataclasses import dataclass
from pathlib import Path
import json
from typing import List, Optional

from mistral.rope import precompute_freqs_cis, apply_rotary_emb
from mistral.cache import CacheView, RotatingBufferCache

from xformers.ops.fmha import (
    memory_efficient_attention,
)


@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    sliding_window: int
    norm_eps: float
    vocab_size: int

    max_batch_size: int = 0


@dataclass
class SimpleInputMetadata:
    # rope absolute positions
    positions: torch.Tensor

    @staticmethod
    def from_seqlens(seqlens: List[int], device: torch.device) -> "SimpleInputMetadata":
        return SimpleInputMetadata(
            positions = torch.cat(
                [torch.arange(0, seqlen) for seqlen in seqlens]
            ).to(device=device, dtype=torch.long)
        )


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int):
    """
    Repeat keys and values along a specified dimension.

    Args:
        keys (torch.Tensor): The keys tensor to repeat.
        values (torch.Tensor): The values tensor to repeat.
        repeats (int): The number of times to repeat.
        dim (int): The dimension along which to repeat.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Repeated keys and values.
    """
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


class Attention(nn.Module):
    """
    A multi-head attention module.

    Args:
        args (ModelArgs): Model configuration arguments.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads
        
        self.repeats = self.n_heads // self.n_kv_heads
        self.sliding_window = self.args.sliding_window

        self.scale = self.args.head_dim**-0.5

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * args.head_dim,
            bias=False
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            bias=False
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            bias=False
        )
        self.wo = nn.Linear(
            args.n_heads * args.head_dim,
            args.dim,
            bias=False
        )
        

    def forward(
        self, x: torch.Tensor, 
        freqs_cis: torch.Tensor,
        cache: Optional[CacheView],
    ) -> torch.Tensor:
        """
        Forward pass of the multi-head attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Circular positional encodings.
            cache (CacheView): Cache view for storing and retrieving keys and values.

        Returns:
            torch.Tensor: Output tensor.
        """
        seqlen_sum, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(seqlen_sum, self.n_heads, self.args.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.args.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.args.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if cache is None:
            key, val = xk, xv
        elif cache.prefill:
            key, val = cache.interleave_kv(xk, xv)
            cache.update(xk, xv)
        else: 
            cache.update(xk, xv)
            key, val = cache.key, cache.value
            key = key.view(seqlen_sum * cache.sliding_window, self.n_kv_heads, self.args.head_dim)
            val = val.view(seqlen_sum * cache.sliding_window, self.n_kv_heads, self.args.head_dim)

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        output = memory_efficient_attention(xq, key, val, None if cache is None else cache.mask)

        return self.wo(output.view_as(x))


class FeedForward(nn.Module):
    """
    A feedforward neural network module.

    Args:
        args (ModelArgs): Model configuration arguments.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(
            args.dim,
            args.hidden_dim,
            bias=False
        )
        self.w2 = nn.Linear(
            args.hidden_dim,
            args.dim,
            bias=False
        )
        self.w3 = nn.Linear(
            args.dim,
            args.hidden_dim,
            bias=False
        )

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the feedforward network.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization module.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value to prevent division by zero. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass of the RMSNorm module.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    """
    A single transformer block.

    Args:
        args (ModelArgs): Model configuration arguments.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, cache: Optional[CacheView]
    ) -> torch.Tensor:
        """
        Forward pass of the transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Circular positional encodings.
            cache (CacheView): Cache view for storing and retrieving keys and values.

        Returns:
            torch.Tensor: Output tensor.
        """
        r = self.attention.forward(self.attention_norm(x), freqs_cis, cache)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class Transformer(nn.Module):
    """
    A Transformer model.

    Args:
        args (ModelArgs): Model configuration arguments.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        assert self.vocab_size > 0

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = torch.nn.ModuleList(
            [TransformerBlock(args=args) for _ in range(args.n_layers)]
        )

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(
            args.dim,
            args.vocab_size,
            bias=False
        )

        self.freqs_cis = precompute_freqs_cis(self.args.head_dim, 128_000).to("cuda")

    @property
    def dtype(self) -> torch.dtype:
        return self.tok_embeddings.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.tok_embeddings.weight.device

    def forward_partial(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        cache: Optional[RotatingBufferCache]=None,
    ) -> torch.Tensor:
        """
        Forward pass of the Transformer model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            cache (RotatingBufferCache): Rotating buffer cache for storing keys and values.
            seqlens (List[int]): List of sequence lengths in the batch.

        Returns:
            torch.Tensor: Output tensor.
        """
        assert len(seqlens) <= self.args.max_batch_size, f"Max batch size is {self.args.max_batch_size}, got batch size of {len(seqlens)}"
        assert sum(seqlens) == input_ids.shape[0], (sum(seqlens), input_ids.shape[0])
        if cache is not None:
            input_metadata = cache.get_input_metadata(seqlens)
        else:
            input_metadata = SimpleInputMetadata.from_seqlens(seqlens, self.device)
        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[input_metadata.positions]

        for layer_id, layer in enumerate(self.layers):
            cache_view = None if cache is None else cache.get_view(layer_id, input_metadata)
            h = layer(h, freqs_cis, cache_view)
        
        if cache is not None:
            cache.update_seqlens(seqlens)

        return self.norm(h)

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        cache: Optional[RotatingBufferCache]=None,
    ) -> torch.Tensor:
        return self.output(self.forward_partial(
            input_ids, seqlens, cache=cache
        )).float()

    @staticmethod
    def from_folder(folder: Path, max_batch_size: int = 1, device="cuda", dtype=torch.float16) -> "Transformer":
        """
        Load a Transformer model from a folder.

        Args:
            folder (Path): Path to the folder containing the model files.
            max_batch_size (int, optional): Maximum batch size for the model. Defaults to 1.
            device (str, optional): Device on which to load the model (e.g., 'cuda' or 'cpu'). Defaults to 'cuda'.
            dtype (torch.dtype, optional): Data type for the model (e.g., torch.float16 or torch.float32). Defaults to torch.float16.

        Returns:
            Transformer: Loaded Transformer model.
        """
        with open(folder / 'params.json', 'r') as f:
            model_args = ModelArgs(**json.loads(f.read()))
        model_args.max_batch_size = max_batch_size
        model = Transformer(model_args).to(device=device, dtype=dtype)
        loaded = torch.load(folder / 'consolidated.00.pth')
        model.load_state_dict(loaded)
        return model
