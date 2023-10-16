import torch
from torch import nn
from dataclasses import dataclass
from pathlib import Path
import json
from typing import List

from mistral.rope import precompute_freqs_cis
from mistral.cache import CacheView, RotatingBufferCache
from mistral.layers.attention import Attention
from mistral.layers.feed_forward import FeedForward
from mistral.layers.rms_norm import RMSNorm
from mistral.config.model_config import ModelArgs


class TransformerBlock(nn.Module):
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
        self, x: torch.Tensor, freqs_cis: torch.Tensor, cache: CacheView
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x), freqs_cis, cache)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class Transformer(nn.Module):
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

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: RotatingBufferCache,
        seqlens: List[int],
    ) -> torch.Tensor:
        assert len(seqlens) <= self.args.max_batch_size, f"Max batch size is {self.args.max_batch_size}, got batch size of {len(seqlens)}"
        assert sum(seqlens) == input_ids.shape[0], (sum(seqlens), input_ids.shape[0])

        input_metadata = cache.get_input_metadata(seqlens)
        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[input_metadata.positions]

        for layer_id, layer in enumerate(self.layers):
            h = layer(h, freqs_cis, cache.get_view(layer_id, input_metadata))

        cache.update_seqlens(seqlens)

        return self.output(self.norm(h)).float()

    @staticmethod
    def from_folder(folder: Path, max_batch_size: int = 1, device="cuda", dtype=torch.float16) -> "Transformer":
        with open(folder / 'params.json', 'r') as f:
            model_args = ModelArgs(**json.loads(f.read()))
        model_args.max_batch_size = max_batch_size
        model = Transformer(model_args).to(device=device, dtype=dtype)
        loaded = torch.load(folder / 'consolidated.00.pth')
        model.load_state_dict(loaded)
        return model
