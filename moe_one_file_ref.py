import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import fire
import torch
from sentencepiece import SentencePieceProcessor
from simple_parsing.helpers import Serializable
from torch import nn


@dataclass
class MoeArgs(Serializable):
    num_experts: int
    num_experts_per_tok: int


@dataclass
class ModelArgs(Serializable):
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    moe: MoeArgs

    max_batch_size: int = 0
    max_seq_len: int = 0


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=2)
    values = torch.repeat_interleave(values, repeats=repeats, dim=2)
    return keys, values


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    freqs_cis: complex - (seq_len, head_dim / 2)
    x: complex - (bsz, seq_len, head_dim / 2)
    """
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        freqs_cis.shape,
        (x.shape[1], x.shape[-1]),
    )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = _reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        self._cache_k: Optional[torch.Tensor] = None
        self._cache_v: Optional[torch.Tensor] = None

    def get_caches(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype, device = x.dtype, x.device
        assert x.shape[0] <= self.args.max_batch_size
        assert x.shape[1] <= self.args.max_seq_len
        if self._cache_k is None:
            self._cache_k = torch.empty(
                (
                    self.args.max_batch_size,
                    self.args.max_seq_len,
                    self.n_kv_heads,
                    self.args.head_dim,
                ),
                dtype=dtype,
                device=device,
            )
        if self._cache_v is None:
            self._cache_v = torch.empty(
                (
                    self.args.max_batch_size,
                    self.args.max_seq_len,
                    self.n_kv_heads,
                    self.args.head_dim,
                ),
                dtype=dtype,
                device=device,
            )
        return self._cache_k, self._cache_v

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        cache_k, cache_v = self.get_caches(x)

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.args.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.args.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.args.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # The cache is a rotating buffer
        scatter_pos = (positions % self.args.max_seq_len)[None, :, None, None]
        scatter_pos = scatter_pos.repeat(bsz, 1, self.n_kv_heads, self.args.head_dim)
        cache_k[:bsz].scatter_(dim=1, index=scatter_pos, src=xk)
        cache_v[:bsz].scatter_(dim=1, index=scatter_pos, src=xv)

        if positions.shape[0] > 1:
            # prefill
            key, value = repeat_kv(xk, xv, self.repeats)
        else:
            assert mask is None
            cur_pos = int(positions[-1].item() + 1)
            key, value = repeat_kv(
                cache_k[:bsz, :cur_pos, ...],
                cache_v[:bsz, :cur_pos, ...],
                self.repeats,
            )

        query = xq.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        # scores : [bsz, n_heads, seqlen | 1, seqlen]
        scores = torch.matmul(query, key.transpose(2, 3)) * self.scale

        if mask is not None:
            scores += mask[None, None, ...]

        scores = scores.float()
        scores = nn.functional.softmax(scores, dim=-1).type_as(query)
        output = torch.matmul(scores, value)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.args = moe_args

    def forward(self, inputs: torch.Tensor):
        inputs_squashed = inputs.view(-1, inputs.shape[-1])
        gate_logits = self.gate(inputs_squashed)
        weights, selected_experts = torch.topk(
            gate_logits, self.args.num_experts_per_tok
        )
        weights = nn.functional.softmax(
            weights,
            dim=1,
            dtype=torch.float,
        ).type_as(inputs)
        results = torch.zeros_like(inputs_squashed)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs_squashed[batch_idx]
            )
        return results.view_as(inputs)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = MoeLayer(
            experts=[FeedForward(args=args) for _ in range(args.moe.num_experts)],
            gate=nn.Linear(args.dim, args.moe.num_experts, bias=False),
            moe_args=args.moe,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x), freqs_cis, positions, mask)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


def precompute_freqs_cis(dim: int, end: int) -> torch.Tensor:
    theta = 1000000.0
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


class Transformer(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        pipeline_rank: int = 0,
        num_pipeline_ranks: int = 1,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0
        assert pipeline_rank < num_pipeline_ranks, (pipeline_rank, num_pipeline_ranks)
        self.pipeline_rank = pipeline_rank
        self.num_pipeline_ranks = num_pipeline_ranks
        self._precomputed_freqs_cis: Optional[torch.Tensor] = None

        # Modules specific to some ranks:
        self.tok_embeddings: Optional[nn.Embedding] = None
        self.norm: Optional[RMSNorm] = None
        self.output: Optional[nn.Linear] = None
        if pipeline_rank == 0:
            self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        if pipeline_rank == num_pipeline_ranks - 1:
            self.norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Initialize all layers but slice off those not of this rank.
        layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        num_layers_per_rank = math.ceil(args.n_layers / self.num_pipeline_ranks)
        offset = self.pipeline_rank * num_layers_per_rank
        end = min(args.n_layers, offset + num_layers_per_rank)
        self.layers = nn.ModuleDict({str(i): layers[i] for i in range(offset, end)})
        self.n_local_layers = len(self.layers)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def freqs_cis(self) -> torch.Tensor:
        # We cache freqs_cis but need to take care that it is on the right device
        # and has the right dtype (complex64). The fact that the dtype is different
        # from the module's  dtype means we cannot register it as a buffer
        if self._precomputed_freqs_cis is None:
            self._precomputed_freqs_cis = precompute_freqs_cis(
                self.args.head_dim, 128_000
            )
        if self._precomputed_freqs_cis.device != self.device:
            self._precomputed_freqs_cis = self._precomputed_freqs_cis.to(
                device=self.device
            )
        return self._precomputed_freqs_cis

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ):
        freqs_cis = self.freqs_cis[positions]

        (bsz, seqlen) = input_ids.shape
        num_toks = bsz * seqlen

        if self.pipeline_rank == 0:
            assert self.tok_embeddings is not None
            h = self.tok_embeddings(input_ids)
            assert h.shape == (bsz, seqlen, self.args.dim)
            assert h.dtype == self.dtype
        else:
            h = torch.empty(
                bsz, seqlen, self.args.dim, device=self.device, dtype=self.dtype
            )
            torch.distributed.recv(h, src=self.pipeline_rank - 1)

        mask: Optional[torch.Tensor] = None
        if input_ids.shape[1] > 1:
            tensor = torch.full(
                (seqlen, seqlen),
                dtype=h.dtype,
                fill_value=1,
                device=h.device,
            )
            mask = torch.log(torch.tril(tensor, diagonal=0)).to(h.dtype)

        for layer in self.layers.values():
            h = layer(h, freqs_cis, positions, mask)

        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            torch.distributed.send(h, dst=self.pipeline_rank + 1)
            outs = torch.empty(
                *h.shape[:-1], self.vocab_size, device=h.device, dtype=h.dtype
            )
        else:
            assert self.output is not None
            assert self.norm is not None
            outs = self.output(self.norm(h))
        if self.num_pipeline_ranks > 1:
            torch.distributed.broadcast(outs, src=self.num_pipeline_ranks - 1)
        return outs.float()

    def load_state_dict(self, state_dict, *args, **kwargs):
        state_to_load = {}
        skipped = set([])
        for k, v in state_dict.items():
            if k.startswith("tok_embeddings"):
                if self.pipeline_rank == 0:
                    state_to_load[k] = v
                else:
                    logging.debug(
                        "Skipping parameter %s at pipeline rank %d",
                        k,
                        self.pipeline_rank,
                    )
                    skipped.add(k)
            elif k.startswith("norm") or k.startswith("output"):
                if self.pipeline_rank == self.num_pipeline_ranks - 1:
                    state_to_load[k] = v
                else:
                    logging.debug(
                        "Skipping parameter %s at pipeline rank %d",
                        k,
                        self.pipeline_rank,
                    )
                    skipped.add(k)
            elif k.startswith("layers"):
                layer_id = k.split(".")[1]
                if layer_id in self.layers:
                    state_to_load[k] = v
                else:
                    logging.debug(
                        "Skipping parameter %s at pipeline rank %d",
                        k,
                        self.pipeline_rank,
                    )
                    skipped.add(k)
            else:
                raise ValueError(f"Unexpected key {k}")
        assert set(state_dict.keys()) == skipped.union(set(state_to_load.keys()))
        super().load_state_dict(state_to_load, *args, **kwargs)

    @staticmethod
    def from_folder(
        folder: Path,
        max_batch_size: int,
        max_seq_len: int,
        num_pipeline_ranks: int = 1,
        device="cuda",
        dtype=torch.float16,
    ) -> "Transformer":
        with open(folder / "params.json", "r") as f:
            model_args = ModelArgs.from_dict(json.load(f))
        model_args.max_batch_size = max_batch_size
        model_args.max_seq_len = max_seq_len
        if num_pipeline_ranks > 1:
            pipeline_rank = torch.distributed.get_rank()
        else:
            pipeline_rank = 0
        with torch.device("meta"):
            model = Transformer(
                model_args,
                pipeline_rank=pipeline_rank,
                num_pipeline_ranks=num_pipeline_ranks,
            )
        loaded = torch.load(str(folder / "consolidated.00.pth"), mmap=True)
        model.load_state_dict(loaded, assign=True)
        return model.to(device=device, dtype=dtype)


class Tokenizer:
    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        assert self._model.vocab_size() == self._model.get_piece_size()

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()

    def encode(self, s: str) -> List[int]:
        return [self._model.bos_id(), *self._model.encode(s)]

    def decode(self, t: List[int]) -> str:
        return self._model.decode(t)


@torch.no_grad()
def generate(
    prompts: List[str], model: Transformer, tokenizer: Tokenizer, max_tokens: int
):
    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    prompt_lens = [len(x) for x in encoded_prompts]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)

    input_tokens = torch.full(
        (len(prompts), max_prompt_len),
        tokenizer.pad_id,
        dtype=torch.long,
        device="cuda",
    )
    for i, encoded in enumerate(encoded_prompts):
        input_tokens[i, : len(encoded)] = torch.tensor(encoded).to(input_tokens)
    input_mask = input_tokens != tokenizer.pad_id

    # pre-fill
    positions = torch.arange(0, min_prompt_len).to("cuda")
    logits = model.forward(input_tokens[:, :min_prompt_len], positions)
    logprobs = nn.functional.log_softmax(logits, dim=-1)

    # decode
    generated = []
    all_logprobs = [
        logprobs[:, :-1, :]
        .gather(2, input_tokens[:, 1:min_prompt_len, None])
        .squeeze(-1),
    ]
    for cur_pos in range(min_prompt_len, max_tokens):
        next_token = torch.argmax(logprobs[:, -1, :], dim=-1)
        if cur_pos < input_mask.shape[1]:
            next_token = torch.where(
                input_mask[:, cur_pos], input_tokens[:, cur_pos], next_token
            )
        all_logprobs.append(
            logprobs[:, -1, :].gather(1, next_token[:, None]),
        )
        generated.append(next_token[:, None])
        logits = model.forward(
            next_token[:, None], torch.LongTensor([cur_pos]).to(next_token)
        )
        logprobs = nn.functional.log_softmax(logits, dim=-1)

    all_logprobs_merged = torch.cat(all_logprobs, 1)
    res = []
    if max_tokens > 0:
        generated = torch.cat(generated, 1)
        for i, x in enumerate(encoded_prompts):
            res.append(tokenizer.decode(x[:min_prompt_len] + generated[i].tolist()))
    return res, all_logprobs_merged


def demo(model_path: str, max_tokens: int = 30, num_pipeline_ranks=2):
    if num_pipeline_ranks > 1:
        torch.distributed.init_process_group()
        torch.cuda.set_device(torch.distributed.get_rank())
        should_print = torch.distributed.get_rank() == 0
    else:
        should_print = True

    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(
        Path(model_path),
        max_batch_size=3,
        max_seq_len=max_tokens,
        num_pipeline_ranks=num_pipeline_ranks,
    )

    res, logprobs = generate(
        [
            "This is a test",
            "This is another great test",
            "This is a third test, mistral AI is very good at testing. ",
        ],
        transformer,
        tokenizer,
        max_tokens=max_tokens,
    )
    if should_print:
        for x, l in zip(res, logprobs):
            print(x)
            logging.debug("logprobs: %s", l)
            print("=====================")


if __name__ == "__main__":
    fire.Fire(demo)
