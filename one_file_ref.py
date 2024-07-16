import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import fire
import safetensors.torch
import torch
from mistral_common.tokens.tokenizers.base import Tokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from simple_parsing.helpers import Serializable
from torch import nn


@dataclass
class TransformerArgs(Serializable):
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int

    # For rotary embeddings. If not set, will be inferred
    rope_theta: Optional[float] = None

    max_seq_len: int = 16384
    max_batch_size: int = 0


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int) -> Tuple[torch.Tensor]:
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
    def __init__(self, args: TransformerArgs):
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
        self.cache_k = torch.empty(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_kv_heads,
                self.args.head_dim,
            ),
            dtype=torch.float16,
        ).cuda()
        self.cache_v = torch.empty(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_kv_heads,
                self.args.head_dim,
            ),
            dtype=torch.float16,
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.args.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.args.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.args.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # cache
        scatter_pos = positions[None, :, None, None].repeat(bsz, 1, self.n_kv_heads, self.args.head_dim)
        self.cache_k[:bsz].scatter_(dim=1, index=scatter_pos, src=xk)
        self.cache_v[:bsz].scatter_(dim=1, index=scatter_pos, src=xv)

        if positions.shape[0] > 1:
            # prefill
            key, value = repeat_kv(xk, xv, self.repeats)
        else:
            cur_pos = positions[-1].item() + 1
            key, value = repeat_kv(
                self.cache_k[:bsz, :cur_pos, ...],
                self.cache_v[:bsz, :cur_pos, ...],
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
    def __init__(self, args: TransformerArgs):
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


class TransformerBlock(nn.Module):
    def __init__(self, args: TransformerArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args=args)
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


def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


class Transformer(nn.Module):
    def __init__(self, args: TransformerArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        assert self.vocab_size > 0

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = torch.nn.ModuleList([TransformerBlock(args=args) for _ in range(args.n_layers)])

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        theta = self.args.rope_theta or 1000000.0
        self.freqs_cis = precompute_freqs_cis(self.args.head_dim, 128_000, theta).to("cuda")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ):
        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[positions]

        mask: Optional[torch.Tensor] = None
        if input_ids.shape[1] > 1:
            seqlen = input_ids.shape[1]
            tensor = torch.full(
                (seqlen, seqlen),
                dtype=h.dtype,
                fill_value=1,
                device=h.device,
            )
            mask = torch.tril(tensor, diagonal=0).to(h.dtype)
            mask = torch.triu(mask, diagonal=-self.args.max_seq_len)
            mask = torch.log(mask)

        for layer in self.layers:
            h = layer(h, freqs_cis, positions, mask)

        return self.output(self.norm(h)).float()

    @staticmethod
    def from_folder(folder: Path, max_batch_size: int = 1, device="cuda", dtype=torch.float16):
        with open(Path(folder) / "params.json", "r") as f:
            model_args = TransformerArgs.from_dict(json.load(f))
        model_args.max_batch_size = max_batch_size

        model = Transformer(model_args)

        pt_model_file = Path(folder) / "consolidated.00.pth"
        safetensors_model_file = Path(folder) / "consolidated.safetensors"

        assert (
            pt_model_file.exists() or safetensors_model_file.exists()
        ), f"Make sure either {pt_model_file} or {safetensors_model_file} exists"
        assert not (
            pt_model_file.exists() and safetensors_model_file.exists()
        ), f"Both {pt_model_file} and {safetensors_model_file} cannot exist"

        if pt_model_file.exists():
            loaded = torch.load(str(pt_model_file), mmap=True)
        else:
            loaded = safetensors.torch.load_file(str(safetensors_model_file))

        model.load_state_dict(loaded, assign=True, strict=True)
        return model.to(device=device, dtype=dtype)


def load_tokenizer(model_path: Path) -> MistralTokenizer:
    tokenizer = [f for f in os.listdir(Path(model_path)) if f.startswith("tokenizer.model")]
    assert (
        len(tokenizer) > 0
    ), f"No tokenizer found in {model_path}, make sure to place a `tokenizer.model.[v1,v2,v3]` file in {model_path}."
    assert (
        len(tokenizer) == 1
    ), f"Multiple tokenizers {', '.join(tokenizer)} found in `model_path`, make sure to only have one tokenizer"

    tokenizer = MistralTokenizer.from_file(str(model_path / tokenizer[0]))

    return tokenizer


@torch.no_grad()
def generate(prompts: List[str], model: Transformer, tokenizer: Tokenizer, max_tokens: int):
    encoded_prompts = [tokenizer.encode(prompt, bos=True, eos=False) for prompt in prompts]
    prompt_lens = [len(x) for x in encoded_prompts]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)

    input_tokens = torch.full(
        (len(prompts), max_prompt_len),
        tokenizer._model.pad_id(),
        dtype=torch.long,
        device="cuda",
    )
    for i, encoded in enumerate(encoded_prompts):
        input_tokens[i, : len(encoded)] = torch.tensor(encoded).to(input_tokens)

    input_mask = input_tokens != tokenizer._model.pad_id()

    # pre-fill
    positions = torch.arange(0, min_prompt_len).to("cuda")
    logits = model.forward(input_tokens[:, :min_prompt_len], positions)
    logprobs = nn.functional.log_softmax(logits, dim=-1)

    # decode
    generated = []
    all_logprobs = [
        logprobs[:, :-1, :].gather(2, input_tokens[:, 1:min_prompt_len, None]).squeeze(-1),
    ]
    cur_pos = min_prompt_len
    for _ in range(max_tokens):
        next_token = torch.argmax(logprobs[:, -1, :], dim=-1)
        if cur_pos < input_mask.shape[1]:
            next_token = torch.where(input_mask[:, cur_pos], input_tokens[:, cur_pos], next_token)
        all_logprobs.append(
            logprobs[:, -1, :].gather(1, next_token[:, None]),
        )
        generated.append(next_token[:, None])
        logits = model.forward(next_token[:, None], torch.LongTensor([cur_pos]).to(next_token))
        logprobs = nn.functional.log_softmax(logits, dim=-1)
        cur_pos += 1

    all_logprobs = torch.cat(all_logprobs, 1)
    res = []
    if max_tokens > 0:
        generated = torch.cat(generated, 1)

        for i, x in enumerate(encoded_prompts):
            res.append(tokenizer.decode(x[:min_prompt_len] + generated[i].tolist()))
    return res, all_logprobs


def demo(model_path: str, max_tokens: int = 35):
    mistral_tokenizer: MistralTokenizer = load_tokenizer(Path(model_path))
    tokenizer: Tokenizer = mistral_tokenizer.instruct_tokenizer.tokenizer

    transformer = Transformer.from_folder(Path(model_path), max_batch_size=3)

    res, _logprobs = generate(
        [
            "This is a test",
            "This is another test",
            "This is a third test, mistral AI is very good at testing. ",
        ],
        transformer,
        tokenizer,
        max_tokens=max_tokens,
    )
    for x in res:
        print(x)
        print("=====================")


if __name__ == "__main__":
    fire.Fire(demo)
