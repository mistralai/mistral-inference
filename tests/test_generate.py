from typing import List

import numpy as np
import torch
from mistral_inference.args import VisionEncoderArgs
from mistral_inference.generate import generate_mamba
from mistral_inference.main import generate
from mistral_inference.mamba import Mamba, MambaArgs
from mistral_inference.transformer import Transformer, TransformerArgs


class DebugTokenizer:
    @property
    def bos_id(self) -> int:
        return 0

    @property
    def eos_id(self) -> int:
        return 1

    @property
    def pad_id(self) -> int:
        return -1

    def encode(self, s: str, bos: bool = True) -> List[int]:
        assert isinstance(s, str)
        t = [int(x) for x in s.split()]
        if bos:
            t = [self.bos_id, *t]
        return t

    def decode(self, t: List[int]) -> str:
        return " ".join([str(x) for x in t])


def test_generation_transformer():
    torch.manual_seed(42)

    sequences = ["1 2 3 4 5 6 7", "0 1 2", "12 13 14", "2 4 34"]
    args = TransformerArgs(
        dim=512,
        n_layers=1,
        head_dim=128,
        hidden_dim=2048,
        n_heads=4,
        n_kv_heads=2,
        norm_eps=1e-5,
        vocab_size=32_000,
        max_batch_size=len(sequences),
    )
    model = Transformer(args).to("cuda", dtype=torch.float32)
    tokenizer = DebugTokenizer()

    encoded = [tokenizer.encode(s, bos=True) for s in sequences]
    toks, all_logprobs_old = generate(encoded, model, temperature=0.0, max_tokens=7)

    # concat generated and prompt
    encoded = [e + t for e, t in zip(encoded, toks)]

    generated, all_logprobs_new = generate(
        encoded, model, temperature=0.0, max_tokens=0
    )

    assert generated == []

    # Verify that logprobs are the same
    assert len(sequences) == len(all_logprobs_old) == len(all_logprobs_new)
    for lp_old, lp_new in zip(all_logprobs_old, all_logprobs_new):
        assert all(
            [abs(x - y) < 5e-4 for x, y in zip(lp_old, lp_new)]
        ), f"\n{lp_old}\n{lp_new}"

    print("All tests passed.")


def test_generation_pixtral():
    torch.manual_seed(42)
    gen = np.random.default_rng(seed=42)

    sequences = ["1 2 2 2 2 4 5 6 7", "12 13 14", "2 2 2 2 7 8 9"]
    images = [[gen.normal(size=(3, 4, 4))], [], [gen.normal(size=(3, 4, 4))]]
    args = TransformerArgs(
        dim=512,
        n_layers=1,
        head_dim=128,
        hidden_dim=2048,
        n_heads=4,
        n_kv_heads=2,
        norm_eps=1e-5,
        vocab_size=32_000,
        max_batch_size=len(sequences),
        vision_encoder=VisionEncoderArgs(
            hidden_size=128,
            num_channels=3,
            image_size=4,
            patch_size=2,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=2,
            rope_theta=10000,
            image_token_id=2,
        ),
    )
    model = Transformer(args).to("cuda", dtype=torch.float32)
    tokenizer = DebugTokenizer()

    encoded = [tokenizer.encode(s, bos=True) for s in sequences]
    toks, all_logprobs_old = generate(
        encoded, model, images=images, temperature=0.0, max_tokens=7
    )

    # concat generated and prompt
    encoded = [e + t for e, t in zip(encoded, toks)]

    generated, all_logprobs_new = generate(
        encoded, model, images=images, temperature=0.0, max_tokens=0
    )

    assert generated == []

    # Verify that logprobs are the same
    assert len(sequences) == len(all_logprobs_old) == len(all_logprobs_new)
    for lp_old, lp_new in zip(all_logprobs_old, all_logprobs_new):
        assert all(
            [abs(x - y) < 5e-4 for x, y in zip(lp_old, lp_new)]
        ), f"\n{lp_old}\n{lp_new}"

    print("All tests passed.")


def test_generation_mamba():
    torch.manual_seed(42)

    sequences = ["1 2 3 4 5 6 7"]
    args = MambaArgs(
        dim=512,
        n_layers=1,
        n_groups=1,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=1,
        tie_embeddings=False,
        vocab_size=32768,
    )
    model = Mamba(args).to("cuda", dtype=torch.float32)
    tokenizer = DebugTokenizer()

    encoded = [tokenizer.encode(s, bos=True) for s in sequences]
    toks, all_logprobs_old = generate_mamba(
        encoded, model, temperature=0.0, max_tokens=7
    )

    assert len(toks[0]) == 7
    assert toks == [[25574, 14821, 11843, 23698, 12735, 23522, 27542]]


def test_chunks_transformer():
    torch.manual_seed(42)

    sequences = [
        " ".join([str(i) for i in range(7)]),
        " ".join([str(i) for i in range(9, 0, -1)]),
    ]
    args = TransformerArgs(
        dim=512,
        n_layers=1,
        head_dim=128,
        hidden_dim=2048,
        n_heads=4,
        n_kv_heads=2,
        norm_eps=1e-5,
        vocab_size=32_000,
        max_batch_size=3,
    )
    model = Transformer(args).to("cuda", dtype=torch.float32)
    tokenizer = DebugTokenizer()

    encoded = [tokenizer.encode(s, bos=True) for s in sequences]
    toks, all_logprobs_old = generate(encoded, model, temperature=0.0, max_tokens=8)

    # concat generated and prompt
    encoded = [e + t for e, t in zip(encoded, toks)]

    generated, all_logprobs_new = generate(
        encoded, model, temperature=0.0, max_tokens=0, chunk_size=5
    )
    assert len(generated) == 0

    for lp_old, lp_new in zip(all_logprobs_old, all_logprobs_new):
        assert all(
            [abs(x - y) < 5e-4 for x, y in zip(lp_old, lp_new)]
        ), f"\n{lp_old}\n{lp_new}"
