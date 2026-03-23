from dataclasses import dataclass
from typing import List, Optional

import torch

import mistral_inference.generate as generate_module


@dataclass
class DummyArgs:
    vocab_size: int = 16
    max_batch_size: int = 2
    n_kv_heads: int = 1
    head_dim: int = 4
    sliding_window: Optional[int] = None


class DummyTransformer:
    def __init__(self) -> None:
        self.args = DummyArgs()
        self.n_local_layers = 1
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.forward_inputs: List[torch.Tensor] = []
        self.forward_seqlens: List[List[int]] = []

    def eval(self) -> "DummyTransformer":
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        cache=None,
        images=None,
    ) -> torch.Tensor:
        self.forward_inputs.append(input_ids.detach().clone())
        self.forward_seqlens.append(list(seqlens))
        return torch.zeros((sum(seqlens), self.args.vocab_size), dtype=torch.float32, device=self.device)


def test_generate_stops_each_row_after_its_own_eos(monkeypatch) -> None:
    # Row 0 emits EOS at step 1, row 1 emits EOS at step 3.
    planned_tokens = [
        torch.tensor([1, 2], dtype=torch.long),
        torch.tensor([7, 3], dtype=torch.long),
        torch.tensor([4, 1], dtype=torch.long),
    ]

    def fake_sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        assert planned_tokens
        return planned_tokens.pop(0).to(device=logits.device)

    monkeypatch.setattr(generate_module, "sample", fake_sample)

    model = DummyTransformer()
    generated, logprobs = generate_module.generate(
        [[10], [20]],
        model,
        max_tokens=4,
        temperature=0.0,
        eos_id=1,
    )

    # Expected behavior: row 0 stops right after EOS; row 1 continues until EOS.
    assert generated == [[1], [2, 3, 1]]
    assert [len(lp) for lp in logprobs] == [1, 3]

    # Prompt prefill + 2 decode forwards (none after all rows finished).
    assert len(model.forward_inputs) == 3
    assert model.forward_inputs[1].tolist() == [1, 2]
    # Row 0 should stay frozen after EOS, not consume token 7.
    assert model.forward_inputs[2].tolist() == [1, 3]
    assert model.forward_seqlens[1:] == [[1, 1], [1, 1]]


def test_generate_keeps_final_eos_when_all_rows_finish_same_step(monkeypatch) -> None:
    planned_tokens = [torch.tensor([1, 1], dtype=torch.long)]

    def fake_sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        assert planned_tokens
        return planned_tokens.pop(0).to(device=logits.device)

    monkeypatch.setattr(generate_module, "sample", fake_sample)

    model = DummyTransformer()
    generated, logprobs = generate_module.generate(
        [[10], [20]],
        model,
        max_tokens=3,
        temperature=0.0,
        eos_id=1,
    )

    # Expected behavior: both rows return their terminating EOS token.
    assert generated == [[1], [1]]
    assert [len(lp) for lp in logprobs] == [1, 1]

    # No decode forward call after first sampled EOS because batch is done.
    assert len(model.forward_inputs) == 1
