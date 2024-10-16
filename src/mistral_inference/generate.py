from typing import List, Optional, Tuple

import numpy as np
import torch

from mistral_inference.cache import BufferCache
from mistral_inference.mamba import Mamba
from mistral_inference.transformer import Transformer


@torch.inference_mode()
def generate_mamba(
    encoded_prompts: List[List[int]],
    model: Mamba,
    *,
    max_tokens: int,
    temperature: float,
    chunk_size: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[float]]]:
    input_ids = torch.tensor(encoded_prompts, device=model.device)
    output = model.model.generate(
        input_ids=input_ids,
        max_length=input_ids.shape[-1] + max_tokens,
        cg=True,
        return_dict_in_generate=True,
        output_scores=True,
        enable_timing=False,
        eos_token_id=eos_id,
        temperature=temperature,
        top_p=0.8,
    )
    generated_tokens = output.sequences[:, input_ids.shape[-1] :].tolist()

    _logprobs: List[List[float]] = [[] for _ in range(len(generated_tokens))]
    for seq_idx, batch_score in enumerate(output.scores):
        for batch_idx, score in enumerate(batch_score.tolist()):
            _logprobs[batch_idx].append(score[generated_tokens[batch_idx][seq_idx]])

    return generated_tokens, _logprobs


@torch.inference_mode()
def generate(
    encoded_prompts: List[List[int]],
    model: Transformer,
    images: List[List[np.ndarray]] = [],
    *,
    max_tokens: int,
    temperature: float,
    chunk_size: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[float]]]:
    images_torch: List[List[torch.Tensor]] = []
    if images:
        assert chunk_size is None
        images_torch = [
            [torch.tensor(im, device=model.device, dtype=model.dtype) for im in images_for_sample]
            for images_for_sample in images
        ]

    model = model.eval()
    B, V = len(encoded_prompts), model.args.vocab_size

    seqlens = [len(x) for x in encoded_prompts]

    # Cache
    cache_window = max(seqlens) + max_tokens
    cache = BufferCache(
        model.n_local_layers,
        model.args.max_batch_size,
        cache_window,
        model.args.n_kv_heads,
        model.args.head_dim,
        model.args.sliding_window,
    )
    cache.to(device=model.device, dtype=model.dtype)
    cache.reset()

    # Bookkeeping
    logprobs: List[List[float]] = [[] for _ in range(B)]
    last_token_prelogits = None

    # One chunk if size not specified
    max_prompt_len = max(seqlens)
    if chunk_size is None:
        chunk_size = max_prompt_len

    flattened_images: List[torch.Tensor] = sum(images_torch, [])

    # Encode prompt by chunks
    for s in range(0, max_prompt_len, chunk_size):
        prompt_chunks = [p[s : s + chunk_size] for p in encoded_prompts]
        assert all(len(p) > 0 for p in prompt_chunks)
        prelogits = model.forward(
            torch.tensor(sum(prompt_chunks, []), device=model.device, dtype=torch.long),
            images=flattened_images,
            seqlens=[len(p) for p in prompt_chunks],
            cache=cache,
        )
        logits = torch.log_softmax(prelogits, dim=-1)

        if last_token_prelogits is not None:
            # Pass > 1
            last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
            for i_seq in range(B):
                logprobs[i_seq].append(last_token_logits[i_seq, prompt_chunks[i_seq][0]].item())

        offset = 0
        for i_seq, sequence in enumerate(prompt_chunks):
            logprobs[i_seq].extend([logits[offset + i, sequence[i + 1]].item() for i in range(len(sequence) - 1)])
            offset += len(sequence)

        last_token_prelogits = prelogits.index_select(
            0,
            torch.tensor([len(p) for p in prompt_chunks], device=prelogits.device).cumsum(dim=0) - 1,
        )
        assert last_token_prelogits.shape == (B, V)

    # decode
    generated_tensors = []
    is_finished = torch.tensor([False for _ in range(B)])

    assert last_token_prelogits is not None
    for _ in range(max_tokens):
        next_token = sample(last_token_prelogits, temperature=temperature, top_p=0.8)

        if eos_id is not None:
            is_finished = is_finished | (next_token == eos_id).cpu()

        if is_finished.all():
            break

        last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
        for i in range(B):
            logprobs[i].append(last_token_logits[i, next_token[i]].item())

        generated_tensors.append(next_token[:, None])
        last_token_prelogits = model.forward(next_token, seqlens=[1] * B, cache=cache)
        assert last_token_prelogits.shape == (B, V)

    generated_tokens: List[List[int]]
    if generated_tensors:
        generated_tokens = torch.cat(generated_tensors, 1).tolist()
    else:
        generated_tokens = []

    return generated_tokens, logprobs


def sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

    return next_token.reshape(-1)


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)
