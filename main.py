from mistral.cache import RotatingBufferCache
import torch
import fire
from typing import List
from pathlib import Path

from mistral.model import Transformer
from mistral.tokenizer import Tokenizer


def sample_top_p(probs: torch.Tensor, p: float):
    assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


def sample(logits: torch.Tensor, temperature: float, top_p: float):
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

    return next_token.reshape(-1)


@torch.inference_mode()
def generate(prompts: List[str], model: Transformer, tokenizer: Tokenizer, *, max_tokens: int, chunk_size: int = None, temperature: float = 0.7):
    model = model.eval()
    B, V = len(prompts), model.args.vocab_size

    # Tokenize
    encoded_prompts = [tokenizer.encode(prompt, bos=True) for prompt in prompts]
    seqlens = [len(x) for x in encoded_prompts]

    # Cache
    cache_window = min(model.args.sliding_window, max(seqlens) + max_tokens)
    cache = RotatingBufferCache(model.args.n_layers, model.args.max_batch_size, cache_window, model.args.n_kv_heads, model.args.head_dim)
    cache.to(device=model.device, dtype=model.dtype)
    cache.reset()
    
    # Bookkeeping
    logprobs = [[] for _ in range(B)]
    last_token_prelogits = None

    # One chunk if size not specified
    max_prompt_len = max(seqlens)
    if chunk_size is None:
        chunk_size = max_prompt_len

    # Encode prompt by chunks
    for s in range(0, max_prompt_len, chunk_size):
        prompt_chunks = [p[s:s+chunk_size] for p in encoded_prompts]
        assert all(len(p) > 0 for p in prompt_chunks)
        prelogits = model.forward(
            torch.tensor(sum(prompt_chunks, []), device=model.device, dtype=torch.long),
            cache,
            seqlens=[len(p) for p in prompt_chunks]
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

        last_token_prelogits = prelogits.index_select(0, torch.tensor([len(p) for p in prompt_chunks], device=prelogits.device).cumsum(dim=0) - 1)
        assert last_token_prelogits.shape == (B, V)

    # decode
    generated_tokens = []
    for i_token in range(max_tokens):
        next_token = sample(last_token_prelogits, temperature=temperature, top_p=0.8)

        last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
        for i in range(B):
            logprobs[i].append(last_token_logits[i, next_token[i]].item())

        generated_tokens.append(next_token[:, None])
        last_token_prelogits = model.forward(next_token, cache, seqlens=[1] * len(prompts))
        assert last_token_prelogits.shape == (B, V)

    generated_words = []
    if generated_tokens:
        generated_tokens = torch.cat(generated_tokens, 1)
        for i, x in enumerate(encoded_prompts):
            generated_words.append(tokenizer.decode(x + generated_tokens[i].tolist()))

    return generated_words, logprobs


def interactive(model_path: str, max_tokens: int = 35, temperature: float = 0.7):
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(Path(model_path), max_batch_size=3)

    while True:
        prompt = input("Prompt: ")
        res, _logprobs = generate(
            [prompt],
            transformer,
            tokenizer,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        print(res[0])
        print("=====================")

def demo(model_path: str, max_tokens: int = 35, temperature: float = 0):
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
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
        temperature=temperature,
    )
    for x in res:
        print(x)
        print("=====================")

if __name__ == "__main__":
    fire.Fire({
        "interactive": interactive,
        "demo": demo,
    })
