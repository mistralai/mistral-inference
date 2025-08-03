import json
import logging
import os
import time
import tracemalloc
from pathlib import Path
from typing import List, Optional, Type, Union

import fire  # type: ignore
import torch
import torch.distributed as dist
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F

from mistral_common.tokens.tokenizers.base import Tokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.sentencepiece import is_sentencepiece
from mistral_common.tokens.tokenizers.tekken import is_tekken
from mistral_inference.args import TransformerArgs
from mistral_inference.generate import generate, generate_mamba
from mistral_inference.mamba import Mamba
from mistral_inference.transformer import Transformer

def is_torchrun() -> bool:
    return all(var in os.environ for var in ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE"])

def load_tokenizer(model_path: Path) -> MistralTokenizer:
    tokenizer_files = [f for f in os.listdir(model_path) if is_tekken(model_path / f) or is_sentencepiece(model_path / f)]
    assert tokenizer_files, f"No tokenizer found in {model_path}"
    assert len(tokenizer_files) == 1, f"Multiple tokenizers found in {model_path}: {tokenizer_files}"

    tokenizer = MistralTokenizer.from_file(str(model_path / tokenizer_files[0]))
    return tokenizer

def get_model_cls(model_path: str) -> Union[Type[Mamba], Type[Transformer]]:
    with open(Path(model_path) / "params.json") as f:
        args_dict = json.load(f)
    return {"mamba": Mamba, "transformer": Transformer}[args_dict.get("model_type", "transformer")]

def pad_and_convert_to_tensor(list_of_lists: List[List[int]], pad_id: int) -> np.ndarray:
    max_len = max(len(lst) for lst in list_of_lists)
    # Align to warp boundary (32)
    max_len_aligned = ((max_len + 31) // 32) * 32
    padded = np.full((len(list_of_lists), max_len_aligned), pad_id, dtype=np.int32)
    for i, row in enumerate(list_of_lists):
        padded[i, -len(row):] = row
    return padded

def run_benchmark(model_path: str, prompts: List[str], max_tokens: int, temperature: float, lora_path: Optional[str]) -> dict:
    torchrun = is_torchrun()
    if torchrun:
        dist.init_process_group()
        local_rank = int(os.getenv("LOCAL_RANK", dist.get_rank()))
        torch.cuda.set_device(local_rank)
        should_print = dist.get_rank() == 0
        num_pipeline_ranks = dist.get_world_size()
    else:
        should_print = True
        num_pipeline_ranks = 1

    tokenizer = load_tokenizer(Path(model_path))
    tokenizer_core = tokenizer.instruct_tokenizer.tokenizer
    model_cls = get_model_cls(model_path)
    model = model_cls.from_folder(Path(model_path), max_batch_size=16, num_pipeline_ranks=num_pipeline_ranks)
    if lora_path:
        model.load_lora(Path(lora_path))

    if torch.cuda.is_available():
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    generate_fn = generate if isinstance(model, Transformer) else generate_mamba
    encoded_prompts = [tokenizer_core.encode(p, bos=True, eos=False) for p in prompts]
    if not isinstance(model, Transformer):
        encoded_prompts = pad_and_convert_to_tensor(encoded_prompts, tokenizer.instruct_tokenizer.BOS)  # type: ignore[attr-defined]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    tracemalloc.start()
    start_mem = tracemalloc.get_traced_memory()[0]
    start_time = time.time()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
        with record_function("generate_and_decode"):
            input_tensor = torch.tensor(encoded_prompts, dtype=torch.long, pin_memory=True)
            input_tensor = input_tensor.to(model.device, non_blocking=True)
            generated_tokens, _ = generate_fn(
                input_tensor,
                model,
                max_tokens=max_tokens,
                temperature=temperature,
                eos_id=tokenizer_core.eos_id
            )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    end_mem = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    duration = end_time - start_time
    total_tokens = sum(len(g) for g in generated_tokens)
    throughput = total_tokens / duration if duration > 0 else 0

    if should_print:
        print("\n=== Profiler Output ===")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    return {
        "model_path": model_path,
        "tokens_generated": total_tokens,
        "duration_sec": duration,
        "throughput": throughput,
        "memory_MB": (end_mem - start_mem)/1024/1024
    }

def compare_versions(before_model_path: str, after_model_path: str, max_tokens: int = 50, temperature: float = 0.7, lora_path: Optional[str] = None) -> None:
    prompts = [
        "Benchmarking prompt one.",
        "Benchmarking prompt two with more content to test token efficiency.",
        "Final benchmarking prompt for performance harness testing."
    ]
    print("\nRunning benchmark on BEFORE version...")
    before = run_benchmark(before_model_path, prompts, max_tokens, temperature, lora_path)

    print("\nRunning benchmark on AFTER version...")
    after = run_benchmark(after_model_path, prompts, max_tokens, temperature, lora_path)

    def fmt(label, val):
        return f"{label:<22}: {val:.2f}"

    print("\n=== Comparison Report ===")
    print(fmt("Tokens Generated (Before)", before["tokens_generated"]))
    print(fmt("Tokens Generated (After) ", after["tokens_generated"]))
    print(fmt("Duration (Before, s) ", before["duration_sec"]))
    print(fmt("Duration (After, s)  ", after["duration_sec"]))
    print(fmt("Throughput Before", before["throughput"]))
    print(fmt("Throughput After ", after["throughput"]))
    print(fmt("Peak Memory (Before, MB)", before["memory_MB"]))
    print(fmt("Peak Memory (After, MB) ", after["memory_MB"]))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire({
        "compare_versions": compare_versions
    })
