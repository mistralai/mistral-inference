import logging
import os
from pathlib import Path
from typing import List, Optional

import fire  # type: ignore
import torch
import torch.distributed as dist
from mistral_common.protocol.instruct.messages import AssistantMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.base import Tokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from mistral_inference.generate import generate
from mistral_inference.model import Transformer


def is_torchrun() -> bool:
    required_vars = ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE"]
    return all(var in os.environ for var in required_vars)


def load_tokenizer(model_path: Path) -> MistralTokenizer:
    tokenizer = [
        f for f in os.listdir(Path(model_path)) if f.startswith("tokenizer.model")
    ]
    assert (
        len(tokenizer) > 0
    ), f"No tokenizer found in {model_path}, make sure to place a `tokenizer.model.[v1,v2,v3]` file in {model_path}."
    assert (
        len(tokenizer) == 1
    ), f"Multiple tokenizers {', '.join(tokenizer)} found in `model_path`, make sure to only have one tokenizer"

    mistral_tokenizer = MistralTokenizer.from_file(str(model_path / tokenizer[0]))

    logging.info(
        f"Loaded tokenizer of type {mistral_tokenizer.instruct_tokenizer.__class__}"
    )

    return mistral_tokenizer


def interactive(
    model_path: str,
    max_tokens: int = 35,
    temperature: float = 0.7,
    num_pipeline_ranks: int = 1,
    instruct: bool = False,
    lora_path: Optional[str] = None,
) -> None:
    if is_torchrun():
        torch.distributed.init_process_group()
        torch.cuda.set_device(torch.distributed.get_rank())
        should_print = torch.distributed.get_rank() == 0

        num_pipeline_ranks = torch.distributed.get_world_size()
    else:
        should_print = True
        num_pipeline_ranks = 1

    mistral_tokenizer: MistralTokenizer = load_tokenizer(Path(model_path))
    tokenizer: Tokenizer = mistral_tokenizer.instruct_tokenizer.tokenizer

    transformer = Transformer.from_folder(
        Path(model_path), max_batch_size=3, num_pipeline_ranks=num_pipeline_ranks
    )

    # load LoRA
    if lora_path is not None:
        transformer.load_lora(Path(lora_path))

    prompt: str = ""
    messages: List[UserMessage | AssistantMessage] = []

    while True:
        if should_print:
            user_input = input("Prompt: ")

            if instruct:
                messages += [UserMessage(content=user_input)]
                chat_completion_request = ChatCompletionRequest(messages=messages)

                tokens = mistral_tokenizer.encode_chat_completion(
                    chat_completion_request
                ).tokens
            else:
                prompt += user_input

                tokens = tokenizer.encode(prompt, bos=True, eos=False)

            length_tensor = torch.tensor([len(tokens)], dtype=torch.int)
        else:
            length_tensor = torch.tensor([0], dtype=torch.int)

        if is_torchrun():
            dist.broadcast(length_tensor, src=0)

        if not should_print:
            tokens = int(length_tensor.item()) * [0]

        generated_tokens, _ = generate(
            [tokens],
            transformer,
            max_tokens=max_tokens,
            temperature=temperature,
            eos_id=tokenizer.eos_id,
        )

        answer = tokenizer.decode(generated_tokens[0])

        if should_print:
            print(answer)
            print("=====================")

        if instruct:
            messages += [AssistantMessage(content=answer)]
        else:
            prompt += answer


def demo(
    model_path: str,
    max_tokens: int = 35,
    temperature: float = 0,
    lora_path: Optional[str] = None,
) -> None:
    if is_torchrun():
        torch.distributed.init_process_group()
        torch.cuda.set_device(torch.distributed.get_rank())
        should_print = torch.distributed.get_rank() == 0

        num_pipeline_ranks = torch.distributed.get_world_size()
    else:
        should_print = True
        num_pipeline_ranks = 1

    transformer = Transformer.from_folder(
        Path(model_path), max_batch_size=3, num_pipeline_ranks=num_pipeline_ranks
    )
    # load LoRA
    if lora_path is not None:
        transformer.load_lora(Path(lora_path))

    mistral_tokenizer: MistralTokenizer = load_tokenizer(Path(model_path))
    tokenizer: Tokenizer = mistral_tokenizer.instruct_tokenizer.tokenizer

    prompts = [
        "This is a test",
        "This is another great test",
        "This is a third test, mistral AI is very good at testing. ",
    ]

    encoded_prompts = [
        tokenizer.encode(prompt, bos=True, eos=False) for prompt in prompts
    ]

    generated_tokens, _logprobs = generate(
        encoded_prompts,
        transformer,
        max_tokens=max_tokens,
        temperature=temperature,
        eos_id=tokenizer.eos_id,
    )

    generated_words = []
    for i, x in enumerate(encoded_prompts):
        generated_words.append(tokenizer.decode(x + generated_tokens[i]))

    res = generated_words

    if should_print:
        for w, logprob in zip(res, _logprobs):
            print(w)
            logging.debug("Logprobs: %s", logprob)
            print("=====================")


def mistral_chat() -> None:
    fire.Fire(interactive)


def mistral_demo() -> None:
    fire.Fire(demo)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(
        {
            "interactive": interactive,
            "demo": demo,
        }
    )
