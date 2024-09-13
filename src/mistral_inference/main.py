import json
import logging
import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Type, Union

import fire  # type: ignore
import torch
import torch.distributed as dist
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ContentChunk,
    ImageChunk,
    ImageURLChunk,
    TextChunk,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.base import Tokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.sentencepiece import is_sentencepiece
from mistral_common.tokens.tokenizers.tekken import (
    SpecialTokenPolicy,
    Tekkenizer,
    is_tekken,
)
from PIL import Image

from mistral_inference.args import TransformerArgs
from mistral_inference.generate import generate, generate_mamba
from mistral_inference.mamba import Mamba
from mistral_inference.transformer import Transformer


def is_torchrun() -> bool:
    required_vars = ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE"]
    return all(var in os.environ for var in required_vars)


def load_tokenizer(model_path: Path) -> MistralTokenizer:
    tokenizer = [f for f in os.listdir(model_path) if is_tekken(model_path / f) or is_sentencepiece(model_path / f)]
    assert (
        len(tokenizer) > 0
    ), f"No tokenizer in {model_path}, place a `tokenizer.model.[v1,v2,v3]` or `tekken.json` file in {model_path}."
    assert (
        len(tokenizer) == 1
    ), f"Multiple tokenizers {', '.join(tokenizer)} found in `model_path`, make sure to only have one tokenizer"

    mistral_tokenizer = MistralTokenizer.from_file(str(model_path / tokenizer[0]))

    if isinstance(mistral_tokenizer.instruct_tokenizer.tokenizer, Tekkenizer):
        mistral_tokenizer.instruct_tokenizer.tokenizer.special_token_policy = SpecialTokenPolicy.KEEP

    logging.info(f"Loaded tokenizer of type {mistral_tokenizer.instruct_tokenizer.__class__}")

    return mistral_tokenizer


def get_model_cls(model_path: str) -> Union[Type[Mamba], Type[Transformer]]:
    with open(Path(model_path) / "params.json", "r") as f:
        args_dict = json.load(f)

    return {"mamba": Mamba, "transformer": Transformer}[args_dict.get("model_type", "transformer")]  # type: ignore[return-value]


def pad_and_convert_to_tensor(list_of_lists: List[List[int]], pad_id: int) -> List[List[int]]:
    # Determine the length of the longest list
    max_len = max(len(lst) for lst in list_of_lists)

    # Left pad each list to the maximum length
    padded_lists = [[pad_id] * (max_len - len(lst)) + lst for lst in list_of_lists]

    return padded_lists


def _get_multimodal_input() -> Tuple[UserMessage, bool]:
    chunks: List[ContentChunk] = []

    response = input("Text prompt: ")
    if response:
        chunks.append(TextChunk(text=response))

    print("[You can input zero, one or more images now.]")
    while True:
        did_something = False
        response = input("Image path or url [Leave empty and press enter to finish image input]: ")
        if response:
            if Path(response).is_file():
                chunks.append(ImageChunk(image=Image.open(response)))
            else:
                assert response.startswith("http"), f"{response} does not seem to be a valid url."
                chunks.append(ImageURLChunk(image_url=response))
            did_something = True

        if not did_something:
            break

    return UserMessage(content=chunks), not chunks


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

    model_cls = get_model_cls(model_path)
    model = model_cls.from_folder(Path(model_path), max_batch_size=3, num_pipeline_ranks=num_pipeline_ranks)
    is_multimodal = isinstance(model.args, TransformerArgs) and model.args.vision_encoder is not None

    if is_multimodal:
        assert instruct, "Multimodal models should only be used in instruct mode"

    # load LoRA
    if lora_path is not None:
        model.load_lora(Path(lora_path))

    prompt: str = ""
    messages: List[UserMessage | AssistantMessage] = []

    while True:
        if should_print:
            if not is_multimodal:
                user_input = input("Prompt: ")

            if instruct:
                if is_multimodal:
                    mm_input, finished = _get_multimodal_input()
                    if finished:
                        break
                    messages += [mm_input]
                else:
                    messages += [UserMessage(content=user_input)]
                chat_completion_request = ChatCompletionRequest(messages=messages)

                tokenized = mistral_tokenizer.encode_chat_completion(chat_completion_request)
                tokens = tokenized.tokens
                images = tokenized.images
            else:
                prompt += user_input

                tokens = tokenizer.encode(prompt, bos=True, eos=False)
                images = []

            length_tensor = torch.tensor([len(tokens)], dtype=torch.int)
        else:
            length_tensor = torch.tensor([0], dtype=torch.int)

        if is_torchrun():
            dist.broadcast(length_tensor, src=0)

        if not should_print:
            tokens = int(length_tensor.item()) * [0]

        generate_fn = generate if isinstance(model, Transformer) else generate_mamba
        generated_tokens, _ = generate_fn(  # type: ignore[operator]
            [tokens],
            model,
            [images],
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

    model_cls = get_model_cls(model_path)
    model = model_cls.from_folder(Path(model_path), max_batch_size=3, num_pipeline_ranks=num_pipeline_ranks)
    # load LoRA
    if lora_path is not None:
        model.load_lora(Path(lora_path))

    mistral_tokenizer: MistralTokenizer = load_tokenizer(Path(model_path))
    tokenizer: Tokenizer = mistral_tokenizer.instruct_tokenizer.tokenizer

    prompts = [
        "This is a test",
        "This is another great test",
        "This is a third test, mistral AI is very good at testing. ",
    ]

    encoded_prompts = [tokenizer.encode(prompt, bos=True, eos=False) for prompt in prompts]

    if isinstance(model, Transformer):
        generate_fn = generate
    else:
        generate_fn = generate_mamba  # type: ignore[assignment]
        warnings.warn(
            "Batched generation is not correctly supported at the moment and therefore might lead to worse results "
            "as compared to non-batched generation. "
            "See https://github.com/state-spaces/mamba/issues/66#issuecomment-1862349718 for more information."
        )
        encoded_prompts = pad_and_convert_to_tensor(encoded_prompts, mistral_tokenizer.instruct_tokenizer.BOS)  # type: ignore[attr-defined]

    generated_tokens, _logprobs = generate_fn(
        encoded_prompts,
        model,  # type: ignore[arg-type]
        max_tokens=max_tokens,
        temperature=temperature,
        eos_id=tokenizer.eos_id,
    )

    generated_words = []
    for i, x in enumerate(generated_tokens):
        generated_words.append(tokenizer.decode(encoded_prompts[i] + x))

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
