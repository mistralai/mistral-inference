# Mistral Inference
<a target="_blank" href="https://colab.research.google.com/github/mistralai/mistral-inference/blob/main/tutorials/getting_started.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


This repository contains minimal code to run Mistral models.

Blog 7B: [https://mistral.ai/news/announcing-mistral-7b/](https://mistral.ai/news/announcing-mistral-7b/)\
Blog 8x7B: [https://mistral.ai/news/mixtral-of-experts/](https://mistral.ai/news/mixtral-of-experts/)\
Blog 8x22B: [https://mistral.ai/news/mixtral-8x22b/](https://mistral.ai/news/mixtral-8x22b/)\
Blog Codestral 22B: [https://mistral.ai/news/codestral](https://mistral.ai/news/codestral/) \
Blog Codestral Mamba 7B: [https://mistral.ai/news/codestral-mamba/](https://mistral.ai/news/codestral-mamba/) \
Blog Mathstral 7B: [https://mistral.ai/news/mathstral/](https://mistral.ai/news/mathstral/) \
Blog Nemo: [https://mistral.ai/news/mistral-nemo/](https://mistral.ai/news/mistral-nemo/) \
Blog Mistral Large 2: [https://mistral.ai/news/mistral-large-2407/](https://mistral.ai/news/mistral-large-2407/) \
Blog Pixtral 12B: [https://mistral.ai/news/pixtral-12b/](https://mistral.ai/news/pixtral-12b/)

Discord: [https://discord.com/invite/mistralai](https://discord.com/invite/mistralai)\
Documentation: [https://docs.mistral.ai/](https://docs.mistral.ai/)\
Guardrailing: [https://docs.mistral.ai/usage/guardrailing](https://docs.mistral.ai/usage/guardrailing)

## Installation

Note: You will use a GPU to install `mistral-inference`, as it currently requires `xformers` to be installed and `xformers` itself needs a GPU for installation.

### PyPI

```
pip install mistral-inference
```

### Local

```
cd $HOME && git clone https://github.com/mistralai/mistral-inference
cd $HOME/mistral-inference && poetry install .
```

## Model download

| Name        | Download | md5sum |
|-------------|-------|-------|
| 7B Instruct | https://models.mistralcdn.com/mistral-7b-v0-3/mistral-7B-Instruct-v0.3.tar | `80b71fcb6416085bcb4efad86dfb4d52` |
| 8x7B Instruct | https://models.mistralcdn.com/mixtral-8x7b-v0-1/Mixtral-8x7B-v0.1-Instruct.tar (**Updated model coming soon!**) | `8e2d3930145dc43d3084396f49d38a3f` |
| 8x22 Instruct | https://models.mistralcdn.com/mixtral-8x22b-v0-3/mixtral-8x22B-Instruct-v0.3.tar | `471a02a6902706a2f1e44a693813855b` |
| 7B Base | https://models.mistralcdn.com/mistral-7b-v0-3/mistral-7B-v0.3.tar | `0663b293810d7571dad25dae2f2a5806` |
| 8x7B |     **Updated model coming soon!**       | - |
| 8x22B | https://models.mistralcdn.com/mixtral-8x22b-v0-3/mixtral-8x22B-v0.3.tar | `a2fa75117174f87d1197e3a4eb50371a` |
| Codestral 22B | https://models.mistralcdn.com/codestral-22b-v0-1/codestral-22B-v0.1.tar | `1ea95d474a1d374b1d1b20a8e0159de3` |
| Mathstral 7B | https://models.mistralcdn.com/mathstral-7b-v0-1/mathstral-7B-v0.1.tar | `5f05443e94489c261462794b1016f10b` |
| Codestral-Mamba 7B | https://models.mistralcdn.com/codestral-mamba-7b-v0-1/codestral-mamba-7B-v0.1.tar | `d3993e4024d1395910c55db0d11db163` |
| Nemo Base | https://models.mistralcdn.com/mistral-nemo-2407/mistral-nemo-base-2407.tar | `c5d079ac4b55fc1ae35f51f0a3c0eb83` |
| Nemo Instruct | https://models.mistralcdn.com/mistral-nemo-2407/mistral-nemo-instruct-2407.tar | `296fbdf911cb88e6f0be74cd04827fe7` |
| Mistral Large 2 | https://models.mistralcdn.com/mistral-large-2407/mistral-large-instruct-2407.tar | `fc602155f9e39151fba81fcaab2fa7c4` |

Note: 
- **Important**:
  - `mixtral-8x22B-Instruct-v0.3.tar` is exactly the same as [Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1), only stored in `.safetensors` format
  - `mixtral-8x22B-v0.3.tar` is the same as [Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1), but has an extended vocabulary of 32768 tokens.
  - `codestral-22B-v0.1.tar` has a custom non-commercial license, called [Mistral AI Non-Production (MNPL) License](https://mistral.ai/licenses/MNPL-0.1.md)
  - `mistral-large-instruct-2407.tar` has a custom non-commercial license, called [Mistral AI Research (MRL) License](https://mistral.ai/licenses/MRL-0.1.md)
- All of the listed models above support function calling. For example, Mistral 7B Base/Instruct v3 is a minor update to Mistral 7B Base/Instruct v2,  with the addition of function calling capabilities. 
- The "coming soon" models will include function calling as well. 
- You can download the previous versions of our models from our [docs](https://docs.mistral.ai/getting-started/open_weight_models/#downloading).

### Usage

**News!!!**: Mistral Large 2 is out. Read more about its capabilities [here](https://mistral.ai/news/mistral-large-2407/).

Create a local folder to store models
```sh
export MISTRAL_MODEL=$HOME/mistral_models
mkdir -p $MISTRAL_MODEL
```

Download any of the above links and extract the content, *e.g.*:

```sh
export 12B_DIR=$MISTRAL_MODEL/12B_Nemo
wget https://models.mistralcdn.com/mistral-nemo-2407/mistral-nemo-instruct-2407.tar
mkdir -p $12B_DIR
tar -xf mistral-nemo-instruct-2407.tar -C $12B_DIR
```

or 

```sh
export M8x7B_DIR=$MISTRAL_MODEL/8x7b_instruct
wget https://models.mistralcdn.com/mixtral-8x7b-v0-1/Mixtral-8x7B-v0.1-Instruct.tar
mkdir -p $M8x7B_DIR
tar -xf Mixtral-8x7B-v0.1-Instruct.tar -C $M8x7B_DIR
```

## Usage

The following sections give an overview of how to run the model from the Command-line interface (CLI) or directly within Python.

### CLI

- **Demo**

To test that a model works in your setup, you can run the `mistral-demo` command.
*E.g.* the 12B Mistral-Nemo model can be tested on a single GPU as follows:

```sh
mistral-demo $12B_DIR
```

Large models, such **8x7B** and **8x22B** have to be run in a multi-GPU setup.
For these models, you can use the following command:

```sh
torchrun --nproc-per-node 2 --no-python mistral-demo $M8x7B_DIR
```

*Note*: Change `--nproc-per-node` to more GPUs if available.

- **Chat**

To interactively chat with the models, you can make use of the `mistral-chat` command.

```sh
mistral-chat $12B_DIR --instruct --max_tokens 1024 --temperature 0.35
```

For large models, you can make use of `torchrun`.

```sh
torchrun --nproc-per-node 2 --no-python mistral-chat $M8x7B_DIR --instruct
```

*Note*: Change `--nproc-per-node` to more GPUs if necessary (*e.g.* for 8x22B).

- **Chat with Codestral**

To use [Codestral](https://mistral.ai/news/codestral/) as a coding assistant you can run the following command using `mistral-chat`.
Make sure `$M22B_CODESTRAL` is set to a valid path to the downloaded codestral folder, e.g. `$HOME/mistral_models/Codestral-22B-v0.1`

```sh
mistral-chat $M22B_CODESTRAL --instruct --max_tokens 256
```

If you prompt it with *"Write me a function that computes fibonacci in Rust"*, the model should generate something along the following lines:

```sh
Sure, here's a simple implementation of a function that computes the Fibonacci sequence in Rust. This function takes an integer `n` as an argument and returns the `n`th Fibonacci number.

fn fibonacci(n: u32) -> u32 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

fn main() {
    let n = 10;
    println!("The {}th Fibonacci number is: {}", n, fibonacci(n));
}

This function uses recursion to calculate the Fibonacci number. However, it's not the most efficient solution because it performs a lot of redundant calculations. A more efficient solution would use a loop to iteratively calculate the Fibonacci numbers.
```

You can continue chatting afterwards, *e.g.* with *"Translate it to Python"*.

- **Chat with Codestral-Mamba**

To use [Codestral-Mamba](https://mistral.ai/news/codestral-mamba/) as a coding assistant you can run the following command using `mistral-chat`.
Make sure `$7B_CODESTRAL_MAMBA` is set to a valid path to the downloaded codestral-mamba folder, e.g. `$HOME/mistral_models/mamba-codestral-7B-v0.1`.

You then need to additionally install the following packages:
  
```
pip install packaging mamba-ssm causal-conv1d transformers
```

before you can start chatting:

```sh
mistral-chat $7B_CODESTRAL_MAMBA --instruct --max_tokens 256
```

- **Chat with Mathstral**

To use [Mathstral](https://mistral.ai/news/mathstral/) as an assistant you can run the following command using `mistral-chat`.
Make sure `$7B_MATHSTRAL` is set to a valid path to the downloaded codestral folder, e.g. `$HOME/mistral_models/mathstral-7B-v0.1`

```sh
mistral-chat $7B_MATHSTRAL --instruct --max_tokens 256
```

If you prompt it with *"Albert likes to surf every week. Each surfing session lasts for 4 hours and costs $20 per hour. How much would Albert spend in 5 weeks?"*, the model should answer with the correct calculation.

You can then continue chatting afterwards, *e.g.* with *"How much would he spend in a year?"*.

### Python

- *Instruction Following*:

```py
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


tokenizer = MistralTokenizer.from_file("./mistral-nemo-instruct-v0.1/tekken.json")  # change to extracted tokenizer file
model = Transformer.from_folder("./mistral-nemo-instruct-v0.1")  # change to extracted model dir

prompt = "How expensive would it be to ask a window cleaner to clean all windows in Paris. Make a reasonable guess in US Dollar."

completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])

tokens = tokenizer.encode_chat_completion(completion_request).tokens

out_tokens, _ = generate([tokens], model, max_tokens=1024, temperature=0.35, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

print(result)
```

- *Function Calling*:

```py
from mistral_common.protocol.instruct.tool_calls import Function, Tool

completion_request = ChatCompletionRequest(
    tools=[
        Tool(
            function=Function(
                name="get_current_weather",
                description="Get the current weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            )
        )
    ],
    messages=[
        UserMessage(content="What's the weather like today in Paris?"),
        ],
)

tokens = tokenizer.encode_chat_completion(completion_request).tokens

out_tokens, _ = generate([tokens], model, max_tokens=64, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

print(result)
```

- *Fill-in-the-middle (FIM)*:

Make sure to have `mistral-common >= 1.2.0` installed:
```
pip install --upgrade mistral-common
```

You can simulate a code completion in-filling as follows.

```py
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.instruct.request import FIMRequest

tokenizer = MistralTokenizer.from_model("codestral-22b")
model = Transformer.from_folder("./mistral_22b_codestral")

prefix = """def add("""
suffix = """    return sum"""

request = FIMRequest(prompt=prefix, suffix=suffix)

tokens = tokenizer.encode_fim(request).tokens

out_tokens, _ = generate([tokens], model, max_tokens=256, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
result = tokenizer.decode(out_tokens[0])

middle = result.split(suffix)[0].strip()
print(middle)
```

### One-file-ref

If you want a self-contained implementation, look at `one_file_ref.py`, or run it with 

```
python -m one_file_ref $M7B_DIR
```

which should give something along the following lines:

```
This is a test of the emergency broadcast system. This is only a test.

If this were a real emergency, you would be told what to do.

This is a test
=====================
This is another test of the new blogging software. I’m not sure if I’m going to keep it or not. I’m not sure if I’m going to keep
=====================
This is a third test, mistral AI is very good at testing. 🙂

This is a third test, mistral AI is very good at testing. 🙂

This
=====================
```

**Note**: To run self-contained implementations, you need to do a local installation.

### Test

To run logits equivalence:
```
python -m pytest tests
```

## Deployment

The `deploy` folder contains code to build a [vLLM](https://M7B_DIR.com/vllm-project/vllm) image with the required dependencies to serve the Mistral AI model. In the image, the [transformers](https://github.com/huggingface/transformers/) library is used instead of the reference implementation. To build it:

```bash
docker build deploy --build-arg MAX_JOBS=8
```

Instructions to run the image can be found in the [official documentation](https://docs.mistral.ai/quickstart).


## Model platforms

- Use Mistral models on [Mistral AI official API](https://console.mistral.ai/) (La Plateforme)
- Use Mistral models via [cloud providers](https://docs.mistral.ai/deployment/cloud/overview/)

## References

[1]: [LoRA](https://arxiv.org/abs/2106.09685): Low-Rank Adaptation of Large Language Models, Hu et al. 2021
