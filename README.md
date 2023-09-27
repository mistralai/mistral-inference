# Mistral Transformer

This repository contains minimal code to run our 7B model.
[Discord](https://discord.com/invite/mistralai)

## Installation

```
pip install -r requirements.txt
```

## Run the model

```
python -m main demo /path/to/model 
# To give your own prompts
python -m main interactive /path/to/model 
```

If you want a self-contained implementation, look at `one_file_ref.py`, or run it with 
```
python -m one_file_ref /path/to/model 

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

To run logits equivalence through chunking and sliding window, launch
```
python -m test_generate
```


# Sliding window attention

## Vanilla attention

Attention is how information is shared between tokens in a sequence.
In vanilla transformers, attention follows a causal mask: each token in the sequence can attend to itself and all the tokens in the past.
This ensures that the model is causal, i.e. it can only use information from the past to predict the future.


![Causal attention mask](assets/full_attention.png)

## Sliding window to speed-up inference and reduce memory pressure

The number of operations of attention is quadratic in the sequence length, and the memory pressure is linear in the sequence length.
At inference time, this incurs higher latency and smaller throughput due to reduced cache availability.
To alleviate this issue, we use a sliding window attention [1,2]: each token can attend to at most W tokens in the past (here, W=3).

![Sliding window attention](assets/sliding_attention.png)

Note that tokens outside the sliding window still influence next word prediction. 
At each attention layer, information can move forward by W tokens at most: after two attention layers, information can move forward by 2W tokens, etc.
For instance in a sequence of length 16K and a sliding window of 4K, after 4 layers, information has propagated to the full sequence length.

![Attention through layers](assets/attention_through_layers.png)

Empirically, we see that longer contexts do help *even outside the sliding window* but when the sequence length becomes too large, the model does not use the full context anymore.

## Rolling buffer cache

We implement a rolling buffer cache.
The cache has a fixed size of W, and we store the (key, value) for position i in cache position i % W.
When the position i is larger than W, past values in the cache are overwritten.

![Rolling cache](assets/rolling_cache.png)

## Pre-fill and chunking

When generating a sequence, we need to predict tokens one-by-one, as each token is conditioned on the previous ones.
However, the prompt is known in advance, and we can pre-fill the (k, v) cache with the prompt.
If the prompt is very large, we can chunk it into smaller pieces, and pre-fill the cache with each chunk.
For this we can choose as chunk size the window size. For each chunk, we thus need to compute the attention over the cache and over the chunk.

![Chunking](assets/chunking.png)


## References

[1] [Generating Long Sequences with Sparse Transformers, Child et al. 2019](https://arxiv.org/pdf/1904.10509.pdf)

[2] [Longformer: The Long-Document Transformer, Beltagy et al. 2020](https://arxiv.org/pdf/2004.05150v2.pdf)