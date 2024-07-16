from dataclasses import dataclass
from typing import Optional

from simple_parsing.helpers import Serializable

from mistral_inference.lora import LoraArgs
from mistral_inference.moe import MoeArgs


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

    max_batch_size: int = 0

    # For rotary embeddings. If not set, will be inferred
    rope_theta: Optional[float] = None
    # If this is set, we will use MoE layers instead of dense layers.
    moe: Optional[MoeArgs] = None
    # If this is set, we will load LoRA linear layers instead of linear layers.
    lora: Optional[LoraArgs] = None
    model_type: str = "transformer"

    def __post_init__(self):
        assert self.model_type == "transformer", self.model_type


@dataclass
class MambaArgs(Serializable):
    dim: int
    n_layers: int
    vocab_size: int
    n_groups: int
    rms_norm: bool
    residual_in_fp32: bool
    fused_add_norm: bool
    pad_vocab_size_multiple: int
    tie_embeddings: bool
    model_type: str = "mamba"

    def __post_init__(self):
        assert self.model_type == "mamba", self.model_type
