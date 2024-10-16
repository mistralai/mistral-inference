from dataclasses import dataclass
from typing import List, Optional

from simple_parsing.helpers import Serializable

from mistral_inference.lora import LoraArgs
from mistral_inference.moe import MoeArgs


@dataclass
class VisionEncoderArgs:
    hidden_size: int
    num_channels: int
    image_size: int
    patch_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    rope_theta: float = 1e4  # for rope-2D
    image_token_id: int = 10


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
    sliding_window: Optional[int] | Optional[List[int]] = None
    _sliding_window: Optional[int] | Optional[List[int]] = None
    model_type: str = "transformer"

    vision_encoder: Optional[VisionEncoderArgs] = None

    def __post_init__(self) -> None:
        assert self.model_type == "transformer", self.model_type
        assert self.sliding_window is None or self._sliding_window is None

        # hack for now so that vLLM is supported correctly
        self.sliding_window = self.sliding_window if self.sliding_window is not None else self._sliding_window


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

    def __post_init__(self) -> None:
        assert self.model_type == "mamba", self.model_type
