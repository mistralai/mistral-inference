import json
from pathlib import Path
from typing import List, Optional, Union

import safetensors
import torch
import torch.nn as nn

from mistral_inference.args import MambaArgs
from mistral_inference.cache import BufferCache
from mistral_inference.model import ModelBase

_is_mamba_installed = False
try:
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    _is_mamba_installed = True
except ImportError:
    _is_mamba_installed = False


class Mamba(ModelBase, nn.Module):
    def __init__(self, args: MambaArgs):
        super().__init__()
        self.args = args
        assert _is_mamba_installed, "Mamba is not installed. Please install it using `pip install mamba-ssm`."

        # make sure naming is consistent with `mamba_ssm`
        config = MambaConfig(
            d_model=args.dim,
            n_layer=args.n_layers,
            vocab_size=args.vocab_size,
            ssm_cfg={"ngroups": args.n_groups, "layer": "Mamba2"},
            attn_layer_idx=[],
            attn_cfg={},
            rms_norm=args.rms_norm,
            residual_in_fp32=args.residual_in_fp32,
            fused_add_norm=args.fused_add_norm,
            pad_vocab_size_multiple=args.pad_vocab_size_multiple,
            tie_embeddings=args.tie_embeddings,
        )
        self.model = MambaLMHeadModel(config)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],  # not supported for now
        cache: Optional[BufferCache] = None,  # not supported for now
    ) -> torch.Tensor:
        lm_output = self.model(input_ids)
        result: torch.Tensor = lm_output.logits
        return result

    @staticmethod
    def from_folder(
        folder: Union[Path, str],
        max_batch_size: int = 1,
        num_pipeline_ranks: int = 1,
        device: Union[torch.device, str] = "cuda",
        dtype: Optional[torch.dtype] = None,
    ) -> "Mamba":
        with open(Path(folder) / "params.json", "r") as f:
            model_args = MambaArgs.from_dict(json.load(f))

        with torch.device("meta"):
            model = Mamba(model_args)

        model_file = Path(folder) / "consolidated.safetensors"

        assert model_file.exists(), f"Make sure {model_file} exists."
        loaded = safetensors.torch.load_file(str(model_file))

        model.load_state_dict(loaded, assign=True, strict=True)
        return model.to(device=device, dtype=dtype)
