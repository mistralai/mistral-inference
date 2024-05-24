import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, NamedTuple, Union

import safetensors.torch
import torch
import torch.nn as nn
from simple_parsing.helpers import Serializable


@dataclass
class LoraArgs(Serializable):
    rank: int
    scaling: float

    def __post_init__(self):
        assert self.rank > 0
        assert self.scaling > 0.0


class LoRALinear(nn.Module):
    """
    Implementation of:
        - LoRA: https://arxiv.org/abs/2106.09685

    Notes:
        - Freezing is handled at network level, not layer level.
        - Scaling factor controls relative importance of LoRA skip
          connection versus original frozen weight. General guidance is
          to keep it to 2.0 and sweep over learning rate when changing
          the rank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        scaling: float,
        bias: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        assert not bias
        self.bias = bias
        self.rank = rank
        self.scaling = scaling

        self.lora_A = nn.Linear(
            self.in_features,
            self.rank,
            bias=self.bias,
        )
        self.lora_B = nn.Linear(
            self.rank,
            self.out_features,
            bias=self.bias,
        )

        self.linear = nn.Linear(self.in_features, self.out_features, bias=self.bias)

        # make sure no LoRA weights are marked as "missing" in load_state_dict
        def ignore_missing_keys(m: nn.Module, incompatible_keys: NamedTuple):
            incompatible_keys.missing_keys[:] = []  # type: ignore

        self.register_load_state_dict_post_hook(ignore_missing_keys)

    def forward(self, x: torch.Tensor):
        lora = self.lora_B(self.lora_A(x))
        return self.linear(x) + lora * self.scaling

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key_name = prefix + "weight"

        # full checkpoint
        if key_name in state_dict:
            w_ref = state_dict[key_name]

            # load frozen weights
            state_dict = {
                "linear.weight": w_ref,
                "lora_A.weight": torch.zeros_like(
                    self.lora_A.weight, device=w_ref.device, dtype=w_ref.dtype
                ),
                "lora_B.weight": torch.zeros_like(
                    self.lora_B.weight, device=w_ref.device, dtype=w_ref.dtype
                ),
            }
            self.load_state_dict(state_dict, assign=True, strict=True)


class LoRALoaderMixin:
    def load_lora(self, lora_path: Union[Path, str], scaling: float = 2.0):
        """Loads LoRA checkpoint"""

        lora_path = Path(lora_path)
        assert lora_path.is_file(), f"{lora_path} does not exist or is not a file"

        state_dict = safetensors.torch.load_file(lora_path)

        self._load_lora_state_dict(state_dict, scaling=scaling)

    def _load_lora_state_dict(
        self, lora_state_dict: Dict[str, torch.Tensor], scaling: float = 2.0
    ):
        """Loads LoRA state_dict"""

        lora_dtypes = set([p.dtype for p in lora_state_dict.values()])
        assert (
            len(lora_dtypes) == 1
        ), f"LoRA weights have multipe different dtypes {lora_dtypes}. All weights need to have the same dtype"
        lora_dtype = lora_dtypes.pop()
        assert (
            lora_dtype == self.dtype
        ), f"LoRA weights dtype differs from model's dtype {lora_dtype} != {self.dtype}"
        assert all("lora" in key for key in lora_state_dict.keys())

        # move tensors to device
        lora_state_dict = {k: v.to(self.device) for k, v in lora_state_dict.items()}

        state_dict = self.state_dict()

        if self.args.lora is None:
            logging.info("Loading and merging LoRA weights...")

            # replace every nn.Linear with a LoRALinear with 'meta' device except the output layer
            named_modules = dict(self.named_modules())
            for name, module in named_modules.items():
                if isinstance(module, nn.Linear) and name != "output":
                    layer_id = name.split(".")[1]
                    if layer_id not in self.layers:
                        logging.debug(
                            "Skipping parameter %s at pipeline rank %d",
                            name,
                            self.pipeline_rank,
                        )
                    else:
                        weight = (
                            module.weight
                            + (
                                lora_state_dict[name + ".lora_B.weight"]
                                @ lora_state_dict[name + ".lora_A.weight"]
                            )
                            * scaling
                        )

                        state_dict[name + ".weight"] = weight
        else:
            logging.info("Loading LoRA weights...")
            for k, v in lora_state_dict.items():
                state_dict.update(lora_state_dict)

                layer_id = k.split(".")[1]
                if layer_id in self.layers:
                    state_dict[k] = v
                else:
                    logging.debug(
                        "Skipping parameter %s at pipeline rank %d",
                        k,
                        self.pipeline_rank,
                    )

        self.load_state_dict(state_dict, strict=True)
