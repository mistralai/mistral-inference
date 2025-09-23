from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn

from mistral_inference.cache import BufferCache


class ModelBase(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],  # not supported for now
        cache: Optional[BufferCache] = None,  # not supported for now
    ) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def from_folder(
        folder: Union[Path, str],
        max_batch_size: int = 1,
        num_pipeline_ranks: int = 1,
        device: Union[torch.device, str] = "cuda",
        dtype: Optional[torch.dtype] = None,
    ) -> "ModelBase":
        pass
