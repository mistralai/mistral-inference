from typing import List, Optional

import torch
import torch.nn as nn
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from mistral_inference.args import VisionEncoderArgs
from mistral_inference.rope import precompute_freqs_cis_2d
from mistral_inference.transformer_layers import RMSNorm, TransformerBlock


def position_meshgrid(
    patch_embeds_list: list[torch.Tensor],
) -> torch.Tensor:
    positions = torch.cat(
        [
            torch.stack(
                torch.meshgrid(
                    torch.arange(p.shape[-2]),
                    torch.arange(p.shape[-1]),
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 2)
            for p in patch_embeds_list
        ]
    )
    return positions


class VisionTransformer(nn.Module):
    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        self.args = args
        self.patch_conv = nn.Conv2d(
            in_channels=args.num_channels,
            out_channels=args.hidden_size,
            kernel_size=args.patch_size,
            stride=args.patch_size,
            bias=False,
        )
        self.ln_pre = RMSNorm(args.hidden_size, eps=1e-5)
        self.transformer = VisionTransformerBlocks(args)

        head_dim = self.args.hidden_size // self.args.num_attention_heads
        assert head_dim % 2 == 0, "ROPE requires even head_dim"
        self._freqs_cis: Optional[torch.Tensor] = None

    @property
    def max_patches_per_side(self) -> int:
        return self.args.image_size // self.args.patch_size

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def freqs_cis(self) -> torch.Tensor:
        if self._freqs_cis is None:
            self._freqs_cis = precompute_freqs_cis_2d(
                dim=self.args.hidden_size // self.args.num_attention_heads,
                height=self.max_patches_per_side,
                width=self.max_patches_per_side,
                theta=self.args.rope_theta,
            )

        if self._freqs_cis.device != self.device:
            self._freqs_cis = self._freqs_cis.to(device=self.device)

        return self._freqs_cis

    def forward(
        self,
        images: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            images: list of N_img images of variable sizes, each of shape (C, H, W)

        Returns:
            image_features: tensor of token features for all tokens of all images of
                shape (N_toks, D)
        """
        # pass images through initial convolution independently
        patch_embeds_list = [self.patch_conv(img.unsqueeze(0)).squeeze(0) for img in images]

        # flatten to a single sequence
        patch_embeds = torch.cat([p.flatten(1).permute(1, 0) for p in patch_embeds_list], dim=0)
        patch_embeds = self.ln_pre(patch_embeds)

        # positional embeddings
        positions = position_meshgrid(patch_embeds_list).to(self.device)
        freqs_cis = self.freqs_cis[positions[:, 0], positions[:, 1]]

        # pass through Transformer with a block diagonal mask delimiting images
        mask = BlockDiagonalMask.from_seqlens(
            [p.shape[-2] * p.shape[-1] for p in patch_embeds_list],
        )
        out = self.transformer(patch_embeds, mask=mask, freqs_cis=freqs_cis)

        # remove batch dimension of the single sequence
        return out  # type: ignore[no-any-return]


class VisionLanguageAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.w_in = nn.Linear(
            in_dim,
            out_dim,
            bias=True,
        )
        self.gelu = nn.GELU()
        self.w_out = nn.Linear(out_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(self.gelu(self.w_in(x)))  # type: ignore[no-any-return]


class VisionTransformerBlocks(nn.Module):
    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(args.num_hidden_layers):
            self.layers.append(
                TransformerBlock(
                    dim=args.hidden_size,
                    hidden_dim=args.intermediate_size,
                    n_heads=args.num_attention_heads,
                    n_kv_heads=args.num_attention_heads,
                    head_dim=args.hidden_size // args.num_attention_heads,
                    norm_eps=1e-5,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: BlockDiagonalMask,
        freqs_cis: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask, freqs_cis=freqs_cis)
        return x


