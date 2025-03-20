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
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.w_in = nn.Linear(
            in_dim,
            out_dim,
            bias=bias,
        )
        self.gelu = nn.GELU()
        self.w_out = nn.Linear(out_dim, out_dim, bias=bias)

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


class PatchMerger(nn.Module):
    """
    Learned merging of spatial_merge_size ** 2 patches
    """

    def __init__(
        self,
        vision_encoder_dim: int,
        spatial_merge_size: int,
    ) -> None:
        super().__init__()

        mlp_input_dim = vision_encoder_dim * (spatial_merge_size**2)

        self.spatial_merge_size = spatial_merge_size
        self.mlp_input_dim = mlp_input_dim

        self.merging_layer = nn.Linear(mlp_input_dim, vision_encoder_dim, bias=False)

    def forward(self, x: torch.Tensor, image_sizes: list[tuple[int, int]]) -> torch.Tensor:
        # image_sizes specified in tokens
        assert sum([h * w for h, w in image_sizes]) == len(x), f"{sum([h * w for h, w in image_sizes])} != {len(x)}"

        # x is (N, vision_encoder_dim)
        x = self.permute(x, image_sizes)

        # x is (N / spatial_merge_size ** 2,
        #       vision_encoder_dim * spatial_merge_size ** 2)
        x = self.merging_layer(x)

        # x is (N / spatial_merge_size ** 2, vision_encoder_dim)
        return x

    def permute(
        self,
        x: torch.Tensor,
        image_sizes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """
        Args:
            x: (N, D) where N is flattened and concatenated patch tokens
                for all images
            image_sizes: list of tuple of (height, width) in tokens for
                each image
        Returns:
            image_features: reorders patch tokens so each grid of
                (spatial_merge_size, spatial_merge_size) is contiguous.
                now (N / spatial_merge_size ** 2, D * spatial_merge_size ** 2)
        """

        sub_grids = get_sub_grids(
            x=x, image_sizes=image_sizes, spatial_merge_size=self.spatial_merge_size
        )  # list of [d x sub_grid_size x sub_grid_size x n_patches]
        permuted_tensor = [
            grid.view(-1, grid.shape[-1]).t() for grid in sub_grids
        ]  # n_patches x d * sub_grid_size * sub_grid_size
        return torch.cat(permuted_tensor, dim=0)  # (N / spatial_merge_size ** 2, d * spatial_merge_size ** 2)


def get_sub_grids(
    x: torch.Tensor,
    image_sizes: list[tuple[int, int]],
    spatial_merge_size: int,
) -> list[torch.Tensor]:
    # image_sizes specified in tokens
    tokens_per_image = [h * w for h, w in image_sizes]
    d = x.shape[-1]
    all_img_sub_grids: list[torch.Tensor] = []
    sub_grid_size = spatial_merge_size

    for image_index, image_tokens in enumerate(x.split(tokens_per_image)):
        # Reshape image_tokens into a 2D grid
        h, w = image_sizes[image_index]
        image_grid = image_tokens.view(h, w, d).permute(2, 0, 1)[None, :, :, :]  # 1 x d x h x w
        sub_grids = torch.nn.functional.unfold(image_grid, kernel_size=sub_grid_size, stride=sub_grid_size)
        sub_grids = sub_grids.view(
            1, d, sub_grid_size, sub_grid_size, -1
        )  # 1 x d x sub_grid_size x sub_grid_size x n_patches

        all_img_sub_grids.append(sub_grids[0])

    return all_img_sub_grids
