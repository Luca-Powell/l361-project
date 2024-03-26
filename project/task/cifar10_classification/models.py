"""Define our models, and training and eval functions for CIFAR10 classification."""

import torch
import torch.nn.functional as F
from torch import nn
import copy
from project.types.common import NetGen
from project.utils.utils import lazy_config_wrapper

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from typing import Any


# helpers
def _pair(t: Any) -> tuple[Any, Any]:
    return t if isinstance(t, tuple) else (t, t)


# classes
class FeedForward(nn.Module):
    """Helper feedforward class for ViT."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ViT."""
        return self.net(x)


class Attention(nn.Module):
    """Helper attention class for ViT."""

    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the attention mechanism."""
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # q,k,v=map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    """Helper transformer class for ViT."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    FeedForward(dim, mlp_dim, dropout=dropout),
                ])
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of transformer."""
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(nn.Module):
    """Vision Transformer class."""

    def __init__(
        self,
        *,
        image_size: int = 32,
        patch_size: int = 8,
        num_classes: int = 10,
        dim: int = 512,
        depth: int = 10,
        heads: int = 8,
        mlp_dim: int = 1024,
        pool: str = "cls",
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        image_height, image_width = _pair(image_size)
        patch_height, patch_width = _pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) \
                                            or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ViT.

        Parameters
        ----------
        img : torch.Tensor
            Input Tensor that will pass through the network

        Returns
        -------
        torch.Tensor
            The resulting Tensor after it has passed through the network
        """
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


# Simple wrapper to match the NetGenerator Interface
get_vit: NetGen = lazy_config_wrapper(ViT)


class Net(nn.Module):
    """Convolutional Neural Network architecture.

    As described in McMahan 2017 paper :

    [Communication-Efficient Learning of Deep Networks from
    Decentralized Data] (https://arxiv.org/pdf/1602.05629.pdf)
    """

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize the network.

        Parameters
        ----------
        num_classes : int
            Number of classes in the dataset.

        Returns
        -------
        None
        """
        super().__init__()

        num_channels = 3

        self.conv1 = nn.Conv2d(num_channels, 32, 5, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)
        self.pool = nn.MaxPool2d(
            kernel_size=(2, 2),
            padding=1,
        )
        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the CNN.

        Parameters
        ----------
        x : torch.Tensor
            Input Tensor that will pass through the network

        Returns
        -------
        torch.Tensor
            The resulting Tensor after it has passed through the network
        """
        output_tensor = F.relu(self.conv1(input_tensor))
        output_tensor = self.pool(output_tensor)

        output_tensor = F.relu(self.conv2(output_tensor))
        output_tensor = self.pool(output_tensor)

        output_tensor = torch.flatten(output_tensor, 1)
        output_tensor = F.relu(self.fc1(output_tensor))

        output_tensor = self.fc2(output_tensor)

        return output_tensor

    def get_activations(
        self,
        input_tensor: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Get the layer activations for a given input."""
        # for storing the activations after all non-linear steps
        activations = []

        output_tensor = F.relu(self.conv1(input_tensor))
        activations.append(copy.deepcopy(output_tensor.detach().cpu().numpy().tolist()))

        output_tensor = self.pool(output_tensor)
        output_tensor = F.relu(self.conv2(output_tensor))
        activations.append(copy.deepcopy(output_tensor.detach().cpu().numpy().tolist()))

        output_tensor = self.pool(output_tensor)
        output_tensor = torch.flatten(output_tensor, 1)
        output_tensor = F.relu(self.fc1(output_tensor))
        activations.append(copy.deepcopy(output_tensor.detach().cpu().numpy().tolist()))

        output_tensor = self.fc2(output_tensor)
        return activations


def load_cnn_pretrained_50percent() -> Net:
    """Load a CNN pre-trained to 50% convergence on cifar10."""
    return torch.load(
        "/nfs-share/lp647/L361/l361-project/project/task/cifar10_classification/models_pretrained/cnn_pretrained_cifar10_50p_batch32.pt"
    )


# Simple wrappers to match the NetGenerator interface
get_net: NetGen = lazy_config_wrapper(Net)
get_net_pretrained_50: NetGen = lazy_config_wrapper(load_cnn_pretrained_50percent)
