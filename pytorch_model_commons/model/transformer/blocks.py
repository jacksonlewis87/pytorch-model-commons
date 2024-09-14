import torch
import torch.nn as nn

from pytorch_model_commons.model.transformer.attention import get_attention_module
from pytorch_model_commons.model.configs.transformer.block_configs import TransformerBlockConfig


class TransformerBlock(nn.Module):
    """
    A single Transformer block consisting of a multi-head self-attention layer and a feed-forward network.

    Parameters:
        config (TransformerBlockConfig): Configuration object with parameters for the Transformer block.
            Attributes:
                - embedding_dim (int): Dimensionality of the input embeddings.
                - attention_config (CausalSelfAttentionConfig): Configuration for the self-attention layer.
                - resid_drop_p (float): Dropout probability for the feed-forward network.
    """

    def __init__(self, config: TransformerBlockConfig):
        super().__init__()
        attention_module = get_attention_module(attention_type=config.attention_type)

        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.ln2 = nn.LayerNorm(config.embedding_dim)
        self.attn = attention_module(config.attention_config)
        self.mlp = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.embedding_dim),
            nn.GELU(),
            nn.Linear(4 * config.embedding_dim, config.embedding_dim),
            nn.Dropout(config.resid_drop_p),
        )

    def forward(self, x, layer_past=None, return_present=False):
        """
        Forward pass through the Transformer block.

        Parameters:
            x (torch.Tensor): Input tensor of shape (B, T, C), where B is batch size, T is sequence length, and C is embedding dimension.
            layer_past (tuple, optional): Tuple of past key and value tensors for autoregressive tasks.
            return_present (bool, optional): Whether to return the present key and value tensors for caching.

        Returns:
            torch.Tensor: Output tensor after applying self-attention and feed-forward network.
            tuple (optional): If return_present is True or layer_past is not None, returns a tuple containing the output tensor and present tensor.
        """

        if return_present:
            assert not self.training, "Error: caching not enabled during training"

        attn, present = self.attn(self.ln1(x), layer_past=layer_past)

        x = x + attn
        x = x + self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            return x, present
        return x
