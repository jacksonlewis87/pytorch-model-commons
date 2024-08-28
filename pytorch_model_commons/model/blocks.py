import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple

from model.model_configs import CausalSelfAttentionConfig, GPTConfig, TransformerBlockConfig


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with an output projection.

    This class implements a self-attention mechanism with multiple heads and a causal (masked) attention pattern.
    It ensures that each token can only attend to earlier tokens in the sequence, making it suitable for autoregressive tasks.

    Attributes:
        num_heads (int): The number of attention heads. This allows the model to jointly attend to information from different representation subspaces.
        embedding_dim (int): The dimensionality of the input and output embeddings. Must be divisible by num_heads.
        key (nn.Linear): Linear layer for projecting the input to the key space.
        query (nn.Linear): Linear layer for projecting the input to the query space.
        value (nn.Linear): Linear layer for projecting the input to the value space.
        attn_drop (nn.Dropout): Dropout layer applied to the attention weights.
        resid_drop (nn.Dropout): Dropout layer applied to the final output.
        proj (nn.Linear): Linear layer for projecting the concatenated multi-head outputs back to the embedding dimension.
        mask (torch.Tensor): Causal mask tensor to ensure that attention is applied only to preceding tokens in the sequence.

    Parameters:
        config (CausalSelfAttentionConfig): Configuration object containing hyperparameters for the attention layer.
            Attributes of the config:
                - embedding_dim (int): Dimensionality of the input embeddings.
                - num_heads (int): Number of attention heads.
                - attn_drop_p (float): Dropout probability for the attention weights.
                - resid_drop_p (float): Dropout probability for the final output.
                - block_size (int): Length of the input sequence (for masking purposes).
    """

    def __init__(self, config: CausalSelfAttentionConfig):
        super().__init__()
        assert (
            config.embedding_dim % config.num_heads == 0
        ), "Error: embedding dimension must be divisible by number of heads"

        self.num_heads = config.num_heads
        self.embedding_dim = config.embedding_dim

        # key, query, value projections for all heads
        self.key = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.query = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.value = nn.Linear(self.embedding_dim, self.embedding_dim)

        # dropout
        self.attn_drop = nn.Dropout(config.attn_drop_p)
        self.resid_drop = nn.Dropout(config.resid_drop_p)

        # output projection
        self.proj = nn.Linear(self.embedding_dim, self.embedding_dim)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x, layer_past=None):
        """
        Perform a forward pass through the causal self-attention layer.

        This method applies self-attention to the input tensor `x` while enforcing a causal (masked) attention pattern,
        ensuring that each token only attends to preceding tokens in the sequence. It also handles caching of past key
        and value tensors if provided, which is useful for autoregressive tasks such as language generation.

        Parameters:
            x (torch.Tensor): Input tensor of shape (B, T, C), where B is the batch size, T is the sequence length, and C
                              is the embedding dimension. This tensor represents the input sequence to the self-attention layer.
            layer_past (tuple of torch.Tensor, optional): A tuple containing past key and value tensors from previous
                      layers or time steps, used for extending the context in autoregressive models. If provided, the past
                      key and value tensors are concatenated with the current key and value tensors to form a longer context.
                      The tuple should have two elements:
                          - past_key (torch.Tensor): Tensor of shape (B, num_heads, past_T, head_dim), where past_T is the length
                            of the past sequence, and head_dim is the dimensionality of each attention head.
                          - past_value (torch.Tensor): Tensor of shape (B, num_heads, past_T, head_dim), corresponding to the past
                            value tensors.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C), representing the result of applying multi-head self-attention
                          and a linear projection to the input tensor `x`.
            tuple (optional): If `layer_past` is provided, returns a tuple containing:
                - output (torch.Tensor): The output tensor after applying self-attention and the output projection.
                - present (tuple of torch.Tensor): A tuple containing the updated key and value tensors of shape (B, num_heads, T, head_dim),
                  which can be used for caching in future time steps. This is useful for autoregressive generation tasks.

        Notes:
            - The method uses a causal (masked) attention mechanism to ensure that the attention weights for each token only
              depend on preceding tokens, not future ones.
            - The attention weights are scaled by the square root of the head dimension and normalized using softmax.
            - Dropout is applied to the attention weights to prevent overfitting.
            - If `layer_past` is not provided, the method assumes it is the first pass and applies masking to the attention
              scores to enforce the causal pattern.
        """

        B, T, C = x.size()

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        att = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)

        if layer_past is None:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Compute the attention output
        y = torch.matmul(att, v)  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs side by side

        # Output projection
        y = self.resid_drop(self.proj(y))

        # Create present tensor for caching if layer_past is used
        present = (k, v) if layer_past is not None else None

        return y, present


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
        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.ln2 = nn.LayerNorm(config.embedding_dim)
        self.attn = CausalSelfAttention(config.attention_config)
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


class GPT(nn.Module):
    """
    GPT model implementation with a stack of transformer blocks.

    Attributes:
        tok_emb (nn.Embedding): Token embedding layer.
        pos_emb (nn.Parameter): Positional embedding parameters.
        drop (nn.Dropout): Dropout layer for embeddings.
        blocks (nn.Sequential): Stack of transformer blocks.
        ln_f (nn.LayerNorm): Final layer normalization.
        head (nn.Linear): Linear layer for predicting vocabulary logits.
        block_size (int): Maximum sequence length the model can handle.
        config (GPTConfig): Configuration object containing model parameters.
    """

    def __init__(self, config: GPTConfig):
        """
        Initializes the GPT model with the given configuration.

        Parameters:
            config (GPTConfig): Configuration object for the GPT model.
        """
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.embedding_dim))
        self.drop = nn.Dropout(config.embed_drop_p)

        # transformer blocks
        self.blocks = nn.Sequential(*[TransformerBlock(config=config.block_config) for _ in range(config.num_layers)])

        # decoder head
        self.ln_f = nn.LayerNorm(config.embedding_dim)
        self.head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initializes weights for linear layers, embeddings, and layer normalization.

        Parameters:
            module (nn.Module): Module to initialize.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self, idx: torch.Tensor, embeddings: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[None]]:
        """
        Forward pass through the GPT model.

        Parameters:
            idx (torch.Tensor): Input tensor of token indices of shape (B, T).
            embeddings (Optional[torch.Tensor]): Optional explicit embeddings to prepend.

        Returns:
            Tuple[torch.Tensor, Optional[None]]:
                - torch.Tensor: Logits of shape (B, T, vocab_size) from the final linear layer.
                - Optional[None]: Placeholder for `layer_past` if not used.
        """
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector

        if embeddings is not None:  # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        return logits, None
