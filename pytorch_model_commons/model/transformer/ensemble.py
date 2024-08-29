import torch
import torch.nn as nn
from typing import Optional, Tuple

from model.configs.transformer.ensemble_configs import GPTConfig
from model.transformer.blocks import TransformerBlock


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
