import torch
import torch.nn as nn
from typing import Optional, Tuple

from model.configs.transformer.ensemble_configs import GPTConfig, TransformerModuleConfig
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


class TransformerModule(nn.Module):
    """
    Transformer module that can be used as an encoder or decoder, depending on the configuration.

    Attributes:
        proj_in (nn.Linear): Linear layer to project input embeddings to the dimension used within transformer blocks.
        pos_emb (nn.Parameter): Positional embedding parameters.
        blocks (nn.Sequential): Stack of transformer blocks.
        proj_out (nn.Linear): Linear layer to project the output of transformer blocks to the desired output dimension.
        drop (nn.Dropout): Dropout layer for embeddings.
        input_dim (int): Dimension of the input embeddings.
        hidden_dim (int): Dimension used within the transformer blocks.
        output_dim (int): Dimension of the output representations.
        block_size (int): Maximum sequence length the model can handle.
        config (TransformerModuleConfig): Configuration object containing model parameters.
    """

    def __init__(self, config: TransformerModuleConfig):
        """
        Initializes the Transformer module with the given configuration.

        Parameters:
            config (TransformerModuleConfig): Configuration object for the Transformer module.
        """
        super().__init__()

        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.block_size = config.block_size

        # Projection layer to transform input embeddings to the dimension used within transformer blocks
        self.proj_in = nn.Linear(self.input_dim, self.hidden_dim)

        # Positional embeddings
        self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, self.hidden_dim))

        # Transformer blocks
        self.blocks = nn.Sequential(*[TransformerBlock(config=config.block_config) for _ in range(config.num_layers)])

        # Dropout layer for embeddings
        self.drop = nn.Dropout(config.embed_drop_p)

        # Projection layer to transform the output of transformer blocks to the desired output dimension
        self.proj_out = nn.Linear(self.hidden_dim, self.output_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initializes weights for linear layers and layer normalization.

        Parameters:
            module (nn.Module): Module to initialize.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer module.

        Parameters:
            x (torch.Tensor): Input tensor of embeddings of shape (B, T, input_dim), where B is the batch size,
                              T is the sequence length, and input_dim is the dimensionality of input embeddings.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, output_dim).
        """
        # Project input embeddings to the dimension used within transformer blocks
        x = self.proj_in(x)

        # Add positional embeddings
        batch_size, seq_length = x.size(0), x.size(1)
        assert seq_length <= self.block_size, "Sequence length exceeds model block size."
        position_embeddings = self.pos_emb[:, :seq_length, :]
        x = x + position_embeddings

        # Apply dropout
        x = self.drop(x)

        # Pass through transformer blocks
        x = self.blocks(x)

        # Project the output of transformer blocks to the desired output dimension
        x = self.proj_out(x)

        return x
