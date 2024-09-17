import torch
import torch.nn as nn

from pytorch_model_commons.model.configs.misc.codebook_configs import CodebookConfig


class Codebook(nn.Module):
    """
    A Codebook module for vector quantization in a neural network.

    This class implements a codebook for vector quantization, where input tensors are mapped to a discrete set of embeddings.
    It is commonly used in scenarios like vector quantization networks or discrete representations in neural networks.

    Attributes:
        codebook_size (int): The number of vectors in the codebook (i.e., the size of the discrete space).
        embedding_dim (int): The dimensionality of each vector in the codebook.
        beta (float): A regularization parameter to balance between reconstruction loss and codebook loss.

        embedding (nn.Embedding): An embedding layer that stores the codebook vectors. Initialized with uniform values.
    """

    def __init__(self, config: CodebookConfig):
        """
        Initializes the Codebook module.

        Parameters:
            config (CodebookConfig): Configuration object containing hyperparameters for the codebook.
                Attributes:
                    codebook_size (int): Number of vectors in the codebook.
                    embedding_dim (int): Dimensionality of each codebook vector.
                    beta (float): Regularization parameter for balancing reconstruction and codebook loss.
        """
        super().__init__()
        self.codebook_size = config.codebook_size
        self.embedding_dim = config.embedding_dim
        self.beta = config.beta

        self.embedding = nn.Embedding(self.codebook_size, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z):
        """
        Forward pass through the codebook layer.

        Parameters:
            z (torch.Tensor): Input tensor of shape (B, T, E), where B is the batch size, T is the sequence length,
                              and E is the embedding dimension. This tensor represents the input to be quantized.

        Returns:
            tuple:
                - z_q (torch.Tensor): Quantized tensor of the same shape as `z`, where each vector is replaced by its nearest
                  codebook vector.
                - min_encoding_indices (torch.Tensor): Indices of the nearest codebook vectors for each input vector.
                - loss (torch.Tensor): Scalar tensor representing the quantization loss, which is a combination of reconstruction
                  loss and codebook loss.
        """
        # Shape of z: (B, T, E)
        # Flatten z to shape (B * T, E) for distance computation
        z_flattened = z.view(-1, self.embedding_dim)

        # Compute the distances between input vectors and codebook vectors
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        # Find the indices of the nearest codebook vectors
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # Compute the loss: mean squared error between quantized and input tensors, with regularization
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # Update quantized tensor with codebook vectors while detaching the gradient
        z_q = z + (z_q - z).detach()

        return z_q, min_encoding_indices, loss
