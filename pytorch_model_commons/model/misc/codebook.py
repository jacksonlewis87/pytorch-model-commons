import torch
import torch.nn as nn

from model.configs.misc.codebook_configs import CodebookConfig


class Codebook(nn.Module):
    def __init__(self, config: CodebookConfig):
        super(Codebook, self).__init__()
        self.codebook_size = config.codebook_size
        self.embedding_dim = config.embedding_dim
        self.beta = config.beta

        self.embedding = nn.Embedding(self.codebook_size, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * (torch.matmul(z_flattened, self.embedding.weight.t()))
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_encoding_indices, loss
