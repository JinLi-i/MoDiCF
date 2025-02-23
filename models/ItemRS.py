import torch
import torch.nn as nn

class ItemRS(nn.Module):
    def __init__(self, input_size, modalities, embedding_dim):
        super(ItemRS, self).__init__()
        self.input_size = input_size
        self.modalities = modalities
        self.embedding_dim = embedding_dim
        self.n_modalities = len(modalities)

        self.encoder = nn.ModuleDict()
        for i in range(self.n_modalities):
            branch_name = modalities[i] + '_enc'
            mid = int(self.input_size[i] / 2)
            if self.embedding_dim < mid:
                mid = self.embedding_dim
            enc = nn.Sequential(
                nn.Linear(self.input_size[i], mid),
                nn.LeakyReLU(),
                nn.Linear(mid, self.embedding_dim),
                nn.LeakyReLU(),
            )
            self.encoder[branch_name] = enc

        self.out = nn.Sequential(
            nn.Linear(self.embedding_dim * self.n_modalities, self.embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(self.embedding_dim, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        encoded = []
        for i in range(self.n_modalities):
            branch_name = self.modalities[i] + '_enc'
            enc = self.encoder[branch_name](x[i])
            encoded.append(enc)
        encoded = torch.cat(encoded, 1)
        out = self.out(encoded)
        out_sigmoid = self.sigmoid(out)
        return out, out_sigmoid