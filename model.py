import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import LogNormal, Dirichlet


class Encoder(nn.Module):
    def __init__(self, trajectory_dim, hidden_size, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(trajectory_dim, hidden_size)
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)

    def forward(self, inputs):
        h1, _ = self.gru(inputs)
        h2 = F.softplus(self.conv(h1))
        return self.drop(h2)


class Hidden(nn.Module):
    def __init__(self, hidden_size, num_topics):
        super().__init__()
        self.fcmu = nn.Linear(hidden_size, num_topics)
        self.fclv = nn.Linear(hidden_size, num_topics)

    def forward(self, hidden):
        mu = self.fcmu(hidden)
        lv = self.fclv(hidden)
        dist = LogNormal(mu, (0.5 * lv).exp())
        return dist


class Decoder(nn.Module):
    def __init__(self, trajectory_dim, hidden_size, num_topics, dropout):
        super().__init__()
        self.convT = nn.ConvTranspose1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.gru = nn.GRU(hidden_size, trajectory_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        h1 = F.relu(self.convT(inputs))
        h2, _ = self.gru(h1)
        return self.drop(h2)


class GeoDrivenTopicModel(nn.Module):
    def __init__(self, trajectory_dim, hidden_size, num_topics, dropout):
        super().__init__()
        self.encode = Encoder(trajectory_dim, hidden_size, dropout)
        self.h2z = HiddenToLogNormal(hidden_size, num_topics)
        self.decode = Decoder(trajectory_dim, hidden_size, num_topics, dropout)

    def forward(self, inputs):
        h = self.encode(inputs)
        posterior = self.h2z(h)
        z = posterior.rsample()
        outputs = self.decode(z)
        return outputs, posterior
