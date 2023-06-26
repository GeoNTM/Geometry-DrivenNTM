import torch
from torch import nn
from torch.nn import functional as F

class GeoDrivenTopicModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_topics):
        super(GeoDrivenTopicModel, self).__init__()
        
        # Inference Network
        self.inference_net = nn.Sequential(
            nn.GRU(input_dim, hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)  
        )
        
        # Generative Network
        self.generative_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GRU(hidden_dim, input_dim),
        )
        
        self.num_topics = num_topics

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x):
        # Inference network
        mu, logvar = self.inference_net(x).chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        
        # Sample a topic
        z_topics = F.softmax(z[:,:self.num_topics], dim=1)
        
        # Generative network
        x_recon = self.generative_net(z)
        
        return x_recon, mu, logvar, z_topics
