import torch
import torchvision
from omniglot_sampler import OmniglotSampler
from torch.utils.data import DataLoader
from torchvision.datasets import Omniglot


# Load Dataset and Sampler
dataset = Omniglot(root="./data", download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

# Build prototypical network
import pudb; pudb.set_trace()

# Training Loop
for i, batch in enumerate(dataloader):
    print(batch)

# Testing Loop