# DeepSeekers
# Convolutional Neural Network (CNN) to predict tau value from heatmap.

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Class for loading heatmap and tau data
class HeatmapTauData(Dataset):
    def __init__(self, hdf5):
        self.hdf5 = hdf5                            # Path to HDF5 file
        self.file = h5py.File(hdf5, 'r')      # Open file in read only
        self.ksz_maps = self.file["ksz_maps"]       # Load heatmap
        self.tau_values = self.file["tau_values"]   # Load tau value
        self.num_samples = self.ksz_maps.shape[0]   # Number of samples

    # Function to get number of samples
    def __len__(self):
        return self.num_samples     # Return the number of samples

    # Function to get a heat map and tau value at index
    def __getitem__(self, index):
        # Load the heatmap and tau at given index
        heatmap = self.ksz_maps[index]
        tauval = self.tau_values[index]

        # Convert heatmap and tau to tensor
        heatmap = torch.tensor(heatmap, dtype=torch.float32)
        tauval = torch.tensor(tauval, dtype=torch.float32)

        # Return the heatmap and tau
        return heatmap, tauval


# CNN Model
class Model(torch.nn.Module):
    def __init__(self):
        # Initialize base class
        super(Model, self).__init__()
        # Convolutional layer: 1 input 16 output
        self.convo1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Max pooling to reduce to half
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # Convolutional layer: 16 input 32 output
        self.convo2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Max pooling to reduce to half
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layer 128 features
        self.linear1 = torch.nn.Linear(32 * 256 * 256, 128)
        # Output layer for the single tau value
        self.linear2 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.convo1(x)))
        x = self.pool2(F.relu(self.convo2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# Load the data
dataset = HeatmapTauData("data/ten_snapshots.hdf5")

# Shuffle and batch the data set
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Set the Gaussian blur
blur = T.GaussianBlur(kernel_size=5, sigma=(1.0, 2.0))

# Create the model, loss, and optimizer
model = Model()
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # lr is learning rate 0.001 standard starting point

# Normalize tau values between [0,1]
tau_min = dataset.tau_values[:].min()
tau_max = dataset.tau_values[:].max()

# Variable to store losses
losses = []

# Loop to train for 5 epochs
for epoch in range(5):
    # Set the model to train
    model.train()
    # Initialize total loss
    totalLoss = 0.0

    # print(f"Epoch {epoch + 1}, Loss: {totalLoss:.4f}")
    # losses.append(totalLoss)

    # Loop to iterate through batches
    for batch in dataloader:
        # Get the input and target
        heatmaps, tauvals = batch
        # Add channel dimension
        heatmaps = heatmaps.unsqueeze(1)
        # Apply the Gaussian blur
        heatmaps = blur(heatmaps)
        # Reshape tau values
        tauvals = tauvals.view(-1, 1)

        # Compute mean and std to normalize
        mean = heatmaps.mean(dim=(2, 3), keepdim=True)
        std = heatmaps.std(dim=(2, 3), keepdim=True)
        heatmaps = (heatmaps - mean) / (std + 1e-8)

        # Normalize tau values to [0,1]
        tauvals = (tauvals - tau_min) / (tau_max - tau_min)

        # Reset gradients
        optimizer.zero_grad() # Reset previous gradients to zero to not mess up calculations
        # Forward pass
        output = model(heatmaps)
        # Compute the loss
        iterationLoss = loss(output, tauvals)
        # Backpropagation
        iterationLoss.backward()
        # Update weights
        optimizer.step()

        # Accumulate the loss
        totalLoss += iterationLoss.item()

    # Print loss at the end of each epoch
    print(f"Epoch {epoch + 1}, Loss: {totalLoss:.4f}")
    losses.append(totalLoss)

# Plot training
plt.plot(losses, marker='o')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()
