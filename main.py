import numpy as np
import h5py
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')  # Set a compatible backend
import matplotlib.pyplot as plt


class HeatmapTauData(Dataset):
    def __init__(self, hdf5):
        self.hdf5 = hdf5
        self.file = h5py.File(hdf5, 'r')
        self.ksz_maps = self.file["ksz_maps"]
        self.tau_values = self.file["tau_values"]
        self.num_samples = self.ksz_maps.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        heatmap = self.ksz_maps[index]
        tauval = self.tau_values[index]

        heatmap = torch.tensor(heatmap, dtype=torch.float32)
        tauval = torch.tensor(tauval, dtype=torch.float32)

        return heatmap, tauval

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.convo1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.convo2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = torch.nn.Linear(32 * 256 * 256, 128)
        self.linear2 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.convo1(x)))
        x = self.pool2(F.relu(self.convo2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

dataset = HeatmapTauData("astro.hdf5")

dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
blur = T.GaussianBlur(kernel_size=5, sigma=(1.0, 2.0))

model = Model()
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #lr is learning rate 0.001 standard starting point



tau_min = dataset.tau_values[:].min()
tau_max = dataset.tau_values[:].max()

losses = []

for epoch in range(5):

    model.train()
    totalLoss = 0.0
    # print(f"Epoch {epoch + 1}, Loss: {totalLoss:.4f}")
    # losses.append(totalLoss)

    for batch in dataloader:
        heatmaps, tauvals = batch
        heatmaps = heatmaps.unsqueeze(1)
        heatmaps = blur(heatmaps)
        tauvals = tauvals.view(-1, 1)

        mean = heatmaps.mean(dim=(2, 3), keepdim=True)
        std = heatmaps.std(dim=(2, 3), keepdim=True)
        heatmaps = (heatmaps - mean) / (std + 1e-8)

        tauvals = (tauvals - tau_min) / (tau_max - tau_min)

        optimizer.zero_grad() #reset previous gradients to zero to not mess up calculations
        output = model(heatmaps)
        iterationLoss = loss(output, tauvals)
        iterationLoss.backward()
        optimizer.step()
        totalLoss += iterationLoss.item()

    print(f"Epoch {epoch + 1}, Loss: {totalLoss:.4f}")
    losses.append(totalLoss)

plt.plot(losses, marker='o')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()

