import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X = torch.linspace(0, 1, 100).view(-1, 1)
y = torch.sin(2 * torch.pi * X) + 0.1 * torch.rand(X.size())

dataset = CustomDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10)

for batch in dataloader:
    data, labels = batch
    print("Batch data shape:", data.shape)
    print("Batch labels shape:", labels.shape)
    print("Data:", data)
    print("Labels:", labels)
    break  # Выход
