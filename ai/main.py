import torch
import dataset
import torchvision

from torch.utils.data import DataLoader

resizer = torchvision.transforms.Resize((256, 256))

data = dataset.ImageDataset(
        "data/",
        transform = lambda img: resizer.forward(img),
        target_transform = lambda y: torch.zeros(
            2, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1)
    )

train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

print(data[0][0].size())
