import os

from torch.utils.data import Dataset
from torchvision.io import read_image

class ImageDataset(Dataset):

    def __init__(self, img_dir, transform=None, target_transform=None):
        self.items = [len(os.listdir(img_dir+"original")), len(os.listdir(img_dir+"modified"))]
        self.working_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return self.items[0]+self.items[1]

    def __getitem__(self, idx):
        if idx < self.items[0]:
            image = read_image(self.working_dir+"original/ori_"+str(idx+1)+".jpg")
            label = 0
        elif idx < self.items[0]+self.items[1]:
            image = read_image(self.working_dir+"modified/mod_"+str(idx+1-self.items[0])+".jpg")
            label = 1
        else:
            print("ERROR: Index higher than limit")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
