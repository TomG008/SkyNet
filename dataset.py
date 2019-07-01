import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class customDataset(Dataset):
    def __init__(self, root, shape=None, transform=None, batch_size=32, num_workers=4):
        self.files = [file for file in os.listdir(root) if os.path.isfile(os.path.join(root, file))]
        self.imageNames = [file.split('.')[0] for file in self.files]
        self.files = [os.path.join(root, file) for file in self.files]
        self.nSamples = len(self.files)
        self.transform = transform
        self.shape = shape
        self.batch_size = batch_size
        self.num_workers = num_workers
       
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        imgpath = self.files[index]
        img = Image.open(imgpath).convert('RGB')
        if self.shape:
            img = img.resize(self.shape)

        if self.transform is not None:
            img = self.transform(img)

        return img
        