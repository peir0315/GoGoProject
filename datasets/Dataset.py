import os
from os.path import join as PJ
from PIL import Image
import torch
from torch.utils.data import Dataset


class CIFAR10Dataset(Dataset):
    def __init__(self, data_root, data_file, class2idx, transform=None):
        self.data_root = PJ(os.getcwd(), *data_root)
        self.class2idx = class2idx
        self.transform = transform

        # Load data path
        with open(PJ(self.data_root, data_file), 'r') as f:
            data = f.readlines()
        self.data = [line.strip().split() for line in data]

    def __len__(self):
        """ Generate index list for __getitem__ """
        return len(self.data)

    def __getitem__(self, index):
        """ Call by DataLoader(an Iterator) """
        image_path, class_name = self.data[index]

        # Label (Type is torch.LongTensor for calculate loss)
        label = torch.LongTensor([self.class2idx[class_name]])

        # Load image and transform
        image = Image.open(PJ(self.data_root, image_path)).convert('RGB')
        image = self.transform(image) if self.transform else image
        return label, image
