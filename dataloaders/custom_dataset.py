import ast
import time
import os
from skimage import io, transform
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataloaders.data_utils import get_unk_mask_indices
from sklearn.preprocessing import MultiLabelBinarizer

def label_encoded(df):
    # unique label이 이미 있는 데이터 셋 가정 

    mlb = MultiLabelBinarizer()
    encoded_label = []
    
    for i in range(len(df)):

        str_label = []

        for i in ast.literal_eval(df.loc[i, 'unique_label']):
            str_label.append(str(i))

        encoded_label.append(str_label)
        
    mlb_label = mlb.fit_transform(encoded_label)
    num_labels = len(mlb.classes_)

    df['encoded'] = [i.tolist() for i in mlb_label]
    
    return df, num_labels


def custom_image_loader(path, transform):
    try:
        image = Image.open(path)
    except FileNotFoundError: # weird issues with loading images on our servers
        # print('FILE NOT FOUND')
        time.sleep(10)
        image = Image.open(path)

    if transform is not None:
        image = transform(image)
    
    image = torch.cat([image, image, image], dim=0)

    return image


class CustomDataset(Dataset):

    def __init__(self, df, split, image_transform, root_dir = 'F:/FID-300/FID-300/references', known_labels=0, testing=False):
        self.df, self.num_labels = label_encoded(df)
        self.image_transform = image_transform
        self.root_dir = root_dir
        self.split = split
        self.known_labels = known_labels
        self.testing=testing

        # Dataset split
        if self.split == 'train':
            self.df = self.df[:int(len(self.df) * 0.9)]
        elif self.split == 'test':
            self.df = self.df[int(len(self.df) * 0.9):]
        else:
            print(f'!!! Invalid split {self.split}... !!!')

        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        label = self.df.iloc[idx, 4]
        label = torch.tensor(label, dtype=torch.float32)

        name = str(self.df.iloc[idx, 1])
        image_path = os.path.join(self.root_dir, name)
        image = custom_image_loader(image_path, self.image_transform)

        unk_mask_indices = get_unk_mask_indices(image, self.testing, self.num_labels, self.known_labels)
        mask = label.clone()
        mask.scatter_(0,torch.Tensor(unk_mask_indices).long() , -1)

        sample = {}
        sample['image'] = image
        sample['labels'] = label
        sample['mask'] = mask
        sample['imageIDs'] = name    

        return sample

