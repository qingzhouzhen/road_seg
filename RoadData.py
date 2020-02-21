import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

import cv2
from onehot import onehot

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class RoadDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        
    def __len__(self):
        return len(os.listdir('road_data'))

    def __getitem__(self, idx):
        img_name = os.listdir('road_data')[idx]
        imgA = cv2.imread('road_data/'+img_name)
        imgA = cv2.resize(imgA, (640, 640))
        imgB = cv2.imread('road_data_msk/'+img_name, 0)
        imgB = cv2.resize(imgB, (640, 640))
        imgB = imgB/255
        imgB = imgB.astype('uint8')
        imgB = onehot(imgB, 2)
        imgB = imgB.transpose(2,0,1)
        imgB = torch.FloatTensor(imgB)
        #print(imgB.shape)
        if self.transform:
            imgA = self.transform(imgA)    

        return imgA, imgB

road = RoadDataset(transform)

train_size = int(1 * len(road))
test_size = len(road) - train_size
train_dataset, test_dataset = random_split(road, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
# test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)


if __name__ =='__main__':

    for train_batch in train_dataloader:
        print(train_batch)

    # for test_batch in test_dataloader:
    #     print(test_batch)
