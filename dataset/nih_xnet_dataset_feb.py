'''
    Image + Label
'''

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
      
TRAIN_IMAGE_LIST = './dataset/train_list.txt'
VAL_IMAGE_LIST = './dataset/val_list.txt'
TEST_IMAGE_LIST = './dataset/test_list.txt'

TRAIN_DATA_DIR = '/home/mhealth-14/Thesis_400_CXR/iqbal/codebase/HAM/nih_dataset_images'
VAL_DATA_DIR = '/home/mhealth-14/Thesis_400_CXR/iqbal/codebase/HAM/nih_dataset_images'
TEST_DATA_DIR = '/home/mhealth-14/Thesis_400_CXR/iqbal/codebase/HAM/nih_dataset_images'

MASK_DIR = '/home/mhealth-14/Thesis_400_CXR/iqbal/codebase/HAM/nih_dataset_masks'

RESIZE = 256
CROP = 224 
N_CLASSES = 15
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia',
                'No_Finding']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_train_transforms():      
    return A.Compose([
            A.Resize(width=RESIZE, height=RESIZE, p=1.0),
            A.RandomCrop(width=CROP, height=CROP, p=1.0),
            # A.HorizontalFlip(p=0.5), 
            # A.VerticalFlip(p=0.5),  
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255,
                p=1.0,
            ),
            ToTensorV2(always_apply=True, p=1.0),
        ], p=1.0)

def get_valid_transforms():      
    return A.Compose([
            A.Resize(width=RESIZE, height=RESIZE, p=1.0),
            A.CenterCrop(width=CROP, height=CROP, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255,
                p=1.0,
            ),
            ToTensorV2(always_apply=True, p=1.0),
        ], p=1.0)

class ChestXrayDataSet_maskfused(Dataset):
    def __init__(self, data_dir, image_list_file, num_class=15, transform=None, mask_resize_dim=84):
        image_names = []
        labels = []
        # mask_names = []
        # mask_dir = MASK_DIR
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                label = items[1:]
                label = [int(i) for i in label]
                if num_class==15:
                    label.append(0 if sum(label) else 1)
                # mask_name = os.path.join(mask_dir, image_name)
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                # mask_names.append(mask_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        # self.mask_names = mask_names
        self.transform = transform
        # self.mask_transform =  A.Compose([
        #                             A.Resize(width=mask_resize_dim, height=mask_resize_dim, p=1.0),
        #                             ToTensorV2(always_apply=True, p=1.0),
        #                         ], p=1.0)
        
    def __getitem__(self, index):
        image_name = self.image_names[index]
        # mask_name = self.mask_names[index]
            
        #read image, mask
        image = Image.open(image_name).convert('RGB')
        image = np.array(image)
        # mask = cv2.imread(mask_name)
        # mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)  
         
        # masks = [mask[:,:,0], mask[:,:,1]]       
        transformed = self.transform(image=image) # transformed = self.transform(image=image, masks=masks)
        transformed_image = transformed['image']
        # transformed_masks = transformed['masks']
        
        # masks = np.stack([transformed_masks[0], transformed_masks[1], np.zeros_like(transformed_masks[0])], axis=2)               
        # transformed_masks = self.mask_transform(image=masks)['image']
        # transformed_masks = (transformed_masks/255).float()
        
        label = self.labels[index]
                  
        return transformed_image, torch.FloatTensor(label) #return transformed_image, transformed_masks, torch.FloatTensor(label)
    
    def __len__(self):
        return len(self.image_names)


def get_datasets():
    train_dataset = ChestXrayDataSet_maskfused(
                        data_dir=TRAIN_DATA_DIR,
                        image_list_file=TRAIN_IMAGE_LIST,
                        num_class=N_CLASSES, 
                        transform=get_train_transforms(),
                        mask_resize_dim=56,
                        )
    valid_dataset = ChestXrayDataSet_maskfused(
                        data_dir=VAL_DATA_DIR,
                        image_list_file=VAL_IMAGE_LIST,
                        num_class=N_CLASSES, 
                        transform=get_valid_transforms(),
                        mask_resize_dim=56,
                        )
    test_dataset = ChestXrayDataSet_maskfused(
                        data_dir=TEST_DATA_DIR,
                        image_list_file=TEST_IMAGE_LIST,
                        num_class=N_CLASSES, 
                        transform=get_valid_transforms(),
                        mask_resize_dim=56,
                        )
    return train_dataset, valid_dataset, test_dataset


if __name__ == '__main__':
    train_dataset = ChestXrayDataSet_maskfused(
                        data_dir=TRAIN_DATA_DIR,
                        image_list_file=TRAIN_IMAGE_LIST,
                        num_class=N_CLASSES, 
                        transform=get_valid_transforms(),
                        mask_resize_dim=56,
                        )
    image, label = train_dataset[16]
    plt.imshow(image.permute(1, 2, 0));plt.show();
    # plt.imshow(masks[0]);plt.show();
    # plt.imshow(masks[1]);plt.show();