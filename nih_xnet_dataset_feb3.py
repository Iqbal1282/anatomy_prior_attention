'''
    Image + Abnorm Mask + Lung Mask + Label
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

def get_image_transform():
    return A.Compose([
        A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255,
                p=1.0,
            )
    ], p=1.0)

def get_train_transforms():      
    return A.Compose([
            # A.Resize(width=CROP, height=CROP, p=1.0),
            A.Resize(width=RESIZE, height=RESIZE, p=1.0),
            A.RandomCrop(width=CROP, height=CROP, p=1.0), 
            # A.GaussianBlur(blur_limit=(3,5), p=0.5),
            # A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.1, rotate_limit=0, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),         
            # A.HorizontalFlip(p=0.5), 
            # A.VerticalFlip(p=0.5),  
            ToTensorV2(always_apply=True, p=1.0),
    ], p=1.0, additional_targets={
        'mask1' : 'mask', 'mask2' : 'mask', 'mask3' : 'mask', 'mask4' : 'mask', 'mask5' : 'mask', 'mask6' : 'mask', 'mask7' : 'mask',
        'mask8' : 'mask', 'mask9' : 'mask', 'mask10' : 'mask', 'mask11' : 'mask', 'mask12' : 'mask', 'mask13' : 'mask', 'mask14' : 'mask', 'mask15' : 'mask'
    })

def get_valid_transforms():      
    return A.Compose([
            # A.Resize(width=CROP, height=CROP, p=1.0),
            A.Resize(width=RESIZE, height=RESIZE, p=1.0),
            A.CenterCrop(width=CROP, height=CROP, p=1.0),
            ToTensorV2(always_apply=True, p=1.0),
    ], p=1.0, additional_targets={
        'mask1' : 'mask', 'mask2' : 'mask', 'mask3' : 'mask', 'mask4' : 'mask', 'mask5' : 'mask', 'mask6' : 'mask', 'mask7' : 'mask',
        'mask8' : 'mask', 'mask9' : 'mask', 'mask10' : 'mask', 'mask11' : 'mask', 'mask12' : 'mask', 'mask13' : 'mask', 'mask14' : 'mask', 'mask15' : 'mask'
    })


class ChestXrayDataSet_maskfused(Dataset):
    '''
    Label + Abnormality Mask + Lung Mask
    '''
    def __init__(self, data_dir, image_list_file, num_class=15, transform=None, mask_resize_dim=7):
        image_names = []
        labels = []
        mask_names = []
        mask_dir = MASK_DIR
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                label = items[1:]
                label = [int(i) for i in label]
                if num_class==15:
                    label.append(0 if sum(label) else 1)
                mask_name = os.path.join(mask_dir, image_name)
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                mask_names.append(mask_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.abnorm_mask = np.load('abnorm_mask_full.npy')
        self.image_transform = get_image_transform()

        self.mask_names = mask_names
        self.transform = transform
        
    def __getitem__(self, index):
        image_name = self.image_names[index]
        mask_name = self.mask_names[index]
            
        #read image, mask
        image = Image.open(image_name).convert('RGB')
        image = np.array(image)
        masks = self.abnorm_mask
        lung_mask = np.array(Image.open(mask_name), dtype=np.float32)
         
        image = self.image_transform(image=image)['image']    
        transformed = self.transform(image=image, mask = masks[0], mask1 = masks[1], mask2 = masks[2], mask3 = masks[3], mask4 = masks[4], mask5 = masks[5], mask6 = masks[6],
                                     mask7 = masks[7], mask8 = masks[8], mask9 = masks[9], mask10 = masks[10], mask11 = masks[11], mask12 = masks[12], mask13 = masks[13], mask14 = masks[14],
                                     mask15 = lung_mask)

        transformed_image = transformed['image']           
        transformed_masks = np.stack((transformed['mask'], transformed['mask1'], transformed['mask2'], transformed['mask3'], transformed['mask4'], transformed['mask5'], transformed['mask6'],
                            transformed['mask7'], transformed['mask8'], transformed['mask9'], transformed['mask10'], transformed['mask11'], transformed['mask12'], transformed['mask13'],
                            transformed['mask14'])
                            )
        lung_mask = transformed['mask15'].unsqueeze(dim=0)
        
        label = self.labels[index]
                  
        return transformed_image, transformed_masks, lung_mask, torch.FloatTensor(label)
    
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