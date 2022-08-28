'''
    Image + Abnorm Mask  + Lung Mask + Label 
    Author: Self (Not from Rafi vai)
'''

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
import os
import pandas as pd
import torch
from ast import literal_eval
from albumentations.pytorch.transforms import ToTensorV2


MASK_DIR = '/home/mhealth-14/Thesis_400_CXR/iqbal/codebase/HAM/nih_dataset_masks'
RESIZE=256
CROP=224

def get_image_transform():
    return A.Compose([
        A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255,
                p=1.0,
            )
    ], p=1.0)


def get_valid_transforms():      
    return A.Compose([
            A.Resize(width=RESIZE, height=RESIZE, p=1.0),
            A.CenterCrop(width=CROP, height=CROP, p=1.0),
            ToTensorV2(always_apply=True, p=1.0),
    ], p=1.0, additional_targets={
        'mask1' : 'mask', 'mask2' : 'mask', 'mask3' : 'mask', 'mask4' : 'mask',
        'mask5' : 'mask', 'mask6' : 'mask', 'mask7' : 'mask', 'mask8' : 'mask'
    })


class LocalizationDataset(Dataset):
    def __init__(self, csv_path):
        self.pathology_list = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate', 'Mass',
                                 'Nodule', 'Pneumonia', 'Pneumothorax']
        self._df = pd.read_csv(csv_path)
        self.abnorm_mask = np.load('abnorm_mask_full.npy')
        self.image_transform = get_image_transform()
        self.transform = get_valid_transforms()

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):
        df_row = self._df.iloc[index]
        image_path = df_row.path
        assert os.path.exists(image_path), image_path

        class_labels = np.zeros(len(self.pathology_list), dtype=np.float32)
        bboxes = np.zeros((len(self.pathology_list), 4), dtype = np.uint8)

        present_pathologies = literal_eval(df_row['Finding Label'])
        x1_list = [float(elem) for elem in literal_eval(df_row['Bbox_x'])]
        y1_list = [float(elem) for elem in literal_eval(df_row['Bbox_y'])]
        width_list = [float(elem) for elem in literal_eval(df_row['Bbox_w'])] 
        height_list = [float(elem) for elem in literal_eval(df_row['Bbox_h'])] 

        for pathology in present_pathologies:
            if pathology in self.pathology_list:
                idx = present_pathologies.index(pathology)
                disease_idx = self.pathology_list.index(pathology)

                class_labels[disease_idx] = 1
                x1 = int(x1_list[idx] * CROP / 1024)
                y1 = int(y1_list[idx] * CROP / 1024)
                x2 = int(x1 + (width_list[idx] * CROP / 1024))
                y2 = int(y1 + (height_list[idx] * CROP / 1024))
                bboxes[disease_idx] = [x1, y1, x2, y2]


        #read image, mask
        image = np.array(Image.open(image_path).convert('RGB'))
        base_name = os.path.basename(image_path)
        mask_path = os.path.join(MASK_DIR, base_name)
        masks = self.abnorm_mask
        lung_mask = np.array(Image.open(mask_path), dtype=np.float32)
         
        image = self.image_transform(image=image)['image']    
        transformed = self.transform(image=image, mask = masks[0], mask1 = masks[1], mask2 = masks[2], mask3 = masks[3],
                                     mask4 = masks[4], mask5 = masks[5], mask6 = masks[6], mask7 = masks[7], mask8 = lung_mask)

        transformed_image = transformed['image']           
        transformed_masks = np.stack((transformed['mask'], transformed['mask1'], transformed['mask2'], transformed['mask3'],
                                    transformed['mask4'], transformed['mask5'], transformed['mask6'], transformed['mask7']))
                                    
        lung_mask = transformed['mask8'].unsqueeze(dim=0)
        
                  
        return transformed_image, transformed_masks, lung_mask, torch.FloatTensor(class_labels), torch.FloatTensor(bboxes)
      

def get_test_localization_ds():
    localization_csv = "/home/mhealth-14/Thesis_400_CXR/iqbal/codebase/HAM/NIH_dataset/localization_label_train.csv"
    return LocalizationDataset(csv_path=localization_csv)