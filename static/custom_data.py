import sys,os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from src.human.exception import CustomException
from src.human.entity.config_entity import *

class CustomData(Dataset):
    try:
        def __init__(self,df, augmentations, prediction_config : PredictionPipelineConfig):
            self.df = df
            self.augmentations = augmentations
            self.prediction_config = prediction_config
            
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self,idx):
            row = self.df.iloc[idx]
            image_path = os.path.join(self.prediction_config.validation_path, row.images)
            mask_path = os.path.join(self.prediction_config.validation_path,row.masks)
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # this have 2 dim (h,w)
            # we need add new dim (h, w, c)
            mask = np.expand_dims(mask, axis= -1)
            
            if self.augmentations:
                data = self.augmentations(image = image, mask = mask)
                image = data['image']
                mask = data['mask']
                
            #(h, w, c) -> (c, h, w)
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)
            
            # convert from np to tensor
            image = torch.Tensor(image) / 255.0
            mask = torch.round(torch.Tensor(mask) / 255.0)
            
            return image, mask
        
    except Exception as e:
        raise CustomException(e, sys)