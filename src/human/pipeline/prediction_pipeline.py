import os,sys
import pandas as pd
import matplotlib.pyplot as plt
from src.human.logger import logging
from src.human.exception import CustomException
from src.human.constants import *
from src.human.utils import *
from src.human.entity.config_entity import *
import albumentations as A
from src.human.cloud_storage.s3_opearations import S3Sync
from src.human.entity.custom_model import SegmentationModel
from static.custom_data import CustomData

DEVICE = get_default_device()

class SinglePrediction:
    def __init__(self):
        try:
            self.s3_sync = S3Sync()
            self.prediction_config = PredictionPipelineConfig()
        except Exception as e:
            raise CustomException(e, sys)
        
    def get_model_in_production(self):
        try:
            model_download_dir = self.prediction_config.model_download_path
            os.makedirs(model_download_dir,exist_ok=True)
            logging.info(f"Model Directory created at {model_download_dir}")
            #Model Directory is empty
            if not any(os.scandir(model_download_dir)):
                self.s3_sync.sync_folder_from_s3(folder=model_download_dir, aws_bucket_url=S3_BUCKET_MODEL_URI)

            validation_dir = self.prediction_config.validation_path
            #Validation Directory is empty
            os.makedirs(validation_dir,exist_ok=True)
            logging.info(f"Validation Directory created at {validation_dir}")
            if not any(os.scandir(validation_dir)):
                self.s3_sync.sync_folder_from_s3(folder=validation_dir, aws_bucket_url=S3_BUCKET_VALIDATION_URI)

        except Exception as e:
            raise CustomException(e, sys)
        
    def get_valid_augmentations(self):
        return A.Compose([
                            A.Resize(IMAGE_SIZE, IMAGE_SIZE)
        ])
        
    def predict(self, filename):
        try:
            self.get_model_in_production() 
            model_download_dir = self.prediction_config.model_download_path
            os.makedirs(model_download_dir,exist_ok=True)
            validation_dir = self.prediction_config.validation_path
            os.makedirs(validation_dir,exist_ok=True)
            prediction_model_path = self.prediction_config.model_path
            model = SegmentationModel()
            model = to_device(model, DEVICE)
            model.load_state_dict(torch.load(prediction_model_path, map_location=torch.device('cpu')))

            # find Index of row in dataframe
            
            df = pd.read_csv(self.prediction_config.csv_file_path)
            df.columns = df.columns.str.replace('\t\t', '')
            df['images'] = df['images'].str.replace('\t\t', '')
            image_name = "Training_Images/"+ filename
            found_df_row = df['images'].str.find(image_name)
            index_value = df.index[found_df_row==0].to_list()

            # prepare data for prediction
            validset = CustomData(df, self.get_valid_augmentations(), self.prediction_config)
            
            image, mask = validset[index_value[0]]
            logits_mask = model(image.to(DEVICE).unsqueeze(0)) # (C, H, W) -> (1, C, H, W)
            pred_mask = torch.sigmoid(logits_mask)
            pred_mask = (pred_mask > 0.5) * 1.0
       
            predict_dir = os.path.join(os.getcwd(),STATIC_DIR,PREDICT_SUB_DIR)
            os.makedirs(predict_dir,exist_ok=True)
            predict_img_path = os.path.join(os.getcwd(),STATIC_DIR,PREDICT_SUB_DIR,PREDICT_IMG_NAME)
            x = pred_mask.detach().cpu().squeeze(0).permute(1,2,0)
            x = x.numpy()
            pred = np.repeat(x, 3, 2)
            plt.imsave(predict_img_path,pred)

        except Exception as e:
            raise CustomException(e, sys)