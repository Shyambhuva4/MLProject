import pandas as pd
from src.exception import CustomException
from src.logger import logging
import os
import sys

from src.components.data_transformation import DataTransformation 
from src.components.data_transformation import DataTranformationConfig
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
##logger
# from datetime import datetime

# LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# logs_path=os.path.join(os.getcwd(),'logs',LOG_FILE)
# os.makedirs(logs_path,exist_ok=True)

# LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

# logging.basicConfig(
# filename=LOG_FILE_PATH,
# format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelno)s - %(message)s',
# level=logging.INFO
# )

@dataclass
class DataIngestionCofig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self) :
        self.ingestion_config=DataIngestionCofig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion")

        try:
         df=pd.read_csv('src/notebook/data/stud.csv')
         logging.info('Read dataset')

         os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

         df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
         logging.info("Split initiated")

         train_set,test_set=train_test_split(df,test_size=0.20,random_state=42)
         train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
         test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

         logging.info('Data ingestion is completed')

         return (self.ingestion_config.train_data_path,
                 self.ingestion_config.test_data_path)


        except Exception as e:
           raise CustomException(e,sys)
        
if __name__=='__main__':
   obj=DataIngestion()
   train_data,test_data=obj.initiate_data_ingestion()

   data_tranformation=DataTransformation()   
   
   train_arr,test_arr,_=data_tranformation.initiate_data_transformation(train_data,test_data)

   modeltrainer=ModelTrainer()
   print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
