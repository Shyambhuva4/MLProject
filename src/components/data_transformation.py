import os
import sys
from dataclasses import  dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

@dataclass
class DataTranformationConfig:
    preprocessor_onj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTranformationConfig()

    def get_data_tranformed_obj(self):

        """
        responsible for data transformation
        """
        try:
            numerical_column=['reading_score', 'writing_score']
            categorical_column= ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy='most_frequent')),
                ("hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info("numerical and categorical encoding is completed")

            preproccessor=ColumnTransformer([
                ("numerical pipeline",num_pipeline,numerical_column),
                ("categorical pipeline",cat_pipeline,categorical_column)
            ])
            
        

            return preproccessor
        
        except Exception as e:
            raise CustomException(e,sys)
        


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test is completed")

            logging.info("obtaining preprocessor object")

            preprocessing_obj=self.get_data_tranformed_obj()
            target_column="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column])
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=[target_column])
            target_feature_test_df=test_df[target_column]
            
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_onj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_onj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)