import logging
import os 

import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Data exploration
# Rename columns
# Encoder
# Split

class DataPreProcessing: 
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if 'leeft' in self.df.columns:
            self.df.drop(columns='leeft', inplace=True)

        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_val = None
        self.y_train = None
        self.y_test = None

    def Data_Exploration(self):
        print("\n Original data before processing \n")
        print(self.df.head())
        print(self.df.info())
        print(self.df.describe())

    def Rename_Cols(self, df): #FIXME: Should this be _ first? 
        #FIXME: Specify return type
        columns_to_rename = { 
            "ql": "Global Health Status",
            "pf": "Physical Functioning",
            "sf": "Social Functioning",
            "ef": "Emotional Functioning",
            "cf": "Cognitive Functioning",
            "rf": "Role Functioning",
            "maladaptive": "Maladaptive Coping Style",
            "nsurv94dich": "Sexual Attractiveness", # self-perception of own sexual attractiveness
            "Targetnew": "Targeted Therapy",
            "Hormnew": "Hormone Therapy",
            "RTnew": "Radiation Therapy",
            "chemonew": "Chemotherapy",
            "partner": "Partner",
            "alg_v1b": "Biological Sex",
            "stadium_opgeschoond": "Tumour Stage",
            "tumsoort_hoofd_new2": "Tumour Type",
            "edu": "Education Level",
            "tijddiagngroep": "Time Since Diagnosis",
            "BMIgroups2": "BMI Group",
            "BIneg_new": "Negative Body Image"
        }

        self.df = self.df.rename(columns=columns_to_rename)
        print("\n Renamed columns in df: \n")
        self.df.info() # Call info() directly to print its output
        
        return self.df

    def Cat_Encoder(self, df):

        """

        Function to transform categorical columns using a OneHotEncoder.

        1. Uses a dictionary (refs) to indicate categorical columns and their reference values to drop using a OneHotEncoder.
        2. Transforms and refits the model with the encoded data.

        Global Variables:
        - refs (dict): A dictionary containing categorical columns and their reference values to drop.
        - df_encoded (pd.DataFrame): The (new) DataFrame containing the encoded data.
        
        Returns:
        - df_encoded (pd.DataFrame): The DataFrame with categorical columns encoded and other columns unchanged.
    
        # Binary features are not encoded.

        """

        refs = {
            "Education Level": 3.0, #FIXME: Given data, no 2.0.
            "Tumour Type": 1.0,
            "Tumour Stage": 1.0,
            "Time Since Diagnosis": 3.0,
            "BMI Group": 1.0,
        }

        # Extract categorical columns and their reference values to drop
        cat_cols = list(refs.keys()) # Get the categorical columns
        refs_to_drop = [refs[col] for col in cat_cols] # Get the reference values to drop  
        print("DROP SPECS:", refs_to_drop)
        
        # Create a OneHotEncoder with drop specifications
        ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

        # Apply the encoder to the specified categorical columns
        cat_col_transformer = ColumnTransformer(
            transformers=[
                ('ohe', ohe, cat_cols)  
            ], remainder='passthrough'  # Keep other columns unchanged
        )
        
        # Fit and transform the DataFrame
        df_transformed = cat_col_transformer.fit_transform(self.df)

        # Get the transformed column names
        ohe_feature_names = cat_col_transformer.named_transformers_['ohe'].get_feature_names_out(cat_cols)
        print("Encoded Categorical Features: ", ohe_feature_names)
        
        # Convert the result back to a DataFrame
        self.df = pd.DataFrame(df_transformed, columns=np.concatenate([ohe_feature_names, self.df.drop(columns=cat_cols).columns]))
        print("\n Encoded df: \n")
        self.df.info()

        self.X = self.df.drop(columns=["Negative Body Image"])
        print("X INFO")
        self.X.info()

        self.y = self.df["Negative Body Image"]
        print("Y INFO")
        self.y.info()

        return self.df, self.X, self.y


    def Ttv_Split(self) -> pd.DataFrame:
        """

        #TODO: Maybe we will select different training and test sizes so set up for that. 

        Train Test Val (Ttv) Split Function
        
        1. Split the Data into training, validation, and test sets. 
            Split:, 80% training, 20% validation, and 10% test.
        2. Put data into dataframe type.

        """
        train_split_size = 0.8
        test_split_size = 0.1
        val_split_size = 0.2
        # TODO: maybe vars of percentages to multiply for test and val sets.

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_split_size, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=val_split_size, random_state=42)
        
        print(f'\n X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}, \n X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}, \n X_val shape: {self.X_val.shape}, y_val shape: {self.y_val.shape}')

        return self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val 

if __name__ == "__main__":
    df_file_path = input("Provide df file path: ")
    df = pd.read_csv(df_file_path, delimiter=',', encoding='latin1')
    
    pre_process_data = DataPreProcessing(df)
    pre_process_data.Data_Exploration()
    pre_process_data.Rename_Cols(df)
    pre_process_data.Cat_Encoder(df)
    pre_process_data.Ttv_Split()

    





