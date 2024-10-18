import logging
import os 

import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder


# Data exploration
# Rename columns
# Encoder
# Split

# Import data

# df_og = pd.read_csv(model_file_path, delimiter=',', encoding='latin1')
# df = pd.DataFrame(df_og.copy())
# df.drop(columns='leeft', inplace=True)

class DataPreProcessing: 
    def __init__(self, df: pd.DataFrame):
        self.df = pd.DataFrame(pd.read_csv(input("Model file path please: "), delimiter=',', encoding='latin1').copy())
        if 'leeft' in self.df.columns:
            self.df.drop(columns='leeft', inplace=True)
        self.X = None
        self.y = None

    def Data_Exploration(self):
        print("\n Original data before processing \n")
        print(self.df.head())
        print(self.df.info())
        print(self.df.describe())

    def Rename_Col(self):
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

        self.df.rename(columns=columns_to_rename, inplace=True)
        self.X = df.drop(columns=["Negative Body Image"])
        self.y = df["Negative Body Image"]

        print("Column rename check: ", self.df.info())

    return self.df


# Usage example
# df = pd.read_csv('your_data.csv')
# processor = DataPreProcessor(df)
# renamed_df = processor.rename_columns()







# def Cat_Encoder():

#     """

#     Function to transform categorical columns using a OneHotEncoder.

#     1. Uses a dictionary (refs) to indicate categorical columns and their reference values to drop using a OneHotEncoder.
#     2. Transforms and refits the model with the encoded data.

#     Global Variables:
#     - refs (dict): A dictionary containing categorical columns and their reference values to drop.
#     - df_encoded (pd.DataFrame): The (new) DataFrame containing the encoded data.
    
#     Returns:
#     - df_encoded (pd.DataFrame): The DataFrame with categorical columns encoded and other columns unchanged.
   
#     # Binary features are not encoded.

#     """

#     global refs, df_encoded, X

#     refs = {
#         "Education Level": 3.0, #FIXME: Given data, no 2.0.
#         "Tumour Type": 1.0,
#         "Tumour Stage": 1.0,
#         "Time Since Diagnosis": 3.0,
#         "BMI Group": 1.0,
#     }

#     # Extract categorical columns and their reference values to drop
#     cat_columns = list(refs.keys()) # Get the categorical columns
#     refs_to_drop = [refs[col] for col in cat_columns] # Get the reference values to drop  
#     print("DROP SPECS:", refs_to_drop)
    
#     # Create a OneHotEncoder with drop specifications
#     ohe = OneHotEncoder(drop=refs_to_drop, sparse_output=False, handle_unknown='ignore')
    
#     # Apply the encoder to the specified categorical columns
#     column_transformer = ColumnTransformer(
#         transformers=[
#             ('ohe', ohe, cat_columns)  
#         ], remainder='passthrough'  # Keep other columns unchanged
#     )
    
#     # Fit and transform the DataFrame
#     df_transformed = column_transformer.fit_transform(df)
#     print("DF (X) TRANSFORMED:", df_transformed)
    
#     # Get the transformed column names
#     ohe_feature_names = column_transformer.named_transformers_['ohe'].get_feature_names_out(cat_columns)
#     print("FT NAMES: ", ohe_feature_names)
    
#     # Convert the result back to a DataFrame
#     df_encoded = pd.DataFrame(df_transformed, columns=np.append(ohe_feature_names, df.drop(columns=cat_columns).columns))
#     print("\n Encoded df:", df_encoded.info())

#     return df_encoded


# # Cat_Encoder()


# def Ttv_Split():
#     """

#     Train Test Val (Ttv) Split Function
    
#     1. Split the Data into training, validation, and test sets. 
#         Split:, 80% training, 20% validation, and 10% test.
#     2. Scale the data.
#     3. Put data into dataframe type.

#     """

#     global X, y, X_train, X_test, X_val, y_val, y_train, y_test

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
#     print(f'\n X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, \n X_test shape: {X_test.shape}, y_test shape: {y_test.shape}, \n X_val shape: {X_val.shape}, y_val shape: {y_val.shape}')

#     X_train = pd.DataFrame(X_train, columns=X.columns)
#     X_val = pd.DataFrame(X_val, columns=X.columns)
#     X_test = pd.DataFrame(X_test, columns=X.columns)

#     return  X_train, X_test, X_val, y_val, y_train, y_test
