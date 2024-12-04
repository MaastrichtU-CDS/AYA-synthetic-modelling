import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Rename columns
# Encoder


class DataPreProcessing: 
    def __init__(self, df: pd.DataFrame):

        """

        Function to preprocess the data for the model.
        - Raname columns for better readability.
        - Encode categorical columns using a OneHotEncoder.
        - Split the data into training and testing sets.

        Input:
            - df (pd.DataFrame): The DataFrame to preprocess. A copy of the original DataFrame is made to avoid altering the original data.

        self.X: SD minus the target column.
        self.y: The target column: Negative Body Image.

        Note: Test data is Original Data
        
        """

        self.df = df.copy()
        if 'leeft' in self.df.columns:
            self.df.drop(columns='leeft', inplace=True)

        self.X = None
        self.y = None


    def Rename_Cols(self, df:pd.DataFrame):

        """

        Function to rename columns of the DataFrame.
       
        """

        columns_to_rename = { 
            "ql": "Global Health Status",
            "pf": "Physical Functioning",
            "sf": "Social Functioning",
            "ef": "Emotional Functioning",
            "cf": "Cognitive Functioning",
            "rf": "Role Functioning",
            "maladaptive": "Maladaptive Coping Style",
            "nsurv94dich": "Sexual Attractiveness",
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
        self.df.info()
        
        return self.df

    def Cat_Encoder(self, df:pd.DataFrame)-> pd.DataFrame:

        """

        Function to:

            1. Transform categorical columns using a OneHotEncoder.
            2. Transform and refit the model with the one-hot-encoded data.

        Input:
            - refs (dict): A dictionary containing categorical columns and reference values established in original study. Reference values dropped.
                
        Returns:
            - df_encoded (pd.DataFrame): The new DataFrame with relevant columns encoded.
        
        """

        refs = {
            "Education Level": 3.0,
            "Tumour Type": 1.0,
            "Tumour Stage": 1.0,
            "Time Since Diagnosis": 3.0,
            "BMI Group": 1.0,
            "Sexual Attractiveness": 0.0,
            "Partner": 1.0,
            "Biological Sex": 0.0,
            "Targeted Therapy": 0.0,
            "Hormone Therapy": 0.0,
            "Radiation Therapy": 0.0,
            "Chemotherapy": 0.0
        }

        # Get categorical columns and their reference values to drop
        cat_cols = list(refs.keys()) # Get the categorical columns
        refs_to_drop = [refs[col] for col in cat_cols] # Get the reference values to drop  
        print("DROP SPECS:", refs_to_drop)
        
        # Create a OneHotEncoder with drop specifications
        ohe = OneHotEncoder(drop =refs_to_drop, sparse_output=False, handle_unknown='ignore')

        # Apply the encoder to the specified  columns
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

        # Rename encoded columns for better readability
        encoded_cols = {
            "Education Level_0.0": "No/Primary Education_1.0",
            "Education Level_1.0": "Secondary Education_2.0",
            "Tumour Type_2.0": "Head/Neck Tumour_2.0",
            "Tumour Type_3.0": "Digestive Tumour_3.0",
            "Tumour Type_4.0": "Colon and Rectal Tumour_4.0",
            "Tumour Type_5.0": "Bone and STS Tumour_5.0",
            "Tumour Type_6.0": "Other Tumour_6.0",
            "Tumour Type_7.0": "Germ Cell Tumour_7.0",
            "Tumour Type_8.0": "Breast Tumour_8.0",
            "Tumour Type_9.0": "Female Genitalia Tumour_9.0",
            "Tumour Type_10.0": "Lymph Tumour_10.0",
            "Tumour Type_11.0": "Myeloid Tumour_11.0",
            "Tumour Type_12.0": "Thyroid Gland Tumour_12.0",
            "Tumour Type_13.0": "CNS Tumour_13.0",
            "Tumour Stage_2.0": "Tumour Stage II_2.0",
            "Tumour Stage_3.0": "Tumour Stage III_3.0",
            "Tumour Stage_4.0": "Tumour Stage IV_4.0",
            "Tumour Stage_99.0": "Missing Tumour Data_99",
            "Time Since Diagnosis_1.0": "TSD >10y_1.0",
            "Time Since Diagnosis_2.0": "TSD 10-15y_2.0",
            "BMI Group_2.0": "BMI Underweight_2.0",
            "BMI Group_3.0": "BMI Overweight_3.0",
            "BMI Group_4.0": "BMI Obese_4.0", 
        }

        self.df = self.df.rename(columns=encoded_cols)

        self.X = self.df.drop(columns=["Negative Body Image"])
        print("X INFO")
        self.X.info()

        self.y = self.df["Negative Body Image"]
        print("Y INFO")
        self.y.info()

        return self.df, self.X, self.y


if __name__ == "__main__":
    df_file_path = input("Provide df file path: ")
    df = pd.read_csv(df_file_path, delimiter=',', encoding='latin1')
    
    pre_process_data = DataPreProcessing(df)
    pre_process_data.Rename_Cols(df)
    pre_process_data.Cat_Encoder(df)




    





