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

    def Rename_Cols(self, df:pd.DataFrame):

        """

        Function to rename columns of the DataFrame (copy).
       
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
            - refs (dict): A dictionary containing categorical columns and their reference values to drop.
                
        Returns:
            - df_encoded (pd.DataFrame): The new DataFrame with categorical columns encoded and other columns unchanged.
        
        Binary features are not encoded.

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

        # Extract categorical columns and their reference values to drop
        cat_cols = list(refs.keys()) # Get the categorical columns
        refs_to_drop = [refs[col] for col in cat_cols] # Get the reference values to drop  
        print("DROP SPECS:", refs_to_drop)
        
        # Create a OneHotEncoder with drop specifications
        ohe = OneHotEncoder(drop =refs_to_drop, sparse_output=False, handle_unknown='ignore')

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
            "BMI Group_4.0": "BMI Obese_4.0", # Unsure if to still keep as Obese; otherwise, Excess Weight   
        }

        self.df = self.df.rename(columns=encoded_cols)

        self.X = self.df.drop(columns=["Negative Body Image"])
        print("X INFO")
        self.X.info()

        self.y = self.df["Negative Body Image"]
        print("Y INFO")
        self.y.info()

        return self.df, self.X, self.y


    def Ttv_Split(self) -> pd.DataFrame:
        
        """

        Train Test Val (Ttv) Split Function
        
            - Split the Data into training, validation, and test sets. 
                Current split:, 70% training, 20% validation, and 10% test.
        
        Input:
            - Splits: A dictionary containing the test size and validation size.
                Key: test_size.val_size
                Value: [test_size, val_size]
                #FIXME: Is this the best way to do this?
                #FIXME: Need to save which split was chosen for reproducibility. 

        Returns the split data as dataframes.

        """
        
        splits = {
            1.1: [0.1, 0.1], # 80% train; 10% test, 10% val 
            1.2: [0.1, 0.2], # 70% train; 10% test, 20% val
            1.3: [0.1, 0.3], # 60% train; 10% test, 30% val
            2.1: [0.2, 0.1], # 70% train; 20% test, 10% val
            2.2: [0.2, 0.2] # 60% train; 20% test, 20% val
        }

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=splits[1.2][0], random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=splits[1.2][1], random_state=42)
        
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



    





