import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class DataPreProcessing: 
    """
    Class to preprocess the data to feed into the models.
    """

    def __init__(self, train_df:pd.DataFrame, test_df:pd.DataFrame):
        """
        Initializes the DataPreProcessing class with the training and testing data.
            Impute missing values in the test dataframe (test_df).
            Raname columns for better readability.
            Encode categorical and binary columns using a OneHotEncoder. Ensures proper reference values are dropped

        Parameters
            pd.DataFrame train_df: The synthetic data generated from the original DataFrame
            pd.DataFrame test_df: The original data.
        """
        
        # A copy of train_df and test_df is made to avoid directly altering the data.
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        # Age ('leeft') dropped from datasets.
        self.train_df.drop(columns='leeft', inplace=True)
        self.test_df.drop(columns='leeft', inplace=True)

        # Columns with data type int64 in test_df are converted to float64 to avoid errors during encoding. Original data had a mix of int and float.
        for col in self.test_df.columns:
            if self.test_df[col].dtype == 'int64':
                self.test_df[col] = self.test_df[col].astype('float64')


    def Impute_Missing(self, zero_threshold=None):
        """
        Impute missing values in test_df.
        
        Parameters:
            zero_threshold (float): define the proportion of values that can be missing. Variables that pass the threshold will be imputed with zero
        
        Returns:
            test_df (pd.DataFrame) with missing values imputed.
        """

        # check if an appropriate threshold was given
        if isinstance(zero_threshold, float) is False:
            zero_threshold = 0.66

            print(f'Using {zero_threshold} as threshold to zero-impute missing values.')

            # retrieve columns with more than threshold proportion of missing values
            high_missing_variables = [variable_name for variable_name in self.test_df.columns if
                                    (self.test_df[variable_name].isna().sum() / len(self.test_df[variable_name])) >= zero_threshold]

            # impute a value of 0 for those with more missing considered desirable
            self.test_df[high_missing_variables] = self.test_df[high_missing_variables].fillna(0)

            # initialise the imputation
            print(f'Imputing using the MICE / Iterative Imputation procedure')

            # use most frequent to cater for the plethora of categorical variables
            imputation = IterativeImputer(keep_empty_features=True, initial_strategy='most_frequent')

            # impute data and convert back to pd.DataFrame
            self.test_df = pd.DataFrame(imputation.fit_transform(self.test_df), columns=self.test_df.keys())

            # check if there are any missing values left
            print("Imputation Test")
            self.test_df.info()

            return self.test_df


    def Rename_Cols(self):
        """
        Rename columns of the DataFrames before OHE for readability.

        Returns:
            train_df (pd.DataFrame): The training dataset with renamed columns
            test_df (pd.DataFrame): The testing dataset with renamed columns
        """

        # Dictionary of columns to rename
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

        self.train_df.rename(columns=columns_to_rename, inplace=True)
        self.test_df.rename(columns=columns_to_rename, inplace=True)

        print('\n Columns Renamed')

        return self.train_df, self.test_df


    def Cat_Encoder(self):
        """
        Encode categorical and binary columns using a OneHotEncoder. Ensures proper reference values are dropped. Model refitted with OHE data.
    
        Attributes:
            - refs (dict): A dictionary containing categorical columns and reference values established in original study. Reference values dropped.
                
        Returns:
            - train_df: The one hot encoded training dataset
            - test_df: The one hot encoded testing dataset
        """

        # Reference values (to be dropped) for categorical and binary columns
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

        # Get columns and their reference values to drop
        ref_cols = list(refs.keys()) # Get the column names
        refs_to_drop = [refs[col] for col in ref_cols] # Get the reference values
        
        # Create a OneHotEncoder and drop the reference values
        ohe = OneHotEncoder(drop=refs_to_drop, sparse_output=False, handle_unknown='ignore')

        # Encode the specified columns
        cat_col_transformer = ColumnTransformer(
            transformers=[
                ('ohe', ohe, ref_cols)  
            ], remainder='passthrough'  # Keep other columns unchanged
        )
        
        # Fit and transform the DataFrame
        train_df_transformed = cat_col_transformer.fit_transform(self.train_df)
        ohe_feature_names = cat_col_transformer.named_transformers_['ohe'].get_feature_names_out(ref_cols)
        self.train_df = pd.DataFrame(train_df_transformed, columns=np.concatenate([ohe_feature_names, self.train_df.drop(columns=ref_cols).columns]))

        test_df_transformed = cat_col_transformer.fit_transform(self.test_df)
        ohe_feature_names = cat_col_transformer.named_transformers_['ohe'].get_feature_names_out(ref_cols)
        self.test_df = pd.DataFrame(test_df_transformed, columns=np.concatenate([ohe_feature_names, self.test_df.drop(columns=ref_cols).columns]))
        
        print('\n Data Encoded')

        # Dictionary to rename the encoded columns
        encoded_cols = {
            "Education Level_0.0": "No/Primary Education_1.0",
            "Education Level_1.0": "Secondary Education_2.0",
            "Tumour Type_2.0": "Tumour Type Head/Neck_2.0",
            "Tumour Type_3.0": "Tumour Type Digestive_3.0",
            "Tumour Type_4.0": "Tumour Type Colon and Rectal_4.0",
            "Tumour Type_5.0": "Tumour Type Bone and STS_5.0",
            "Tumour Type_6.0": "Tumour Type Other_6.0",
            "Tumour Type_7.0": "Tumour Type Germ Cell_7.0",
            "Tumour Type_8.0": "Tumour Type Breast_8.0",
            "Tumour Type_9.0": "Tumour Type Female Genitalia_9.0",
            "Tumour Type_10.0": "Tumour Type Lymph_10.0",
            "Tumour Type_11.0": "Tumour Type Myeloid_11.0",
            "Tumour Type_12.0": "Tumour Type Thyroid Gland_12.0",
            "Tumour Type_13.0": "Tumour Type CNS_13.0",
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
        
        # Rename encoded columns for better readability
        self.train_df = self.train_df.rename(columns={col: encoded_cols.get(col, col) for col in self.train_df.columns})
        print('\n Train Columns Renamed')
        self.train_df.info()
        self.test_df = self.test_df.rename(columns={col: encoded_cols.get(col, col) for col in self.test_df.columns})
        print('\n Test Columns Renamed')
        self.test_df.info()

        return self.train_df, self.test_df

    def Train_Test_Split(self):
        """
        Split the data into X and y for training and testing.

        Returns:
            - X_train (pd.DataFrame): The training data
            - y_train (pd.Series): The training labels
            - X_test (pd.DataFrame): The testing data
            - y_test (pd.Series): The testing labels
        """
        # Split the data into X and y
        self.X_train = self.train_df.drop(columns=["Negative Body Image"])
        self.y_train = self.train_df["Negative Body Image"]
        self.X_test = self.test_df.drop(columns=["Negative Body Image"])
        self.y_test = self.test_df["Negative Body Image"]

        print('X_train shape', self.X_train.shape)
        print('y_train shape', self.y_train.shape)
        print('X_test shape', self.X_test.shape)
        print('y_test shape', self.y_test.shape)

        return self.X_train, self.y_train, self.X_test, self.y_test, 


if __name__ == "__main__":
    train_path = input("Input the training data file path: ")
    train_df = pd.read_csv(train_path, delimiter=',', encoding='latin1')
    
    test_path = input("Input the testing data file path: ")
    test_df = pd.read_csv(test_path, delimiter=',', encoding='latin1')
    
    pre_process_data = DataPreProcessing(train_df, test_df)
    pre_process_data.Impute_Missing()
    pre_process_data.Rename_Cols()
    pre_process_data.Cat_Encoder()
    pre_process_data.Train_Test_Split()



    





