import logging
import os 

from data_pre_processing import DataPreProcessing


import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# Create model
# Feature selection
# Parameter tuning
# Feature importance


class DataProcessing:
    def __init__(self, df: pd.DataFrame): #FIXME: Maybe do this things in previous script/class and then return it as a var name.
        self.df = df.copy()
        self.pre_processor = DataPreProcessing(self.df)
        self.pre_processor.Data_Exploration()
        self.pre_processor.Rename_Cols(self.df)
        self.pre_processor.Cat_Encoder(self.df)

        self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val, = self.pre_processor.Ttv_Split()

        self.X_val_ft = None
        self.X_train_ft = None
        self.selected_features = None
        self.log_reg_e = None
        self.log_reg_e_ft = None
        self.y_proba_ft = None
        self.y_pred_ft = None
        self.feature_importance_df = None

    def Create_Model(self, df):

        """

        Function to

            1. Initialize logistic regression model with elastic net regularization: log_reg_e
                The l1_ratio parameter controls the mix of L1 and L2 regularization: 0.5   
                C controls the sparsity: the smaller C the fewer features selected: 0.1
            2. Resample the training data using SMOTE to handle class imbalance.
            3. Fits the model on the training data.
        
        """

        self.log_reg_e = LogisticRegression(penalty='elasticnet', solver ='saga', l1_ratio=0.5, C = 0.1)
        
        sm = SMOTE(random_state=42)
        self.X_train, self.y_train = sm.fit_resample(self.X_train, self.y_train)        

        self.log_reg_e.fit(self.X_train, self.y_train)

        ("X_Train info check: ")
        self.X_train.info()

        return self.log_reg_e, self.X_train, self.y_train


    def Feature_Selection(self, df):

        """

        Function to select features for the model.

        Inputs
            1. SelectFromModel: Selects features based on the fitted model.
                -  Threshold parameter: If None and if the estimator has a parameter penalty set to l1, 
                        either explicitly or implicitly (e.g, Lasso), the threshold used is 1e-5.
                #TODO: Implement?: max_features int, callable, default=None.
                    If an integer, then it specifies the maximum number of features to allow. 
                    If a calable, then it specifies how to calculate the maximum number of features allowed by using the output of max_features(X).
                    To only select based on max_features, set threshold=-np.inf.
            2. selected_features_mask: A mask indicating which features were selected.
            3. selected_features: The names of the selected features from mask
            4. Transforms the training and validation datasets to `X_train_ft` and `X_val_ft`) to include only the selected features.
        
        """
        
        model_log_features = SelectFromModel(self.log_reg_e, prefit=True)
        selected_features_mask = model_log_features.get_support()
        self.selected_features = self.X_train.columns[selected_features_mask]

        self.X_train_ft = pd.DataFrame(model_log_features.transform(self.X_train), columns=self.selected_features)
        self.X_val_ft = pd.DataFrame(model_log_features.transform(self.X_val), columns=self.selected_features)

        print("\n See selected features in X_train_ft: \n ")
        self.X_train_ft.info()

        return self.X_train, self.X_val, self.X_val_ft, self.log_reg_e, self.X_train_ft, self.selected_features

    def Parameter_Tuning(self, df):

        """

        Function to tune hyperparameters and refit model on selected features.
    
        1. Computes class weights to handle class imbalance.
        2. Performs a grid search to find the best hyperparameters for the logistic regression model.
        3. Refits the logistic regression model with the best hyperparameters on the selected features.
        4. Makes predictions on the validation data.

        """

        #compute weights for class imbalance based on frequency in training data
        class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

        param_grid_elas = {
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],  # Range of l1_ratio values to search
            'C': [0.01, 0.1, 1, 10, 100],  # Range of regularization strengths. Smaller values, stronger regularization. 0.01 resulted in 'no predicted samples' for precision and would have to initiate the zero_division parameter.
                }
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        grid_search = GridSearchCV(self.log_reg_e, param_grid=param_grid_elas, scoring=scoring_metrics, refit='accuracy', cv=5) 
        grid_search.fit(self.X_train_ft, self.y_train) 
        best_params = grid_search.best_params_
        print("\n Best parameters found: ", best_params)
        print("\n Best score found: ", grid_search.best_score_)

        # Refit the logistic regression model on X_train_ft using balanced classes and best params.
        self.log_reg_e_ft = LogisticRegression(class_weight=class_weight_dict, **best_params).fit(self.X_train_ft, self.y_train)
        print('\n Class weight', class_weight_dict)

        # Predictions
        self.y_pred_ft = self.log_reg_e_ft.predict(self.X_val_ft)
        self.y_proba_ft = self.log_reg_e_ft.predict_proba(self.X_val_ft)

        return self.y_train, self.X_train_ft, self.X_val_ft, self.log_reg_e, self.log_reg_e_ft, self.y_proba_ft, self.y_pred_ft

    def Feature_Importance(self, df):

        """ 
        Fucntion to:

        1. Compute feature importance using coefficients, and odds ratios (of selected features).
        2. Calculate standard errors and confidence intervals for coefficients.
        3. Create a DataFrame of features, coefficients, and odds ratios.
        4. Plot the odds ratios with 95% confidence intervals.

        """

        feature_importance = pd.Series(self.log_reg_e_ft.coef_.flatten(), index=self.selected_features)
        feature_importance = feature_importance.round(3)

        coef = self.log_reg_e_ft.coef_.flatten()
        coef = np.sort(coef, axis=None)[::-1] # to sort df by coef
        coef = np.round(coef, 3)

        # Calculate standard errors
        se = np.sqrt(np.diag(np.linalg.inv(np.dot(self.X_train_ft.T, self.X_train_ft))))

        # Calculate CIs for coefficients
        z = 1.96  # for 95% confidence interval
        ci_lower = coef - z * se
        ci_upper = coef + z * se

        # Convert coefficients to CIs
        ci_lower_or = np.exp(ci_lower)
        ci_upper_or = np.exp(ci_upper)
        odds_ratios = np.exp(coef)
        odds_ratios = np.round(odds_ratios, 3)

        # Create a DataFrame of features, coefficients, and odds ratios
        self.feature_importance_df = pd.DataFrame({
            'Feature': feature_importance.index,
            'Importance': feature_importance.values,
            'Coefficient': coef.flatten(),
            'Odds Ratio': odds_ratios.flatten(),
            'CI Lower': ci_lower_or,
            'CI Upper': ci_upper_or
        })

        # ft_impt_file_path = os.path.join(target_folder, f'{model_name}_Feature_Importance_Table_{date_time}.csv')
        # feature_importance_df.to_csv(ft_impt_file_path, index=False)
        
        print('\n', self.feature_importance_df)

        return coef, ci_lower_or, ci_upper_or, odds_ratios, feature_importance, self.feature_importance_df

if __name__ == "__main__":
    df_file_path = input("Provide df file path: ")
    df = pd.read_csv(df_file_path, delimiter=',', encoding='latin1')

    post_process_data = DataProcessing(df)
    post_process_data.Create_Model(df)
    post_process_data.Feature_Selection(df)
    post_process_data.Parameter_Tuning(df)
    post_process_data.Feature_Importance(df)