import logging
import os 

from data_pre_processing import DataPreProcessing

import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
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
        self.random_forest = None
        self.random_forest_ft = None
        self.model_log_features = None
        self.selected_features_mask = None

        self.sm = SMOTE(random_state=42)
        self.X_train, self.y_train = self.sm.fit_resample(self.X_train, self.y_train)    

    def Create_Model(self, df, model):

        """

        Function to

            1. Initialize logistic regression model with elastic net regularization: log_reg_e
                The l1_ratio parameter controls the mix of L1 and L2 regularization: 0.5   
                C controls the sparsity: the smaller C the fewer features selected: 0.1
            2. Resample the training data using SMOTE to handle class imbalance.
            3. Fits the model on the training data.
        
        """

        if model == 'log_reg':
            self.log_reg_e = LogisticRegression(penalty='elasticnet', solver ='saga', l1_ratio=0.5, C = 0.1, max_iter=10000, random_state=42)    
            self.log_reg_e.fit(self.X_train, self.y_train)
            # return self.log_reg_e


        if model == 'rf':
            self.random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
            self.random_forest.fit(self.X_train, self.y_train)
            # return self.random_forest

        return self.X_train, self.y_train, self.log_reg_e, self.random_forest
    

    def Feature_Selection(self, df, model):

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

        if model=="log_reg":
            self.model_log_features = SelectFromModel(self.log_reg_e, prefit=True)
            self.selected_features_mask = self.model_log_features.get_support()
            self.selected_features = self.X_train.columns[self.selected_features_mask]
            self.X_train_ft = pd.DataFrame(self.model_log_features.transform(self.X_train), columns=self.selected_features)
            self.X_val_ft = pd.DataFrame(self.model_log_features.transform(self.X_val), columns=self.selected_features)


        if model=="rf":
            self.model_log_features = SelectFromModel(self.random_forest, prefit=True)
            self.selected_features_mask = self.model_log_features.get_support()
            self.selected_features = self.X_train.columns[self.selected_features_mask]
            self.X_train_ft = pd.DataFrame(self.model_log_features.transform(self.X_train), columns=self.selected_features)
            self.X_val_ft = pd.DataFrame(self.model_log_features.transform(self.X_val), columns=self.selected_features)



        print("\n See selected features in X_train_ft: \n ")
        self.X_train_ft.info()

        return self.X_train, self.X_val, self.X_val_ft, self.log_reg_e, self.X_train_ft, self.selected_features

    def Parameter_Tuning(self, df, model):

        """

        Function to tune hyperparameters and refit model on selected features.
    
        1. Computes class weights to handle class imbalance.
        2. Performs a grid search to find the best hyperparameters for the logistic regression model.
        3. Refits the logistic regression model with the best hyperparameters on the selected features.
        4. Makes predictions on the validation data.

        """

        #compute weights for class imbalance based on frequency in training data
        self.class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        self.class_weight_dict = {i: self.class_weights[i] for i in range(len(self.class_weights))}

        if model == 'log_reg':
            self.param_grid_elas = {
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],  # Range of l1_ratio values to search
                'C': [0.01, 0.1, 1, 10, 100],  # Range of regularization strengths. Smaller values, stronger regularization. 0.01 resulted in 'no predicted samples' for precision and would have to initiate the zero_division parameter.
                    }
            self.scoring_metrics_log = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            self.grid_search_log = GridSearchCV(self.log_reg_e, param_grid=self.param_grid_elas, scoring=self.scoring_metrics_log, refit='accuracy', cv=5) 
            self.grid_search_log.fit(self.X_train_ft, self.y_train) 
            self.best_params_log = self.grid_search_log.best_params_
            print("\n Best parameters found [log_reg_e]: ", self.best_params_log)
            print("\n Best score found [log_reg_e]: ", self.grid_search_log.best_score_) #FIXME: What is the name of this score?

            # Refit the logistic regression model on X_train_ft using balanced classes and best params.
            self.log_reg_e_ft = LogisticRegression(class_weight=self.class_weight_dict, **self.best_params_log).fit(self.X_train_ft, self.y_train)
            
            print('\n Class weight', self.class_weight_dict)

            self.y_pred_ft = self.log_reg_e_ft.predict(self.X_val_ft)
            self.y_proba_ft = self.log_reg_e_ft.predict_proba(self.X_val_ft)

            print("\n model = log_reg")

        if model == 'rf':

            self.param_grid_rf = {
                'n_estimators': [50, 100, 150, 200],
            }
            self.scoring_metrics_rf = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            self.grid_search_rf = GridSearchCV(self.random_forest, param_grid=self.param_grid_rf, scoring=self.scoring_metrics_rf, refit='accuracy', cv=5) 
            self.grid_search_rf.fit(self.X_train_ft, self.y_train)
            self.best_params_rf = self.grid_search_rf.best_params_
            print("\n Best parameters found [rf]: ", self.best_params_rf)
            print("\n Best score found [rf]: ", self.grid_search_rf.best_score_) #FIXME: What is the name of this score?

            self.random_forest_ft = RandomForestClassifier(**self.best_params_rf).fit(self.X_train_ft, self.y_train)

            self.y_pred_ft = self.random_forest_ft.predict(self.X_val_ft)
            self.y_proba_ft = self.random_forest_ft.predict_proba(self.X_val_ft)

            print("\n model = rf")

        return self.y_train, self.X_train_ft, self.X_val_ft, self.log_reg_e, self.log_reg_e_ft, self.y_proba_ft, self.y_pred_ft, self.random_forest, self.random_forest_ft

    def Feature_Importance(self, df, model):

        """ 
        Fucntion to:

        1. Compute feature importance using coefficients, and odds ratios (of selected features).
        2. Calculate standard errors and confidence intervals for coefficients.
        3. Create a DataFrame of features, coefficients, and odds ratios.
        4. Plot the odds ratios with 95% confidence intervals.

        """

        #FIXME: /root/.venv/lib/python3.10/site-packages/sklearn/linear_model/_sag.py:349: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge warnings.warn. 

        if model == 'log_reg':
            self.feature_importance_log = pd.Series(self.log_reg_e_ft.coef_.flatten(), index=self.selected_features)
            self.feature_importance_log = self.feature_importance_log.round(3)

            self.coef = self.log_reg_e_ft.coef_.flatten()
            self.coef = np.sort(self.coef, axis=None)[::-1] # to sort df by coef
            self.coef = np.round(self.coef, 3)

            # Calculate standard errors
            self.se = np.sqrt(np.diag(np.linalg.inv(np.dot(self.X_train_ft.T, self.X_train_ft))))

            # Calculate CIs for coefficients
            self.z = 1.96  # for 95% confidence interval
            self.ci_lower = self.coef - self.z * self.se
            self.ci_upper = self.coef + self.z * self.se

            # Convert coefficients to CIs
            self.ci_lower_or = np.exp(self.ci_lower)
            self.ci_upper_or = np.exp(self.ci_upper)
            self.odds_ratios = np.exp(self.coef)
            self.odds_ratios = np.round(self.odds_ratios, 3)

            # Create a DataFrame of features, coefficients, and odds ratios
            self.feature_importance_log_df = pd.DataFrame({
                'Feature': self.feature_importance_log.index,
                'Importance': self.feature_importance_log.values,
                'Coefficient': self.coef.flatten(),
                'Odds Ratio': self.odds_ratios.flatten(),
                'CI Lower': self.ci_lower_or,
                'CI Upper': self.ci_upper_or
            })

            print('\n', self.feature_importance_log_df)
            return self.coef, self.ci_lower_or, self.ci_upper_or, self.odds_ratios,  self.feature_importance_log_df, self.feature_importance_log


        if model == 'rf':
            self.feature_importance_rf = self.random_forest_ft.feature_importances_
            self.feature_importance_rf = self.feature_importance_rf.round(3)

            #RF doesn't return coef_ like linear models do. instead they provide feature importances
            self.feature_importance_rf_df = pd.DataFrame({
                'Feature': self.selected_features,
                'Importance': self.feature_importance_rf
            }) 
            
            print('\n', self.feature_importance_rf_df)

            # ft_impt_file_path = os.path.join(target_folder, f'{model_name}_Feature_Importance_Table_{date_time}.csv')
            # feature_importance_df.to_csv(ft_impt_file_path, index=False)
            
            return self.feature_importance_rf_df, self.feature_importance_rf

if __name__ == "__main__":
    # df_file_path = input("Provide df file path: ")
    df_file_path = '/root/ctgan/synthetic_data/survaya_03062022_clean_n_output_model_CTGAN_3735_samples.csv'
    df = pd.read_csv(df_file_path, delimiter=',', encoding='latin1')

    model = "log_reg" #model options: log_reg, rf

    post_process_data = DataProcessing(df)
    post_process_data.Create_Model(df, model=model)
    post_process_data.Feature_Selection(df, model=model)
    post_process_data.Parameter_Tuning(df, model=model)
    post_process_data.Feature_Importance(df, model=model)