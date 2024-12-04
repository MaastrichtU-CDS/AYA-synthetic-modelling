import os 

from data_pre_processing import DataPreProcessing

import pandas as pd
import numpy as np
import sklearn
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import  GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance

# SMOTE & class (im)balance
# Create model
# Feature selection
# Parameter tuning
# Feature importance


class DataProcessing:
    def __init__(self, df: pd.DataFrame, algorithm: str): 
        self.df = df.copy()
        self.algorithm = algorithm

        self.pre_processor = DataPreProcessing(self.df)
        self.pre_processor.Rename_Cols(self.df)
        self.pre_processor.Cat_Encoder(self.df)
        
        self.X = self.pre_processor.X
        self.y = self.pre_processor.y
        self.X_ft = None
        self.y_proba_ft = None
        self.y_pred_ft = None
   
        self.selected_features = None

        self.log_reg_e = None
        self.log_reg_e_ft = None
        self.best_estimator_log_reg = None
        self.model_log_features = None 
        self.best_params_log = None
        self.selected_features_log = None
        self.selected_features_mask_log = None
        self.feature_importance_log_reg = None
        self.feature_importance_log_reg_df = None


        self.random_forest = None
        self.random_forest_ft = None
        self.model_rf_features = None
        self.best_estimator_rf = None
        self.best_params_rf = None
        self.selected_features_rf = None
        self.selected_features_mask_rf = None
        self.importance_scores_rf = None
        self.feature_importance_rf_df = None


        self.mlp = None
        self.mlp_ft = None
        self.model_mlp_features = None
        self.best_estimator_mlp = None
        self.best_params_mlp = None
        self.selected_features_mlp = None
        self.selected_features_mask_mlp = None 
        self.importance_scores_mlp = None
        self.feature_importance_mlp_df = None


        self.orig_train_size = None
        self.post_smote_train_size = None
        self.class_weight_before_smote = None
        self.class_weight_post_smote = None

    def Smote_Class_Balance(self) -> pd.DataFrame:

        '''

        Function to

            1. Compute class weights and train size before and after SMOTE.
            2. Balance the dataset using SMOTE.
        
        '''

        self.orig_train_size = self.X.shape
        self.class_weight_before_smote = compute_class_weight(class_weight='balanced', classes=np.unique(self.y), y=self.y)

        self.sm = SMOTE(random_state=42) # Apply SMOTE to balance DS
        self.X, self.y = self.sm.fit_resample(self.X, self.y)    
        
        self.post_smote_train_size = self.X.shape
        self.class_weight_post_smote = compute_class_weight(class_weight='balanced', classes=np.unique(self.y), y=self.y)
        
        print (f'\n Original training data size: {self.orig_train_size}, \n Post-SMOTE training data size: {self.post_smote_train_size}')
        print (f'\n Class weights before SMOTE: {self.class_weight_before_smote}, \n Class weights after SMOTE: {self.class_weight_post_smote}')
    
        return self.X, self.y, self.class_weight_before_smote, self.class_weight_post_smote, self.orig_train_size, self.post_smote_train_size

    def Create_Model(self):

        """

        Function to initialize and fit the indicated model
        
        Inputs
            log_reg_e: logistic regression model with elastic net regularization
                - The l1_ratio parameter controls the mix of L1 and L2 regularization: 0.5 for Lasso   
                - C controls the sparsity: the smaller C the fewer features selected: 0.1
            
            rf: Random Forest Classifier 
                - n_estimators = trees. The more trees, the less likely the model is to overfit. General range: 50-200
            
            mlp: Multi-layer Perceptron Classifier 
                - Activation function: logistic
                - alpha: Alpha in a MLP typically represents L2 regularization strength only. 
                - hidden_layer_sizes: 100 is defult. 50, 100, 200 are common choices.
        
        """

        if self.algorithm == 'log_reg':
            self.log_reg_e = LogisticRegression(penalty='elasticnet', solver ='saga', l1_ratio=0.5, C = 0.1, max_iter=10000, random_state=42)    
            self.log_reg_e.fit(self.X, self.y)
            return self.log_reg_e

        if self.algorithm == 'rf':
            self.random_forest = RandomForestClassifier(random_state=42) 
            self.random_forest.fit(self.X, self.y)
            return self.random_forest

        if self.algorithm == 'mlp':
            self.mlp = MLPClassifier(shuffle=True, activation='logistic', random_state=42)
            self.mlp.fit(self.X, self.y)
            return self.mlp
    

    def Feature_Selection(self):

        """

        Function to select features based on importance for the model.

        Inputs
            SelectFromModel applied to log_reg_e
                - Selects features based on the fitted model.
                - Threshold parameter: If None and if the estimator has a parameter penalty set to l1, 
                        either explicitly or implicitly (e.g, Lasso), the threshold used is 1e-5.

            Permutation Importance only applied to mlp and rf. No coef_ attribute.
                - permutation_importance: Computed feature importances using permutation importance.
                    - n_repeats: Number of times to permute the feature. Default is 5.
                    - random_state: Random seed for reproducibility.
                - importance_scores_mlp: The feature importance scores.
                - threshold: The threshold for selecting the top 20% of features.

            Across all models:
                - selected_features_mask: A mask indicating which features were selected.
                - Transforms the training dataset (X_ft) to include only the selected features.
       
        Outputs
            X_ft: The training dataset with only the selected features.
            selected_features_log: The selected features for the logistic regression model.
            selected_features_rf: The selected features for the random forest model.
            selected_features_mlp: The selected features for the MLP model.        
        """ 

        if self.algorithm=="log_reg":
            self.model_log_features = SelectFromModel(self.log_reg_e, prefit=True)
            self.selected_features_mask_log = self.model_log_features.get_support()
            self.selected_features_log = self.X.columns[self.selected_features_mask_log]
            self.X_ft = pd.DataFrame(self.model_log_features.transform(self.X), columns=self.selected_features_log)

        if self.algorithm=="rf":
            self.importance_scores_rf = self.random_forest.feature_importances_
            self.threshold_rf = sorted(self.importance_scores_rf, reverse=True)[int(len(self.importance_scores_rf) * 0.2)]  # Select top 20% features
            self.selected_features_mask_rf = self.importance_scores_rf >= self.threshold_rf
            self.selected_features_rf = self.X.columns[self.selected_features_mask_rf]
            self.X_ft = pd.DataFrame(self.X.loc[:, self.selected_features_rf])
    
        if self.algorithm=="mlp":
            self.importance_scores_mlp = permutation_importance(self.mlp, self.X, self.y, random_state=42, scoring='accuracy') 
            self.importance_scores_mean_mlp = np.array(self.importance_scores_mlp.importances_mean)
            self.threshold_mlp = sorted(self.importance_scores_mean_mlp, reverse=True)[int(len(self.importance_scores_mean_mlp) * 0.2)]  # Select top 20% features
            self.selected_features_mask_mlp = self.importance_scores_mean_mlp >= self.threshold_mlp
            self.selected_features_mlp = self.X.columns[self.selected_features_mask_mlp]
            self.X_ft = pd.DataFrame(self.X.loc[:, self.selected_features_mlp])



        print("\n See selected features in X_ft: \n ")
        self.X_ft.info()

        return self.X_ft, self.selected_features_log, self.selected_features_rf, self.selected_features_mlp, self.importance_scores_rf, self.importance_scores_mlp, self.model_log_features, self.selected_features_mask_log, self.selected_features_mask_rf, self.selected_features_mask_mlp

      

    def Parameter_Tuning(self):

        """

        Function to tune hyperparameters and refit model on selected features.
    
            1. Computes class weights to handle class imbalance
            2. Performs a grid search to find the best hyperparameters for the chosen model: grid_search_log, grid_search_rf, grid_search_mlp
            3. Refits the model with the best hyperparameters on the selected features: Param_grid_elas, Param_grid_rf, Param_grid_mlp
            4. Makes predictions on the validation data on the selected features (ft).

        Inputs
            - param_grid: The hyperparameters to search for in each model.
            - scoring_metrics_log: The scoring metrics to choose for each model.
            - grid_search: The grid search object for each model.
 
        Outputs
            - log_reg_e_ft: The refitted logistic regression model on the selected features.
            - random_forest_ft: The refitted random forest model on the selected features.
            - mlp_ft: The refitted MLP model on the selected features.

            - best_params: The best hyperparameters found for each model.
            - best_estimator: The best estimator found for each model.
            - y_proba_ft: The predicted probabilities on the validation data.
            - y_pred_ft: The predicted labels on the validation data.

        """

        if self.algorithm == 'log_reg':
            self.param_grid_elas = {
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],  # Range of l1_ratio values to searchjobs
                'C': [0.01, 0.1, 1, 10, 100],  # Range of regularization strengths. Smaller values, stronger regularization. 0.01 resulted in 'no predicted samples' for precision and would have to initiate the zero_division parameter.
                    }
            self.scoring_metrics_log = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            self.scoring_metrics_log_dict = {metric: metric for metric in self.scoring_metrics_log}
            self.grid_search_log = GridSearchCV(
                estimator=self.log_reg_e, 
                param_grid=self.param_grid_elas, 
                scoring=self.scoring_metrics_log_dict,  
                refit='accuracy', 
                cv=5, # cv = cross-validation = n folds
                ) 
            self.grid_search_log.fit(self.X_ft, self.y) 
            self.best_params_log = self.grid_search_log.best_params_ 
            self.best_estimator_log_reg = self.grid_search_log.best_estimator_ 
            print("\n Best estimator found [log_reg_e]: ", self.best_estimator_log_reg)
            print("\n Best score found based on refit method [log_reg_e]: ", self.grid_search_log.best_score_)
        
            # Refit on X_ft using best params
            self.log_reg_e_ft = LogisticRegression(penalty='elasticnet', solver='saga', random_state=42, **self.best_params_log).fit(self.X_ft, self.y)
            print("\n Done fitting log_reg with best params")

            self.y_pred_ft = self.log_reg_e_ft.predict(self.X_ft) 
            self.y_proba_ft = self.log_reg_e_ft.predict_proba(self.X_ft) 

            return self.log_reg_e, self.log_reg_e_ft, self.best_estimator_log_reg, self.X_ft, self.y_proba_ft, self.y_pred_ft

        if self.algorithm == 'rf':

            self.param_grid_rf = {
                'n_estimators': [10, 50, 100, 150, 200],
                }
            self.scoring_metrics_rf = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            self.scoring_metrics_rf_dict = {metric: metric for metric in self.scoring_metrics_rf}
            self.grid_search_rf = GridSearchCV(
                estimator=self.random_forest, 
                param_grid=self.param_grid_rf, 
                scoring=self.scoring_metrics_rf_dict, 
                refit='accuracy', 
                cv=5,
                ) 
            self.grid_search_rf.fit(self.X_ft, self.y)
            self.best_estimator_rf = self.grid_search_rf.best_estimator_
            self.best_params_rf = self.grid_search_rf.best_params_

            print("\n Best estimator found [rf]: ", self.best_estimator_rf)
            print("\n Best score found based on refit method [rf]: ", self.grid_search_rf.best_score_)

            self.random_forest_ft = RandomForestClassifier(**self.best_params_rf, random_state=42).fit(self.X_ft, self.y)

            print("\n Done fitting rf with best params")
            
            self.y_pred_ft = self.random_forest_ft.predict(self.X_ft) 
            self.y_proba_ft = self.random_forest_ft.predict_proba(self.X_ft) 

            return self.random_forest, self.random_forest_ft, self.best_estimator_rf, self.X_ft, self.y_proba_ft, self.y_pred_ft

        if self.algorithm == 'mlp': #FIXME: Cross validation needed for MLP; Hidden layer sizes default: 100
            
            self.mlp_ft = MLPClassifier(shuffle = True, activation='logistic', random_state=42).fit(self.X_ft, self.y)
            print("\n Done fitting mlp with X_fr; activation: logistic")

            self.y_pred_ft = self.mlp_ft.predict(self.X_ft)
            self.y_proba_ft = self.mlp_ft.predict_proba(self.X_ft)

            return self.mlp, self.mlp_ft, self.X_ft, self.y_pred_ft, self.y_proba_ft


    def Feature_Importance(self)-> pd.DataFrame:

        """ 

        Function to:

            1. Compute feature importances for the selected model using the selected features.
                - For log_reg: using coefficients, and odds ratios (of selected features). 
                        Then calculate standard errors and confidence intervals for coefficients.

                - For rf: using feature importances. Random Forests inherently provide feature importance scores 
                        based on how useful each feature is in reducing the impurity (e.g., Gini impurity or entropy) 
                        across the trees in the forest. This is a built-in mechanism that leverages the structure of decision trees

                - For mlp: using permutation importance. Permutation importance, on the other hand, is a model-agnostic method 
                        that can be applied to any predictive model, including MLPs. The weights in an MLP are distributed across 
                        many neurons and layers, making it difficult to directly interpret the importance of individual features
                
            2. Create & return a DataFrame (df) of importances
                - feature_importance_log_df: features, coefficients, odds ratios, p-values, CIs
                - feature_importance_rf_df and feature_importance_mlp_df: features, importances, normalized importances

        """

        if self.algorithm == 'log_reg':

            print("\n Feature importance for log_reg: ")

            self.feature_importance_log_reg = pd.Series(self.log_reg_e_ft.coef_.flatten(), index=self.selected_features_log)

            self.coef = self.log_reg_e_ft.coef_.flatten()

            self.normalized_importance = np.abs(self.coef) / np.sum(np.abs(self.coef))

            # Calculate standard errors
            self.se = np.sqrt(np.diag(np.linalg.inv(np.dot(self.X_ft.T, self.X_ft))))

            # Calculate CIs for coefficients
            self.z = 1.96  # for 95% confidence interval
            self.ci_lower = self.coef - self.z * self.se
            self.ci_upper = self.coef + self.z * self.se

            # Convert coefficients to CIs
            self.ci_lower_or = np.exp(self.ci_lower)
            self.ci_upper_or = np.exp(self.ci_upper)
            self.odds_ratios = np.exp(self.coef)

            self.p_values = 2 * (1 - stats.norm.cdf(np.abs(self.coef / self.se)))  # two-tailed test
            self.p_values = np.round(self.p_values, 3)
            
            # Create a DataFrame of features, coefficients, odds ratios, p-values, and CIs
            self.feature_importance_log_reg_df = pd.DataFrame({
                'Feature': self.feature_importance_log_reg.index,
                'Importance': self.feature_importance_log_reg.values.round(3), 
                'Normalized Importance': self.normalized_importance.round(3),
                'Coefficient': self.coef.flatten().round(3),
                'Odds Ratio': self.odds_ratios.flatten().round(3),
                'CI Lower': self.ci_lower_or.round(3),
                'CI Upper': self.ci_upper_or.round(3),
                'Standard Error': self.se.flatten().round(3),
                'P Value': self.p_values.flatten()
            }).sort_values(by='Importance', ascending=False)

            print('\n', self.feature_importance_log_reg_df)

            return self.feature_importance_log_reg_df


        if self.algorithm == 'rf':
            
            print("\n Feature importance for rf: ")

            self.importance_scores_rf = self.random_forest_ft.feature_importances_
            self.normalized_importance_rf = self.importance_scores_rf / np.sum(self.importance_scores_rf)

            self.feature_importance_rf_df = pd.DataFrame({
                'Feature': self.selected_features_rf,
                'Importance': self.importance_scores_rf.round(3),
                'Normalized Importance': self.normalized_importance_rf.round(3)
            }).sort_values(by='Importance', ascending=False)
            
            print('\n', self.feature_importance_rf_df)

            return self.feature_importance_rf_df


        if self.algorithm == 'mlp':

            print("\n Feature importance for mlp: ")

            self.importance_scores_mlp = permutation_importance(self.mlp_ft, self.X_ft, self.y, n_repeats=5, random_state=42)
            self.importance_scores_mean_mlp = np.array(self.importance_scores_mlp.importances_mean)
            self.selected_features_mlp = self.X_ft.columns
            self.normalized_importance_mlp = self.importance_scores_mean_mlp / np.sum(self.importance_scores_mean_mlp)

            self.feature_importance_mlp_df = pd.DataFrame({
                'Feature': self.selected_features_mlp,
                'Importance': self.importance_scores_mean_mlp.round(3), 
                'Normalized Importance': self.normalized_importance_mlp.round(3)
            }).sort_values(by='Importance', ascending=False)

            print('\n', self.feature_importance_mlp_df)

            return self.feature_importance_mlp_df
    

if __name__ == "__main__":
    df_file_path = input("Provide df file path: ")
    df = pd.read_csv(df_file_path, delimiter=',', encoding='latin1')

    post_process_data = DataProcessing(df, algorithm=input("Provide algorithm (log_reg, rf, mlp): "))
    post_process_data.Smote_Class_Balance()
    post_process_data.Create_Model()
    post_process_data.Feature_Selection()
    post_process_data.Parameter_Tuning()
    post_process_data.Feature_Importance()
