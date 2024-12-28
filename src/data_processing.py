from data_pre_processing import DataPreProcessing

import logging
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import  GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance

# Configure logging
logging.basicConfig(filename='/root/log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessing:
    """
    Class to processes data for modeling by:
        - Balancing the dataset using SMOTE
        - Initializing the model
        - Selecting features within initialized model
        - Tuning hyperparameters to optimize the model
        - Computing feature importance of optimized model
    """

    def __init__(self, algorithm:str, train_df:pd.DataFrame, test_df:pd.DataFrame): 
        """
        Initialize the class with the algorithm, training and testing dataframes.

        Parameters
            algorithm (str): The algorithm to use for modeling.
            train_df (pd.DataFrame): The training dataset.
            test_df (pd.DataFrame): The testing dataset.
        """

        logging.info('\n Initializing DataProcessing class.')

        self.algorithm = algorithm
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        
        self.pre_processor = DataPreProcessing(self.train_df, self.test_df)
        self.pre_processor.Impute_Missing()
        self.pre_processor.Rename_Cols()
        self.pre_processor.Cat_Encoder()
        self.pre_processor.Train_Test_Split()
        
        self.X_train= self.pre_processor.X_train
        self.y_train = self.pre_processor.y_train
        self.X_test = self.pre_processor.X_test
        self.y_test = self.pre_processor.y_test

        self.orig_train_size = None
        self.post_smote_train_size = None
        self.class_weight_before_smote = None
        self.class_weight_post_smote = None

        # Initialize model for hyperparameter tuning
        self.train_model = None

        # Initialize optimized model after hyperparameter tuning
        self.opt_model = None

        # Initialize model parameters for hyperparameter tuning
        self.param_grid = None
        self.grid_search = None
        self.best_estimator = None
        self.best_params = None
        self.selected_features = None
        self.selected_features_mask = None
        self.feature_importance = None
        self.importance_scores = None
        self.normalized_importance = None
        self.feature_importance_df = None


    def Balance_Smote(self):
        '''
        Implement SMOTE to balance the dataset & resample.

        Returns:
            Values of class weights and train sizes before and after SMOTE.
        '''

        # Get train size before smote
        self.orig_train_size = self.X_train.shape

        # Compute class weights before SMOTE
        self.class_weight_before_smote = np.round(compute_class_weight(class_weight='balanced', classes=np.unique(self.y_train), y=self.y_train), 2)

        logging.info('Applying SMOTE to balance the dataset.')
        
        # Apply SMOTE to balance the dataset
        self.X_train, self.y_train = SMOTE(random_state=42).fit_resample(self.X_train, self.y_train)

        # Get train size after smote
        self.post_smote_train_size = self.X_train.shape
       
        # Compute class weights after SMOTE
        self.class_weight_post_smote = np.round(compute_class_weight(class_weight='balanced', classes=np.unique(self.y_train), y=self.y_train), 2)
        
        logging.info(f'Dataset size before {self.orig_train_size}, Dataset size after {self.post_smote_train_size}. '
                     f'Class weights before {self.class_weight_before_smote}, Class weights after {self.class_weight_post_smote}')
        
        return self.X_train, self.y_train, self.class_weight_before_smote, self.class_weight_post_smote, self.orig_train_size, self.post_smote_train_size
    

    def Initialize_Model(self) -> pd.DataFrame:
        """
        Initialize model based on the chosen algorithm.
        All models are initialized with random_state=42.

        Returns:
            train_model: The initialized model for the chosen algorithm.
        """

        # Elastic Net Logistic Regression (log_reg)
        if self.algorithm == 'log_reg':
            self.train_model = LogisticRegression(penalty='elasticnet', solver ='saga', l1_ratio=0.5, C = 0.1, max_iter=2000, random_state=42)    
            logging.info('Initialized logistic regression (log_reg) with standard elastic net parameters (l1_ratio=0.5, C=0.1). Max iterations set to 2000.')
            logging.captureWarnings(True)

        # Random Forest Classifier (rf)
        if self.algorithm == 'rf':
            self.train_model = RandomForestClassifier(random_state=42) 
            logging.info('Initialized random forest classifier (rf) with standard parameters and random state=42')

        # Multi-layer Perceptron Classifier (mlp)
        if self.algorithm == 'mlp':
            self.train_model = MLPClassifier(shuffle=True, activation='logistic', random_state=42, max_iter=1000)
            logging.info('Initialized multi-layer perceptron (mlp) classifier with the logistic activation function for binary classification')

        # Fit the model on the training data
        self.train_model.fit(self.X_train, self.y_train)
        logging.info(f'Model fitted on training data.')

        return self.train_model
    

    def Feature_Selection_Train(self) -> pd.DataFrame:
        """
        Select features based internal feature importance methods for each fitted model.
                
        Attributes across all models:
            feature_selector: The feature selection method used to select features.
            selected_features_mask: A mask indicating which features were selected.
            selected_features: The names of the selected features.
            Transforms the X_train and X_test datasets to include only the selected features.

        Returns: 
            X_train (pd.DataFrame): Training dataset with only the selected features
            X_test (pd.DataFrame): Testing dataset with only the selected features
            selected_features (pd.Index): The names of the selected features
        """ 

        logging.info('Selecting features based on internal feature importance methods for each fitted model.')

        if self.algorithm=="log_reg": 
            self.feature_selector = SelectFromModel(estimator = self.train_model).fit(self.X_train, self.y_train)
            self.selected_features_mask = self.feature_selector.get_support()
            self.selected_features = self.X_train.columns[self.selected_features_mask]
            self.X_train = pd.DataFrame(self.feature_selector.transform(self.X_train), columns=self.selected_features)
            logging.info(f"See selected features in X_train [logistic regression]: {self.X_train.columns} ")

        if self.algorithm=="rf":
            self.feature_selector = self.train_model.feature_importances_
            self.threshold = sorted(self.feature_selector, reverse=True)[int(len(self.feature_selector) * 0.2)]  # Select top 20% features
            self.selected_features_mask = self.feature_selector >= self.threshold
            self.selected_features = self.X_train.columns[self.selected_features_mask]
            self.X_train = pd.DataFrame(self.X_train.loc[:, self.selected_features])
            logging.info(f"See selected features in X_train [random forest]: {self.X_train.columns} ")
        
        if self.algorithm=="mlp":
            self.feature_selector = permutation_importance(self.train_model, self.X_train, self.y_train, random_state=42, scoring='accuracy') 
            self.feature_selector_mean = np.array(self.feature_selector.importances_mean)
            self.threshold = sorted(self.feature_selector_mean, reverse=True)[int(len(self.feature_selector_mean) * 0.2)]  # Select top 20% features
            self.selected_features_mask = self.feature_selector_mean >= self.threshold
            self.selected_features = self.X_train.columns[self.selected_features_mask]
            self.X_train = pd.DataFrame(self.X_train.loc[:, self.selected_features])
            logging.info(f"See selected features in X_train [multi-layer perceptron classifier]: {self.X_train.columns} ")

        # Ensure test data has same features as training data
        self.X_test = self.X_test[self.X_train.columns]
        
        return self.X_train, self.X_test, self.selected_features
      

    def Hyperparameter_Tuning(self) -> pd.DataFrame:
        """
        Tune hyperparameters and refit model based on training data to create an optimized model (opt_model).

        Attributes:
            grid_search: Performs a search on param_grid to find the best hyperparameter values for the chosen model.
                Across models, grid search is refit on accuracy, has a 5-fold cross-validation, and uses static scoring metrics (scoring_metrics_dict).

        Returns:
            param_grid: The hyperparameters and value ranges to search through to find best hyperparameter values for the data.
            best_params: The best hyperparameters found from param_grid.
            best_estimator: The best estimator (model parameters/values) found.
            opt_model: The refitted model on training data with the best estimator.
        """

        # Define (static) scoring metrics used in hyperparameter tuning
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        scoring_metrics_dict = {metric: metric for metric in scoring_metrics}

        logging.info('Tuning hyperparameters and refitting model based on training data to create an optimized model (opt_model).')
        logging.info(f'Scoring metrics used in hyperparameter tuning: {scoring_metrics}')
        # Hyperparameter tuning for log_reg
        if self.algorithm == 'log_reg':
            self.param_grid= {
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0], 
                'C': [0.01, 0.1, 1, 10, 100], 
                    }
            grid_search = GridSearchCV(
                estimator=self.train_model, 
                param_grid=self.param_grid, 
                scoring=scoring_metrics_dict,  
                refit='accuracy', 
                cv=5,
                ) 
            grid_search.fit(self.X_train, self.y_train) 

            self.best_params = grid_search.best_params_ 
            logging.info(f'Best params found [logistic regression]: {self.best_params}')
            self.best_estimator = grid_search.best_estimator_ 
            logging.info(f'Best estimator found [logistic regression]: {self.best_estimator}')
        
            # Refit model using best params
            self.opt_model = LogisticRegression(penalty='elasticnet', solver='saga', random_state=42, **self.best_params).fit(self.X_train, self.y_train)
            logging.info('Refitted logistic regression with best params.')

        # Hyperparameter tuning for rf
        if self.algorithm == 'rf':

            self.param_grid = {
                'n_estimators': [50, 100, 200, 300, 400, 500],
                'criterion': ['gini', 'entropy'],
                'max_depth': [5, 10, 15],
                }
            grid_search = GridSearchCV(
                    estimator=self.train_model, 
                    param_grid=self.param_grid, 
                    scoring=scoring_metrics_dict, 
                    refit='accuracy', 
                    cv=5,
                ) 
            grid_search.fit(self.X_train, self.y_train)

            self.best_params = grid_search.best_params_
            logging.info(f'Best params found [random forest]: {self.best_params}')
            self.best_estimator = grid_search.best_estimator_
            logging.info(f'Best estimator found [random forest]: {self.best_estimator}')

            # Refit model using best params
            self.opt_model = RandomForestClassifier(**self.best_params, random_state=42).fit(self.X_train, self.y_train)
            logging.info('Refitted random forest with best params.')

        # Hyperparameter tuning for mlp
        if self.algorithm == 'mlp':

            self.param_grid = {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1], 
            }
            grid_search = GridSearchCV(
                    estimator=self.train_model, 
                    param_grid=self.param_grid, 
                    scoring=scoring_metrics_dict, 
                    refit='accuracy', 
                    cv=5,
                ) 
            grid_search.fit(self.X_train, self.y_train)
        
            self.best_params = grid_search.best_params_
            logging.info(f'Best params found [multi-layer perceptron classifier]: {self.best_params}')
            self.best_estimator = grid_search.best_estimator_
            logging.info(f'Best estimator found [multi-layer perceptron classifier]: {self.best_estimator}')

            # Refit model using best params 
            self.opt_model = MLPClassifier(**self.best_params, shuffle = True, activation='logistic', random_state=42, max_iter=1000).fit(self.X_train, self.y_train)
            logging.info('Refitted multi-layer perceptron classifier with best params.')
        
        return self.opt_model, self.best_estimator, self.best_params, self.param_grid


    def Feature_Importance_Opt(self)-> pd.DataFrame:
        """
        Compute feature importances for each optimized model (opt_model) using the selected features during training.
        
        Returns:
            feature_importance_df (pd.DataFrame) Feature importance dataframe for each model containing feature, importance, normalized importance.
        """

        # Compute feature importance scores for elastic net logistic regression model optimized model with coefficients
        if self.algorithm == 'log_reg':
            self.importance_scores = pd.Series(self.opt_model.coef_.flatten(), index=self.selected_features)
            self.normalized_importance = self.importance_scores / np.sum(self.importance_scores)
            logging.info('Computed feature importance for logistic regression.')

        # Compute feature importance scores for random forest optimized model with built in feature_importances_
        if self.algorithm == 'rf':
            self.importance_scores = pd.Series(self.opt_model.feature_importances_, index=self.selected_features)
            self.normalized_importance = self.importance_scores / np.sum(self.importance_scores)
            logging.info('Computed feature importance for random forest.')

        # Compute feature importance scores for multi-layer perceptron classifier optimized model with permutation_importance
        if self.algorithm == 'mlp':
            self.importance_scores = pd.Series(permutation_importance(self.opt_model, self.X_train, self.y_train, random_state=42).importances_mean, index=self.selected_features)
            self.importance_scores_mean = np.array(self.importance_scores.importances_mean)
            self.selected_features = self.X_train.columns
            self.normalized_importance = self.importance_scores_mean / np.sum(self.importance_scores_mean)
            logging.info('Computed feature importance for multi-layer perceptron classifier.')

        # Create a DataFrame of feature importances, coefficients, odds ratios, p-values, and CIs
        self.feature_importance_df = pd.DataFrame({
            'Feature': self.importance_scores.index,
            'Importance': self.importance_scores.values.round(2), 
            'Normalized Importance': self.normalized_importance.round(2),
        }).sort_values(by='Importance', ascending=False)
        logging.info('Created feature importance dataframe for relevant algorithm and thus optimized model.')
        logging.info('n\ Data Processing complete.')

        return self.importance_scores, self.normalized_importance, self.feature_importance_df
   

if __name__ == "__main__":
    
    # Create an argument parser for shell script (run.sh) to provide architecture, algorithm, train_df, and test_df
    parser = argparse.ArgumentParser(description='Provide algorithm and architecture.')
    parser.add_argument('--architecture', choices=['ctgan', 'fast_ml', 'gc'], required=True, help='Architecture to use (ctgan, fast_ml, gc)')
    parser.add_argument('--algorithm', choices=['log_reg', 'rf', 'mlp'], required=True, help='Algorithm to use (log_reg, rf, mlp)')
    parser.add_argument('--test_df', required=True, help='Path to test df to use')
    parser.add_argument('--train_df', required=True, help='Path to train df to use')
    args = parser.parse_args()
    
    post_process_data = DataProcessing(algorithm = args.algorithm, train_df=args.train_df, test_df=args.test_df)

    post_process_data.Balance_Smote()
    post_process_data.Initialize_Model()
    post_process_data.Feature_Selection_Train()
    post_process_data.Hyperparameter_Tuning()
    post_process_data.Feature_Importance_Opt()




