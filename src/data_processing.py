from data_pre_processing import DataPreProcessing

import json

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import  GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance

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
        print (f'\n Original training data size: {self.orig_train_size}')

        # Compute class weights before SMOTE
        self.class_weight_before_smote = compute_class_weight(class_weight='balanced', classes=np.unique(self.y_train), y=self.y_train)

        # Apply SMOTE to balance the dataset
        self.X_train, self.y_train = SMOTE(random_state=42).fit_resample(self.X_train, self.y_train)
        
        # Get train size after smote
        self.post_smote_train_size = self.X_train.shape
        print (f'\n Training data size after SMOTE: {self.post_smote_train_size}')

        # Compute class weights after SMOTE
        self.class_weight_post_smote = compute_class_weight(class_weight='balanced', classes=np.unique(self.y_train), y=self.y_train)
        
        print (f'\n Class weights before SMOTE: {self.class_weight_before_smote}, \n Class weights after SMOTE: {self.class_weight_post_smote}')

        return self.X_train, self.y_train, self.class_weight_before_smote, self.class_weight_post_smote, self.orig_train_size, self.post_smote_train_size
    

    def Initialize_Model(self) -> pd.DataFrame:
        """
        Initialize model based on the chosen algorithm.
        All models are initialized with random_state=42.

        Returns:
            train_model: The initialized model for the chosen algorithm.
        """

        # Elastic Net Logistic Regression (log_reg)
        # Initialized with standard elastic net parameters (l1_ratio=0.5, C=0.1)
        if self.algorithm == 'log_reg':
            self.train_model = LogisticRegression(penalty='elasticnet', solver ='saga', l1_ratio=0.5, C = 0.1, max_iter=1500, random_state=42)    

        # Random Forest Classifier (rf)
        if self.algorithm == 'rf':
            self.train_model = RandomForestClassifier(random_state=42) 

        # Multi-layer Perceptron Classifier (mlp)
        # Initialized with the logistic activation function for binary classification.
        if self.algorithm == 'mlp':
            self.train_model = MLPClassifier(shuffle=True, activation='logistic', random_state=42, max_iter=1000)
        
        # Fit the model on the training data
        self.train_model.fit(self.X_train, self.y_train)

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

        if self.algorithm=="log_reg": 
            self.feature_selector = SelectFromModel(estimator = self.train_model).fit(self.X_train, self.y_train)
            self.selected_features_mask = self.feature_selector.get_support()
            self.selected_features = self.X_train.columns[self.selected_features_mask]
            self.X_train = pd.DataFrame(self.feature_selector.transform(self.X_train), columns=self.selected_features)
            print("\n See selected features in X_train [log_reg]: \n ")
            self.X_train.info()

        if self.algorithm=="rf":
            self.feature_selector = self.train_model.feature_importances_
            self.threshold = sorted(self.feature_selector, reverse=True)[int(len(self.feature_selector) * 0.2)]  # Select top 20% features
            self.selected_features_mask = self.feature_selector >= self.threshold
            self.selected_features = self.X_train.columns[self.selected_features_mask]
            self.X_train = pd.DataFrame(self.X_train.loc[:, self.selected_features])
            print("\n See selected features in X_train [rf]: \n ")
            self.X_train.info()
        
        if self.algorithm=="mlp":
            self.feature_selector = permutation_importance(self.train_model, self.X_train, self.y_train, random_state=42, scoring='accuracy') 
            self.feature_selector_mean = np.array(self.feature_selector.importances_mean)
            self.threshold = sorted(self.feature_selector_mean, reverse=True)[int(len(self.feature_selector_mean) * 0.2)]  # Select top 20% features
            self.selected_features_mask = self.feature_selector_mean >= self.threshold
            self.selected_features = self.X_train.columns[self.selected_features_mask]
            self.X_train = pd.DataFrame(self.X_train.loc[:, self.selected_features])
            print("\n See selected features in X_train [mlp]: \n ")
            self.X_train.info()

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
            print("\n Best params found [log_reg]: ", self.best_params)
            self.best_estimator = grid_search.best_estimator_ 
            print("\n Best estimator found [log_reg]: ", self.best_estimator)
        
            # Refit model using best params
            self.opt_model = LogisticRegression(penalty='elasticnet', solver='saga', random_state=42, **self.best_params).fit(self.X_train, self.y_train)
            print("\n Done refitting log_reg with best estimator")


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
            print("\n Best params found [rf]: ", self.best_params)
            self.best_estimator = grid_search.best_estimator_
            print("\n Best estimator found [rf]: ", self.best_estimator)

            
            # Refit model using best params
            self.opt_model = RandomForestClassifier(**self.best_params, random_state=42).fit(self.X_train, self.y_train)
            print("\n Done refitting rf with best estimator")

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
            print("\n Best params found [mlp]: ", self.best_params)
            self.best_estimator = grid_search.best_estimator_
            print("\n Best estimator found [mlp]: ", self.best_estimator)

            # Refit model using best params 
            self.opt_model = MLPClassifier(**self.best_params, shuffle = True, activation='logistic', random_state=42, max_iter=1000).fit(self.X_train, self.y_train)
            print("\n Done refitting mlp with best estimator")

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
            print("\n Feature importance for log_reg: ")

        # Compute feature importance scores for random forest optimized model with built in feature_importances_
        if self.algorithm == 'rf':
            self.importance_scores = self.opt_model.feature_importances_
            self.normalized_importance = self.importance_scores / np.sum(self.importance_scores)
            print("\n Feature importance for rf: ")

        # Compute feature importance scores for multi-layer perceptron classifier optimized model with permutation_importance
        if self.algorithm == 'mlp':
            self.importance_scores = permutation_importance(self.opt_model, self.X_train, self.y_train, random_state=42)
            self.importance_scores_mean = np.array(self.importance_scores.importances_mean)
            self.selected_features = self.X_train.columns
            self.normalized_importance = self.importance_scores_mean / np.sum(self.importance_scores_mean)
            print("\n Feature importance for mlp: ")

        # Create a DataFrame of feature importances, coefficients, odds ratios, p-values, and CIs
        self.feature_importance_df = pd.DataFrame({
            'Feature': self.importance_scores.index,
            'Importance': self.importance_scores.values.round(2), 
            'Normalized Importance': self.normalized_importance.round(2),
        }).sort_values(by='Importance', ascending=False)
        print(self.feature_importance_df)

        return self.importance_scores, self.normalized_importance, self.feature_importance_df
   

if __name__ == "__main__":

    algorithm = input("Provide algorithm (log_reg, rf, mlp): ")
    train_path = input("Input the training data file path: ")
    train_df = pd.read_csv(train_path, delimiter=',', encoding='latin1')
    test_path = input("Input the testing data file path: ")
    test_df = pd.read_csv(test_path, delimiter=',', encoding='latin1')
    
    post_process_data = DataProcessing(algorithm, train_df, test_df)
    post_process_data.Balance_Smote()
    post_process_data.Initialize_Model()
    post_process_data.Feature_Selection_Train()
    post_process_data.Hyperparameter_Tuning()
    post_process_data.Feature_Importance_Opt()




