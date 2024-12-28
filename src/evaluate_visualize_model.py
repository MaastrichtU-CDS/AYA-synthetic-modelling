from data_processing import DataProcessing
import logging
import argparse

from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, log_loss, classification_report
from sklearn.model_selection import cross_val_score
from scipy import stats

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(filename='/root/log', filemode='a', encoding='utf-8', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EvaluateModel:
    """
    Class to evaluate the performance of the model using the following metrics: 
        Accuracy, balanced accuracy, precision, recall, f1, log loss, roc auc scores are computed.
        Cross validation scores are also computed.
        Confusion matrix, correlation matrix and ROC curve plots are generated.
    """

    def __init__(self, algorithm:str, train_df:pd.DataFrame, test_df:pd.DataFrame):
        """
        Initialize the class with the algorithm, training and testing dataframes.

        Parameters
            algorithm (str): The algorithm to use for modeling.
            train_df (pd.DataFrame): The training dataset.
            test_df (pd.DataFrame): The testing dataset.
        """
        logging.info(f'Initializing EvaluateModel class')

        self.algorithm = algorithm
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()

        self.processed_data = DataProcessing(self.algorithm, self.train_df, self.test_df)
        self.processed_data.Balance_Smote()
        self.processed_data.Initialize_Model()
        self.processed_data.Feature_Selection_Train()
        self.processed_data.Hyperparameter_Tuning()
        self.processed_data.Feature_Importance_Opt()

        self.X_train = self.processed_data.X_train
        self.y_train = self.processed_data.y_train
        self.X_test = self.processed_data.X_test
        self.y_test = self.processed_data.y_test
        self.opt_model = self.processed_data.opt_model

        self.y_pred_train = None
        self.y_proba_train = None
        self.y_pred_test = None
        self.y_proba_test = None

        self.metrics_train = None
        self.metrics_test = None

        self.train_cv_scores = None
        self.train_cv_scores_mean = None
        self.train_cv_scores_std = None
        self.test_cv_scores = None
        self.test_cv_scores_mean = None
        self.test_cv_scores_std = None

        self.odds_ratios = None
        self.coefs_mean = None
        self.se = None
        self.effect_mean = None
        self.effect_stdv = None
        self.coefs_stdv = None
        self.ci_lower_or = None
        self.ci_upper_or = None
        self.p_values = None
        self.Coef_Error_Plot = None

        self.Confusion_Matrix_Plot_Train = None
        self.Correlation_Plot_Train = None
        self.Roc_Curve_Plot_Train = None
        self.auc_train = None

        self.Confusion_Matrix_Plot_Test = None
        self.Correlation_Plot_Test = None
        self.Roc_Curve_Plot_Test = None
        self.auc_test = None

    def Model_Performance(self):
        """
        Evaluate the model performance using the following metrics:
            Accuracy, balanced accuracy, precision, recall, f1, log loss, roc auc scores
            Cross validation scores

        Returns:
            y_pred_train (np.array): Predicted values for training data.
            y_proba_train (np.array): Predicted probabilities for training data.
            metrics_train (dict): Dictionary of training metrics.
            train_cv_scores (np.array): Cross validation scores for training data.
            train_cv_scores_mean (float): Mean of cross validation scores for training data.
            train_cv_scores_std (float): Standard deviation of cross validation scores for training data.
            
            y_pred_test (np.array): Predicted values for testing data.
            y_proba_test (np.array): Predicted probabilities for testing data.
            metrics_test (dict): Dictionary of testing metrics.
            test_cv_scores (np.array): Cross validation scores for testing data.
            test_cv_scores_mean (float): Mean of cross validation scores for testing data.
            test_cv_scores_std (float): Standard deviation of cross validation scores for testing data.

        """

        logging.info(f"Model prediction scores {self.algorithm} are being computed")

        # Get model prediction scores for optimal model on train and test data
        self.y_pred_train = self.opt_model.predict(self.X_train)
        self.y_proba_train = self.opt_model.predict_proba(self.X_train)[:,1]
        self.y_pred_test = self.opt_model.predict(self.X_test)
        self.y_proba_test = self.opt_model.predict_proba(self.X_test)[:,1]
   
        # Compute Opt Model Training Metrics
        self.train_cv_scores = np.round(cross_val_score(self.opt_model, self.X_train, self.y_train, cv=5, scoring='roc_auc'), 2)
        self.train_cv_scores_mean = round(np.mean(self.train_cv_scores), 2)
        self.train_cv_scores_std = round(np.std(self.train_cv_scores), 2)

        ## Dictionary of training metrics
        self.metrics_train = {
            'Accuracy Score (Train)': round(accuracy_score(self.y_train, self.y_pred_train), 2), 
            'Balanced Accuracy Score (Train)': round(balanced_accuracy_score(self.y_train, self.y_pred_train), 2),
            'Precision Score (Train)': round(precision_score(self.y_train, self.y_pred_train), 2),
            'Recall Score (Train)': round(recall_score(self.y_train, self.y_pred_train), 2),
            'F1 Score (Train)': round(f1_score(self.y_train, self.y_pred_train), 2), 
            'Log Loss (Train)': round(log_loss(self.y_train, self.y_proba_train), 2), 
            'ROC AUC Score (Train)': round(roc_auc_score(self.y_train, self.y_proba_train), 2),
            'Cross Validation Scores (Train)': self.train_cv_scores,
            'Cross Validation Mean (Train)': self.train_cv_scores_mean,
            'Cross Validation Std (Train)': self.train_cv_scores_std
        }
        logging.info(f'Training metrics computed {self.metrics_train}')

        # Compute Opt Model Testing Metrics
        self.test_cv_scores = np.round(cross_val_score(self.opt_model, self.X_test, self.y_test, cv=5, scoring='roc_auc'), 2)
        self.test_cv_scores_mean = round(np.mean(self.test_cv_scores), 2)
        self.test_cv_scores_std = round(np.std(self.test_cv_scores), 2)
        
        ## Dictionary of testing metrics
        self.metrics_test = {
            'Accuracy Score (Test)': round(accuracy_score(self.y_test, self.y_pred_test), 2), 
            'Balanced Accuracy Score (Test)': round(balanced_accuracy_score(self.y_test, self.y_pred_test), 2),
            'Precision Score (Test)': round(precision_score(self.y_test, self.y_pred_test), 2),
            'Recall Score (Test)': round(recall_score(self.y_test, self.y_pred_test), 2),
            'F1 Score (Test)': round(f1_score(self.y_test, self.y_pred_test), 2), 
            'Log Loss (Test)': round(log_loss(self.y_test, self.y_proba_test), 2), 
            'ROC AUC Score (Test)': round(roc_auc_score(self.y_test, self.y_proba_test), 2),
            'Cross Validation Scores (Test)': self.test_cv_scores,
            'Cross Validation Mean (Test)': self.test_cv_scores_mean,
            'Cross Validation Std (Test)': self.test_cv_scores_std,
        }
        logging.info(f'Testing metrics computed {self.metrics_test}')

        if self.algorithm == 'log_reg':
            logging.info('Coefficients, standard errors, odds ratios, confidence intervals, p-values and effect sizes are being computed for logistic regression model')
            
            # Calculate coefficients
            self.coefs_opt_model = np.sort(self.opt_model.coef_.flatten())
            self.coefs_mean = round(np.mean(self.coefs_opt_model), 2)

            # Calculate standard deviation of coefficients
            self.coefs_stdv = round(np.std(self.coefs_opt_model), 2)

            # Calculate standard errors using the covariance matrix of the coefficients
            self.se = np.round(np.sqrt(np.diag(np.linalg.inv(np.dot(self.X_train.T, self.X_train)))), 2)

            # Calculate CIs for coefficients
            self.z = 1.96  # for 95% confidence interval
            self.ci_lower = self.coefs_opt_model - self.z * self.se
            self.ci_upper = self.coefs_opt_model + self.z * self.se

            # Convert coefficients to CIs
            self.ci_lower_or = np.round(np.exp(self.ci_lower), 2)
            self.ci_upper_or = np.round(np.exp(self.ci_upper), 2)
            self.odds_ratios = np.round(np.exp(self.coefs_opt_model), 2)

            # Calculate p-values
            self.p_values = 2 * (1 - stats.norm.cdf(np.abs(self.coefs_opt_model / self.se)))  # two-tailed test
            self.p_values = np.round(self.p_values, 2)

            # Calculate effect sizes
            self.effect_sizes = np.abs(self.coefs_opt_model)
            self.effect_mean = round(np.mean(self.effect_sizes), 2)
            self.effect_stdv = round(np.std(self.effect_sizes), 2)

        return self.y_pred_train, self.y_proba_train, self.y_pred_test, self.y_proba_test, self.metrics_train, self.metrics_test, self.train_cv_scores, self.train_cv_scores_mean, self.train_cv_scores_std, self.test_cv_scores, self.test_cv_scores_mean, self.test_cv_scores_std, self.se, self.coefs_stdv, self.effect_mean, self.coefs_mean, self.effect_stdv, self.odds_ratios, self.ci_lower_or, self.ci_upper_or, self.p_values
    
    def Make_Coef_Error(self):
        """
        Plot the coefficients with standard error as error bars for the logistic regression model.
        Calculate the odds ratios, confidence intervals, p-values and effect sizes for the coefficients of logistic regression model.

        Returns:
            coefs_opt_model (np.array): Coefficients of the optimized model.
            Coef_Error_Plot (plt.figure): Plot of coefficients with standard error as error bars.
            se (np.array): Standard errors of the coefficients.
            effect_sizes (np.array): Effect sizes of the coefficients.
            standard_deviation (float): Standard deviation of the coefficients.
        """

        if self.algorithm == 'log_reg':
            logging.info('Coefficients with standard error as error bars for logistic regression model computed')
            self.Coef_Error_Plot, ax_test = plt.subplots(figsize=(10, 8))
            ax_test.errorbar(self.coefs_opt_model, self.X_train.columns,
                              xerr=self.se, fmt='o', ecolor='r', capsize=3, capthick=1)
            ax_test.axvline(x=0, color='red', linestyle='--')
            ax_test.tick_params(axis='y', rotation=0, labelsize=12)
            ax_test.set_ylabel('Features', fontsize=14)
            ax_test.set_xlabel('Coefficient', fontsize=14)
            ax_test.set_title('Logistic Regression Coefficients', fontsize=16) # with standard error (se) as error bars
            ax_test.spines['top'].set_color('black')
            ax_test.spines['right'].set_color('black') 
            ax_test.spines['bottom'].set_color('black')
            ax_test.spines['left'].set_color('black')
            # Also set line width
            for spine in ax_test.spines.values():
                spine.set_linewidth(0.5)
            ax_test.set_facecolor('white')
            plt.show()

            return self.Coef_Error_Plot
        

    def Make_Conf(self):
        """
        Create and save confusion matrix plot for training and testing performance on optimized model.

        Returns:
            Confusion_Matrix_Plot_Train (plt.figure): Confusion matrix plot for training performance.
            Confusion_Matrix_Plot_Test (plt.figure): Confusion matrix plot for testing performance.
        """

        logging.info('Confusion matrix plots for training and testing performance on optimized model computed')

        self.conf_matrix_train = confusion_matrix(self.y_train, self.y_pred_train)
        self.Confusion_Matrix_Plot_Train, ax_train = plt.subplots(figsize=(10, 8)) 

        sns.heatmap(self.conf_matrix_train, annot=True, cmap='Blues', cbar=True, square=True, fmt='d',
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        ax_train.set_xlabel('Predicted')
        ax_train.set_ylabel('Actual')
        if self.algorithm == 'log_reg':
            ax_train.set_title('Logistic Regression Confusion Matrix (Training Set)', fontsize=14)
            logging.info('Logistic regression confusion matrix plot for training set created')
        elif self.algorithm == 'rf':
            ax_train.set_title('Random Forest Confusion Matrix (Training Set)', fontsize=14)
            logging.info('Random forest confusion matrix plot for training set created')
        elif self.algorithm == 'mlp':
            ax_train.set_title('Multi-layer Perceptron Classifier Confusion Matrix (Training Set)', fontsize=14)
            logging.info('Multi-layer perceptron classifier confusion matrix plot for training set created')
        plt.show()

        
        self.conf_matrix_test = confusion_matrix(self.y_test, self.y_pred_test)
        self.Confusion_Matrix_Plot_Test, ax_test = plt.subplots(figsize=(10, 8)) 

        sns.heatmap(self.conf_matrix_test, annot=True, cmap='Blues', cbar=True, square=True, fmt = 'd',
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        ax_test.set_xlabel('Predicted')
        ax_test.set_ylabel('Actual')
        if self.algorithm == 'log_reg':
            ax_test.set_title('Logistic Regression Confusion Matrix (Testing Set)', fontsize=14)
            logging.info('Logistic regression confusion matrix plot for testing set created')
        elif self.algorithm == 'rf':
            ax_test.set_title('Random Forest Confusion Matrix (Testing Set)', fontsize=14)
            logging.info('Random forest confusion matrix plot for testing set created')
        elif self.algorithm == 'mlp':
            ax_test.set_title('Multi-layer Perceptron Classifier Confusion Matrix (Testing Set)', fontsize=14)
            logging.info('Multi-layer perceptron classifier confusion matrix plot for testing set created')
        plt.show()

        return self.Confusion_Matrix_Plot_Train, self.Confusion_Matrix_Plot_Test

    def Make_Corr(self):
        """
        Create and save correlation matrix plot using Spearman correlation.

        Returns:
            Correlation_Plot_Train (plt.figure): Correlation matrix plot for feature correlation in train set.
            Correlation_Plot_Test (plt.figure): Correlation matrix plot for feature correlation in test set.
        """

        logging.info('Correlation matrix plots for training and testing on feature correlation computed using Spearman correlation')

        self.correlation_matrix_train = self.X_train.corr(method="spearman")
        self.Correlation_Plot_Train, ax_train = plt.subplots(figsize=(10, 8))
        ax_train.tick_params(axis='x', labelsize=12)
        ax_train.tick_params(axis='y', labelsize=12)
        sns.set_theme(font_scale=0.9)
        sns.heatmap(self.correlation_matrix_train, annot=True, cmap="Blues",  cbar=True, fmt='.2f', square=True)
        if self.algorithm == 'log_reg':
            ax_train.set_title('Logistic Regression Correlation Matrix (Training Set)', fontsize=14)
            logging.info('Logistic regression correlation matrix plot for training set created')
        elif self.algorithm == 'rf':
            ax_train.set_title('Random Forest Correlation Matrix (Training Set)', fontsize=14)
            logging.info('Random forest correlation matrix plot for training set created')
        elif self.algorithm == 'mlp':
            ax_train.set_title('Multi-layer Perceptron Classifier Correlation Matrix (Training Set)', fontsize=14)
            logging.info('Multi-layer perceptron classifier correlation matrix plot for training set created')
        plt.show()

        self.correlation_matrix_test = self.X_test.corr(method="spearman")
        self.Correlation_Plot_Test, ax_test = plt.subplots(figsize=(10, 8))
        ax_test.tick_params(axis='x', labelsize=12)
        ax_test.tick_params(axis='y', labelsize=12)
        sns.set_theme(font_scale=0.9)
        sns.heatmap(self.correlation_matrix_test, annot=True, cmap="Blues", cbar=True, fmt='.2f', square=True)
        if self.algorithm == 'log_reg':
            ax_test.set_title('Logistic Regression Correlation Matrix (Testing Set)', fontsize=14)
            logging.info('Logistic regression correlation matrix plot for testing set created')
        elif self.algorithm == 'rf':
            ax_test.set_title('Random Forest Correlation Matrix (Testing Set)', fontsize=14)
            logging.info('Random forest correlation matrix plot for testing set created')
        elif self.algorithm == 'mlp':
            ax_test.set_title('Multi-layer Perceptron Classifier Correlation Matrix (Testing Set)', fontsize=14)
            logging.info('Multi-layer perceptron classifier correlation matrix plot for testing set created')
        plt.show()

        return self.Correlation_Plot_Train, self.Correlation_Plot_Test

    def Make_AUC_ROC(self):
        """
        Create and save ROC curve plot to object for training and testing performance on optimized model.

        Returns:
            Roc_Curve_Plot_Train (plt.figure): AUC-ROC curve plot for training performance.
            Roc_Curve_Plot_Test (plt.figure): AUC-ROC curve plot for testing performance
        """

        logging.info('ROC curve plots for training and testing performance on optimized model created')

        self.fpr_train, self.tpr_train, _ = roc_curve(self.y_train, self.y_proba_train)
        self.auc_train = roc_auc_score(self.y_train, self.y_proba_train)
        logging.info(f'AUC ROC score for training set {self.auc_train:.2f}')

        self.Roc_Curve_Plot_Train, ax_train = plt.subplots(figsize=(10, 8))
        ax_train.plot(self.fpr_train, self.tpr_train, label=f'AUC = {self.auc_train:.2f}')
        ax_train.plot([0, 1], [0, 1], 'k--')
        ax_train.set_xlabel('False Positive Rate')
        ax_train.set_ylabel('True Positive Rate')
        ax_train.tick_params(axis='x')
        ax_train.tick_params(axis='y')
        ax_train.set_title('AUC-ROC Curve (Training Set)', fontsize=14)
        ax_train.legend(loc='best', facecolor='white')
        ax_train.spines['top'].set_color('black')
        ax_train.spines['right'].set_color('black')
        ax_train.spines['bottom'].set_color('black')
        ax_train.spines['left'].set_color('black')
        # Also set line width
        for spine in ax_train.spines.values():
            spine.set_linewidth(0.5)
        ax_train.set_facecolor('white')
        ax_train.grid(False)
        plt.show()

        self.fpr_test, self.tpr_test, _ = roc_curve(self.y_test, self.y_proba_test)
        self.auc_test = roc_auc_score(self.y_test, self.y_proba_test)
        logging.info(f'AUC ROC score for testing set {self.auc_test:.2f}')

        self.Roc_Curve_Plot_Test, ax_test = plt.subplots(figsize=(10, 8))
        ax_test.plot(self.fpr_test, self.tpr_test, label=f'AUC = {self.auc_test:.2f}')
        ax_test.plot([0, 1], [0, 1], 'k--')
        ax_test.set_xlabel('False Positive Rate')
        ax_test.set_ylabel('True Positive Rate')
        ax_test.tick_params(axis='x')
        ax_test.tick_params(axis='y')
        ax_test.set_title('AUC-ROC Curve (Testing Set)', fontsize=14)
        ax_test.legend(loc='best', facecolor='white')
        ax_test.spines['top'].set_color('black')
        ax_test.spines['right'].set_color('black') 
        ax_test.spines['bottom'].set_color('black')
        ax_test.spines['left'].set_color('black')
        # Also set line width
        for spine in ax_test.spines.values():
            spine.set_linewidth(0.5)
        ax_test.set_facecolor('white')
        ax_test.grid(False)
        plt.show()

        logging.info('Evaluations complete.')

        return self.Roc_Curve_Plot_Train, self.Roc_Curve_Plot_Test

if __name__ == "__main__":
    
     # Create an argument parser for shell script (run.sh) to provide architecture, algorithm, train_df, and test_df
    parser = argparse.ArgumentParser(description='Provide algorithm and architecture.')
    parser.add_argument('--architecture', choices=['ctgan', 'fast_ml', 'gc'], required=True, help='Architecture to use (ctgan, fast_ml, gc)')
    parser.add_argument('--algorithm', choices=['log_reg', 'rf', 'mlp'], required=True, help='Algorithm to use (log_reg, rf, mlp)')
    parser.add_argument('--test_df', required=True, help='Path to test df to use')
    parser.add_argument('--train_df', required=True, help='Path to train df to use')
    args = parser.parse_args()

    
    evaluate_data = EvaluateModel(algorithm=args.algorithm, train_df=args.train_df, test_df=args.test_df)  
    evaluate_data.Model_Performance()  
    evaluate_data.Make_Coef_Error()
    evaluate_data.Make_Conf()
    evaluate_data.Make_Corr()
    evaluate_data.Make_AUC_ROC()
    

