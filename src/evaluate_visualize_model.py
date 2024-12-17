from data_processing import DataProcessing

from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, log_loss, classification_report
from sklearn.model_selection import cross_val_score
from scipy import stats

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

        self.coefs_opt_model = None
        self.effect_sizes = None
        self.coefs_stdv = None
        self.se = None
        self.Coef_Error_Plot = None

        self.Confusion_Matrix_Plot_Train = None
        self.Correlation_Plot_Train = None
        self.Roc_Curve_Plot_Train = None

        self.Confusion_Matrix_Plot_Test = None
        self.Correlation_Plot_Test = None
        self.Roc_Curve_Plot_Test = None


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

        # Print dictionary of training metrics
        for metric, value in self.metrics_train.items():
            print(f'{metric}: {value}')


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

        # Print dictionary of testing metrics
        for metric, value in self.metrics_test.items():
            print(f'{metric}: {value}')  

        return self.y_pred_train, self.y_proba_train, self.y_pred_test, self.y_proba_test, self.metrics_train, self.metrics_test, self.train_cv_scores, self.train_cv_scores_mean, self.train_cv_scores_std, self.test_cv_scores, self.test_cv_scores_mean, self.test_cv_scores_std
    
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
            
            # Calculate coefficients
            self.coefs_opt_model = np.sort(self.opt_model.coef_.flatten())
            self.coef_mean = round(np.mean(self.coefs_opt_model), 2)

            # Calculate standard deviation of coefficients
            self.coefs_stdv = np.std(self.coefs_opt_model)
            print("Standard Deviation of Coefficients: ", self.coefs_stdv)

            # Calculate standard errors using the covariance matrix of the coefficients
            self.se = np.sqrt(np.diag(np.linalg.inv(np.dot(self.X_train.T, self.X_train))))

            # Calculate CIs for coefficients
            self.z = 1.96  # for 95% confidence interval
            self.ci_lower = self.coefs_opt_model - self.z * self.se
            self.ci_upper = self.coefs_opt_model + self.z * self.se

            # Convert coefficients to CIs
            self.ci_lower_or = np.exp(self.ci_lower)
            self.ci_upper_or = np.exp(self.ci_upper)
            self.odds_ratios = np.exp(self.coefs_opt_model)

            # Calculate p-values
            self.p_values = 2 * (1 - stats.norm.cdf(np.abs(self.coefs_opt_model / self.se)))  # two-tailed test
            self.p_values = np.round(self.p_values, 2)

            # Calculate effect sizes
            self.effect_sizes = np.abs(self.coefs_opt_model)
            self.effect_avg = round(np.mean(self.effect_sizes), 2)
            self.effect_stdv = round(np.std(self.effect_sizes), 2)

            self.Coef_Error_Plot, ax_test = plt.subplots(figsize=(10, 10))
            ax_test.errorbar(self.coefs_opt_model, self.X_train.columns,
                              xerr=self.se, fmt='o', ecolor='r', capsize=5, capthick=2)
            ax_test.axvline(x=0, color='gray', linestyle='--')
            ax_test.tick_params(axis='y', rotation=0, labelsize=11)
            ax_test.set_ylabel('Features', fontsize=14)
            ax_test.set_xlabel('Coefficient', fontsize=14)
            ax_test.set_title('Coefficients with standard error as Error Bars', fontsize=16)
            plt.show()

            return self.coefs_opt_model, self.Coef_Error_Plot, self.se, self.effect_sizes, self.coefs_stdv, self.effect_avg, self.coef_mean
        

    def Make_Conf(self):
        """
        Create and save confusion matrix plot for training and testing performance on optimized model.

        Returns:
            Confusion_Matrix_Plot_Train (plt.figure): Confusion matrix plot for training performance.
            Confusion_Matrix_Plot_Test (plt.figure): Confusion matrix plot for testing performance.
        """

        self.conf_matrix_train = confusion_matrix(self.y_train, self.y_pred_train)
        self.Confusion_Matrix_Plot_Train, ax_train = plt.subplots(figsize=(10, 10)) 

        sns.heatmap(self.conf_matrix_train, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        ax_train.set_xlabel('Predicted', fontsize=14)
        ax_train.set_ylabel('Actual', fontsize=14)
        ax_train.set_title(f'{self.algorithm} Train Confusion Matrix', fontsize=16)
        plt.show()

        
        self.conf_matrix_test = confusion_matrix(self.y_test, self.y_pred_test)
        self.Confusion_Matrix_Plot_Test, ax_test = plt.subplots(figsize=(10, 10)) 

        sns.heatmap(self.conf_matrix_test, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        ax_test.set_xlabel('Predicted', fontsize=14)
        ax_test.set_ylabel('Actual', fontsize=14)
        ax_test.set_title(f'{self.algorithm} Test Confusion Matrix', fontsize=16)
        plt.show()

        return self.Confusion_Matrix_Plot_Train, self.Confusion_Matrix_Plot_Test

    def Make_Corr(self):
        """
        Create and save correlation matrix plot using Spearman correlation for optimized model.

        Returns:
            Correlation_Plot_Train (plt.figure): Correlation matrix plot for training performance.
            Correlation_Plot_Test (plt.figure): Correlation matrix plot for testing performance.
        """

        self.correlation_matrix_train = self.X_train.corr(method="spearman")
        self.Correlation_Plot_Train, ax_train = plt.subplots(figsize=(10, 10))
        ax_train.tick_params(axis='x', labelsize=10)
        ax_train.tick_params(axis='y', labelsize=10)
        sns.set_theme(font_scale=0.8)
        sns.heatmap(self.correlation_matrix_train, annot=True, cmap="RdYlGn", linewidths=.5)
        ax_train.set_title(f'{self.algorithm} Train Correlation Matrix Heatmap of Selected Variables', fontsize=16)
        plt.show()

        self.correlation_matrix_test = self.X_test.corr(method="spearman")
        self.Correlation_Plot_Test, ax_test = plt.subplots(figsize=(10, 10))
        ax_test.tick_params(axis='x', labelsize=10)
        ax_test.tick_params(axis='y', labelsize=10)
        sns.set_theme(font_scale=0.8)
        sns.heatmap(self.correlation_matrix_test, annot=True, cmap="RdYlGn", linewidths=.5)
        ax_test.set_title(f'{self.algorithm} Test Correlation Matrix Heatmap of Selected Variables', fontsize=16)
        plt.show()

        return self.Correlation_Plot_Train, self.Correlation_Plot_Test

    def Make_ROC(self):
        """
        Create and save ROC curve plot to object for training and testing performance on optimized model.

        Returns:
            Roc_Curve_Plot_Train (plt.figure): ROC curve plot for training performance.
            Roc_Curve_Plot_Test (plt.figure): ROC curve plot for testing performance
        """

        self.fpr_train, self.tpr_train, _ = roc_curve(self.y_train, self.y_proba_train)
        self.auc_train = roc_auc_score(self.y_train, self.y_proba_train)

        self.Roc_Curve_Plot_Train, ax_train = plt.subplots(figsize=(10, 10))
        ax_train.plot(self.fpr_train, self.tpr_train, label=f'AUC = {self.auc_train:.2f}')
        ax_train.plot([0, 1], [0, 1], 'k--')
        ax_train.set_xlabel('False Positive Rate', fontsize=14)
        ax_train.set_ylabel('True Positive Rate', fontsize=14)
        ax_train.tick_params(axis='x', labelsize=10)
        ax_train.tick_params(axis='y', labelsize=10)
        ax_train.set_title(f'{self.algorithm} Train ROC Curve', fontsize=16)
        ax_train.legend(loc='best')
        plt.show()

        self.fpr_test, self.tpr_test, _ = roc_curve(self.y_test, self.y_proba_test)
        self.auc_test = roc_auc_score(self.y_test, self.y_proba_test)

        self.Roc_Curve_Plot_Test, ax_test = plt.subplots(figsize=(10, 10))
        ax_test.plot(self.fpr_test, self.tpr_test, label=f'AUC = {self.auc_test:.2f}')
        ax_test.plot([0, 1], [0, 1], 'k--')
        ax_test.set_xlabel('False Positive Rate', fontsize=14)
        ax_test.set_ylabel('True Positive Rate', fontsize=14)
        ax_test.tick_params(axis='x', labelsize=10)
        ax_test.tick_params(axis='y', labelsize=10)
        ax_test.set_title(f'{self.algorithm} Test ROC Curve', fontsize=16)
        ax_test.legend(loc='best')
        plt.show()

        return self.Roc_Curve_Plot_Train, self.Roc_Curve_Plot_Test


if __name__ == "__main__":
    algorithm = input("Provide algorithm (log_reg, rf, mlp): ")
    
    train_path = input("Input the training data file path: ")
    train_df = pd.read_csv(train_path, delimiter=',', encoding='latin1')
    test_path = input("Input the testing data file path: ")
    test_df = pd.read_csv(test_path, delimiter=',', encoding='latin1')
    
    evaluate_data = EvaluateModel(algorithm, train_df, test_df)  
    evaluate_data.Model_Performance()  
    evaluate_data.Make_Coef_Error()
    evaluate_data.Make_Conf()
    evaluate_data.Make_Corr()
    evaluate_data.Make_ROC()
