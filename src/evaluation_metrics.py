import os 
import io
from data_pre_processing import DataPreProcessing
from data_processing import DataProcessing
import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, log_loss, classification_report

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix
# Correlation matrix
# ROC curve
# OR CI plot
# Normalized importance
# Metrics


class EvaluateModel:
    def __init__(self, df: pd.DataFrame, algorithm: str):
        self.df = df.copy()
        self.algorithm = algorithm

        self.processor = DataProcessing(self.df, self.algorithm)
        self.processor.Smote_Class_Balance()
        self.processor.Create_Model()
        self.processor.Feature_Selection()
        self.processor.Parameter_Tuning()
        self.processor.Feature_Importance()

        self.y_val = self.processor.y_val
        self.y_pred_ft = self.processor.y_pred_ft
        self.y_proba_ft = self.processor.y_proba_ft
        self.X_train_ft = self.processor.X_train_ft

        self.feature_importance_log_df = self.processor.feature_importance_log_df
        self.feature_importance_rf_df = self.processor.feature_importance_rf_df
        self.feature_importance_mlp_df = self.processor.feature_importance_mlp_df
        
        self.Confusion_Matrix_Plot = None
        self.Correlation_Plot = None
        self.ROC_Curve_plot = None
        self.OR_CI_Plot = None
        self.Normalized_Importance_log_Plot = None
        self.Normalized_Importance_rf_Plot = None
        self.Normalized_Importance_mlp_Plot = None
    
        self.metrics_ft = None


    def Make_Conf(self):

        """

        Function to create and save confusion matrix plot to object.
        
        """

        self.conf_matrix = confusion_matrix(self.y_val, self.y_pred_ft)

        self.Confusion_Matrix_Plot, ax = plt.subplots(figsize=(10, 10)) 

        sns.heatmap(self.conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        ax.set_xlabel('Predicted', fontsize=14)
        ax.set_ylabel('Actual', fontsize=14)
        ax.set_title(f'{self.algorithm} Confusion Matrix', fontsize=16)

        plt.show()

        return self.Confusion_Matrix_Plot

    def Make_Corr(self):

        """

        Function to create and save correlation matrix plot to object 
            using Spearman correlation to account for non-linear relationships.
        
        """

        self.correlation_matrix = self.X_train_ft.corr(method="spearman")

        self.Correlation_Plot, ax = plt.subplots(figsize=(10, 10))
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        sns.set_theme(font_scale=0.8)
        sns.heatmap(self.correlation_matrix, annot=True, cmap="RdYlGn", linewidths=.5)
        ax.set_title(f'{self.algorithm} Correlation Matrix Heatmap of Selected Variables', fontsize=16)

        plt.show()

        return self.Correlation_Plot

    def Make_ROC(self):

        """

        Function to create and save ROC curve plot to object.
        
        """

        self.fpr, self.tpr, _ = roc_curve(self.y_val, self.y_proba_ft[:, 1])
        self.auc = roc_auc_score(self.y_val, self.y_proba_ft[:, 1])

        self.ROC_Curve_plot, ax = plt.subplots(figsize=(10, 10))

        ax.plot(self.fpr, self.tpr, label=f'AUC = {self.auc:.2f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

        ax.set_title(f'{self.algorithm} ROC Curve', fontsize=16)
        ax.legend(loc='best')

        plt.show()

        return self.ROC_Curve_plot


    def Make_OR_CI(self):

        """

        Function to create and save Odds ratio plot with 95% confidence intervals for the selected features.
        
        Only for log_reg because rf nor mlp don't have odds ratio.
        
        """

        self.OR_CI_Plot, ax = plt.subplots(figsize=(10, 10))

        if self.algorithm == 'log_reg':
            ax.errorbar(self.feature_importance_log_df['Odds Ratio'], self.feature_importance_log_df['Feature'],
                    xerr=[(self.feature_importance_log_df['Odds Ratio'] - self.feature_importance_log_df['CI Lower']).values,
                            (self.feature_importance_log_df['CI Upper'] - self.feature_importance_log_df['Odds Ratio']).values],
                    fmt='o', ecolor='r', capsize=5, capthick=2)
       
            ax.axvline(x=1, color='gray', linestyle='--')
            ax.tick_params(axis='y', rotation=0, labelsize=11)
            ax.set_ylabel('Features', fontsize=14)
            ax.set_xlabel('Odds Ratio', fontsize=14)
            ax.set_title(f'{self.algorithm} Odds Ratios with 95% Confidence Intervals', fontsize=16)

            plt.show()

            return self.OR_CI_Plot
    
    def Make_Normalized_Importance(self):

        """

        Function to 
        
            1. Create and save normalized importance plot.
            2. Sort the DataFrame by normalized importance for better visualization.
            3. Plot and return the normalized importances.

        """

        if self.algorithm == "log_reg":
                
            self.Normalized_Importance_log_Plot, ax = plt.subplots(figsize=(10, 10))
    
            # Sort the DataFrame by normalized importance for better visualization
            self.feature_importance_log_df = self.feature_importance_log_df.sort_values(by='Normalized Importance', ascending=False)
            
            # Plotting the normalized importances
            ax.barh(self.feature_importance_log_df['Feature'], self.feature_importance_log_df['Normalized Importance'], color='skyblue')
            
            ax.set_xlabel('Normalized Importance')
            ax.set_ylabel('Feature')
            ax.set_title(f'{self.algorithm} Normalized Feature Importances')
            ax.grid(True)
            plt.show()
                
            return self.Normalized_Importance_log_Plot

        if self.algorithm == "rf":

            self.Normalized_Importance_rf_Plot, ax = plt.subplots(figsize=(10, 10))
    
            # Sort the DataFrame by normalized importance for better visualization
            self.feature_importance_rf_df = self.feature_importance_rf_df.sort_values(by='Normalized Importance', ascending=False)
            
            # Plotting the normalized importances
            ax.barh(self.feature_importance_rf_df['Feature'], self.feature_importance_rf_df['Normalized Importance'], color='skyblue')
            
            ax.set_xlabel('Normalized Importance')
            ax.set_ylabel('Feature')
            ax.set_title(f'{self.algorithm} RF Normalized Feature Importances')
            ax.grid(True)
            plt.show()
            
            return self.Normalized_Importance_rf_Plot
        
        if self.algorithm == "mlp":

            self.Normalized_Importance_mlp_Plot, ax = plt.subplots(figsize=(10, 10))
    
            # Sort the DataFrame by normalized importance for better visualization
            self.feature_importance_mlp_df = self.feature_importance_mlp_df.sort_values(by='Normalized Importance', ascending=False)
            
            # Plotting the normalized importances
            ax.barh(self.feature_importance_mlp_df['Feature'], self.feature_importance_mlp_df['Normalized Importance'], color='skyblue')
            
            ax.set_xlabel('Normalized Importance')
            ax.set_ylabel('Feature')
            ax.set_title(f'{self.algorithm} MLP Normalized Feature Importances')
            ax.grid(True)
            plt.show()
            
            return self.Normalized_Importance_mlp_Plot

    def Make_Metrics(self):
        
        """

        Function to save other evaluation metrics to object: metrics_ft. 
        
        'ft' indicates that the object accounts for selected features.
        
        """

        self.metrics_ft = {
            'Accuracy Score': round(accuracy_score(self.y_val, self.y_pred_ft), 3),
            'Balanced Accuracy Score': round(balanced_accuracy_score(self.y_val, self.y_pred_ft), 3),
            'Precision Score': round(precision_score(self.y_val, self.y_pred_ft), 3),
            'Recall Score': round(recall_score(self.y_val, self.y_pred_ft), 3),
            'F1 Score': round(f1_score(self.y_val, self.y_pred_ft), 3),
            'Log Loss': round(log_loss(self.y_val, self.y_proba_ft), 3),
            'ROC AUC Score': round(roc_auc_score(self.y_val, self.y_proba_ft[:, 1]), 3)
        }

        for metric, value in self.metrics_ft.items():
            print(f'{metric}: {value}')

        return self.metrics_ft


if __name__ == "__main__":
    df_file_path = input("Provide df file path: ")
    df = pd.read_csv(df_file_path, delimiter=',', encoding='latin1')
    
    evaluate_data = EvaluateModel(df, algorithm= input("Choose algorithm: log_reg, rf, mlp: "))    
    evaluate_data.Make_Conf()
    evaluate_data.Make_Corr()
    evaluate_data.Make_ROC()
    evaluate_data.Make_OR_CI()
    evaluate_data.Make_Normalized_Importance()
    evaluate_data.Make_Metrics()
