import logging
import os 
from data_pre_processing import DataPreProcessing
from data_processing import DataProcessing
import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, log_loss, classification_report

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#TODO: PVALUE, ERROR

class EvaluateModel:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        self.evaluator = DataProcessing(self.df)
        self.evaluator.Create_Model(self.df)
        self.evaluator.Feature_Selection(self.df)
        self.evaluator.Parameter_Tuning(self.df)
        self.evaluator.Feature_Importance(self.df)

        self.y_val = self.evaluator.y_val
        self.y_pred_ft = self.evaluator.y_pred_ft
        self.y_proba_ft = self.evaluator.y_proba_ft
        self.X_train_ft = self.evaluator.X_train_ft
        self.feature_importance_df = self.evaluator.feature_importance_df

    def Make_Conf(self):
        conf_matrix = confusion_matrix(self.y_val, self.y_pred_ft)
        plt.figure(figsize=(10, 10)) 
        sns.set(font_scale=1.3)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix', fontsize=13)
        
        # plt.savefig(os.path.join(target_folder, f'{model_name}_Confusion_Matrix_Plot_{date_time}.pdf'), format='pdf') # Save the figure to a file
        # confusion_matrix_plot = os.path.join(target_folder, f'{model_name}_Confusion_Matrix_Plot_{date_time}.pdf')

        plt.show()
        plt.close()

    def Make_Corr(self):
        correlation_matrix = self.X_train_ft.corr(method="spearman")
        plt.figure(figsize=(15, 15))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        sns.heatmap(correlation_matrix, annot=True, cmap="RdYlGn", linewidths=.5)
        plt.title(f"Correlation Matrix Heatmap of Selected Variables", fontsize=25)

        # plt.savefig(os.path.join(target_folder, f'{model_name}_Correlation_Matrix_Plot_{date_time}.pdf'), format='pdf') # Save the figure to a file
        # correlation_matrix_plot = os.path.join(target_folder, f'{model_name}_Correlation_Matrix_Plot_{date_time}.pdf')

        plt.show()
        plt.close()

    def Make_ROC(self):
        fpr, tpr, _ = roc_curve(self.y_val, self.y_proba_ft[:, 1])
        auc = round(roc_auc_score(self.y_val, self.y_proba_ft[:, 1]), 3)
        plt.figure(figsize=(15, 15))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate', fontsize=18)
        plt.ylabel('True Positive Rate', fontsize=18)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        # plt.tight_layout()
        plt.title(f'ROC Curve', fontsize=20)
        plt.legend(loc='best')
        
        # plt.savefig(os.path.join(target_folder, f'{model_name}_ROC_Curve_Plot_{date_time}.pdf'), format='pdf') # Save the figure to a file
        # roc_curve_plot = os.path.join(target_folder, f'{model_name}_ROC_Curve_Plot_{date_time}.pdf')

        plt.show()
        plt.close()

    def Make_OR_CI(self):
        plt.figure(figsize=(10, 10))
        plt.errorbar(self.feature_importance_df['Odds Ratio'], self.feature_importance_df['Feature'],
                    xerr=[self.feature_importance_df['Odds Ratio'] - self.feature_importance_df['CI Lower'],
                        self.feature_importance_df['CI Upper'] - self.feature_importance_df['Odds Ratio']],
                    fmt='o', ecolor='r', capsize=5, capthick=2)
        plt.axvline(x=1, color='gray', linestyle='--')
        plt.yticks(rotation=0, fontsize=11)
        plt.ylabel('Features', fontsize=11)
        plt.xlabel('Odds Ratio', fontsize=11)
        plt.title(f'Odds Ratios with 95% Confidence Intervals', fontsize=13)
        plt.tight_layout()

        # plt.savefig(os.path.join(target_folder, f'{model_name}_OR_CI_Plot_{date_time}.pdf'), format='pdf') # Save the figure to a file
        # or_ci_plot = os.path.join(target_folder, f'{model_name}_OR_CI_Plot_{date_time}.pdf')

        plt.show()
        plt.close()

    def Make_Metrics(self):
        metrics_ft = [
            ['Accuracy Score', round(accuracy_score(self.y_val, self.y_pred_ft), 3)],
            ['Balanced Accuracy Score', round(balanced_accuracy_score(self.y_val, self.y_pred_ft), 3)],
            ['Precision Score', round(precision_score(self.y_val, self.y_pred_ft), 3)],
            ['Recall Score', round(recall_score(self.y_val, self.y_pred_ft), 3)],
            ['F1 Score', round(f1_score(self.y_val, self.y_pred_ft), 3)],
            ['Log Loss', round(log_loss(self.y_val, self.y_proba_ft), 3)],
            ['ROC AUC Score', round(roc_auc_score(self.y_val, self.y_proba_ft[:, 1]), 3)]
        ]

        for metric, value in metrics_ft:
            print(f'{metric}: {value}')


if __name__ == "__main__":
    df_file_path = input("Provide df file path: ")
    df = pd.read_csv(df_file_path, delimiter=',', encoding='latin1')
    
    evaluate_data = EvaluateModel(df)    
    evaluate_data.Make_Conf()
    evaluate_data.Make_Corr()
    evaluate_data.Make_ROC()
    evaluate_data.Make_OR_CI()
    evaluate_data.Make_Metrics()