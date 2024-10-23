import logging
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

#TODO: PVALUE, ERROR

class EvaluateModel:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        self.model="rf"

        self.processor = DataProcessing(self.df)
        self.processor.Create_Model(self.df, model=self.model)
        self.processor.Feature_Selection(self.df, model=self.model)
        self.processor.Parameter_Tuning(self.df, model=self.model)
        self.processor.Feature_Importance(self.df, model=self.model)

        self.y_val = self.processor.y_val
        self.y_pred_ft = self.processor.y_pred_ft
        self.y_proba_ft = self.processor.y_proba_ft
        self.X_train_ft = self.processor.X_train_ft
        self.feature_importance_df = self.processor.feature_importance_df

        self.Confusion_Matrix_Plot = None
        self.Correlation_Plot = None
        self.ROC_Curve_plot = None
        self.OR_CI_Plot = None

    def Make_Conf(self):

        self.conf_matrix = confusion_matrix(self.y_val, self.y_pred_ft)
        plt.figure(figsize=(10, 10)) 
        # sns.set(font_scale=1.3)
        sns.heatmap(self.conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix', fontsize=13)

        self.Confusion_Matrix_Plot = io.BytesIO()
        plt.savefig(self.Confusion_Matrix_Plot, format='pdf')
        self.Confusion_Matrix_Plot.seek(0)
        
        
        # plt.savefig('Confusion_Matrix_Plot.pdf', format='pdf') # Save the figure to a file in src folder
        # Confusion_Matrix_Plot = os.path.join(target_folder, f'{model_name}_Confusion_Matrix_Plot_{date_time}.pdf')

        plt.show()

        return self.Confusion_Matrix_Plot

    def Make_Corr(self):
        self.correlation_matrix = self.X_train_ft.corr(method="spearman")
        plt.figure(figsize=(15, 15))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        sns.heatmap(self.correlation_matrix, annot=True, cmap="RdYlGn", linewidths=.5)
        plt.title(f"Correlation Matrix Heatmap of Selected Variables", fontsize=25)

        self.Correlation_Plot = io.BytesIO()
        plt.savefig(self.Correlation_Plot, format='pdf')
        self.Correlation_Plot.seek(0)

        # plt.savefig(os.path.join(target_folder, f'{model_name}_Correlation_Matrix_Plot_{date_time}.pdf'), format='pdf') # Save the figure to a file
        # correlation_matrix_plot = os.path.join(target_folder, f'{model_name}_Correlation_Matrix_Plot_{date_time}.pdf')

        plt.show()

        return self.Correlation_Plot

    def Make_ROC(self):
        self.fpr, self.tpr, _ = roc_curve(self.y_val, self.y_proba_ft[:, 1])
        self.auc = round(roc_auc_score(self.y_val, self.y_proba_ft[:, 1]), 3)
        plt.figure(figsize=(15, 15))
        plt.plot(self.fpr, self.tpr, label=f'AUC = {self.auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate', fontsize=18)
        plt.ylabel('True Positive Rate', fontsize=18)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        # plt.tight_layout()
        plt.title(f'ROC Curve', fontsize=20)
        plt.legend(loc='best')

        self.ROC_Curve_plot = io.BytesIO()
        plt.savefig(self.ROC_Curve_plot, format='pdf')
        self.ROC_Curve_plot.seek(0)
        
        # plt.savefig(os.path.join(target_folder, f'{model_name}_ROC_Curve_Plot_{date_time}.pdf'), format='pdf') # Save the figure to a file
        # roc_curve_plot = os.path.join(target_folder, f'{model_name}_ROC_Curve_Plot_{date_time}.pdf')

        plt.show()

        return self.ROC_Curve_plot


    def Make_OR_CI(self):
        plt.figure(figsize=(10, 10))
        plt.errorbar(self.feature_importance_df['Odds Ratio'], self.feature_importance_df['Feature'],
                    xerr=[(self.feature_importance_df['Odds Ratio'] - self.feature_importance_df['CI Lower']).values,
                          (self.feature_importance_df['CI Upper'] - self.feature_importance_df['Odds Ratio']).values],
                    fmt='o', ecolor='r', capsize=5, capthick=2)
        plt.axvline(x=1, color='gray', linestyle='--')
        plt.yticks(rotation=0, fontsize=11)
        plt.ylabel('Features', fontsize=11)
        plt.xlabel('Odds Ratio', fontsize=11)
        plt.title(f'Odds Ratios with 95% Confidence Intervals', fontsize=13)
        plt.tight_layout()

        # plt.savefig(os.path.join(target_folder, f'{model_name}_OR_CI_Plot_{date_time}.pdf'), format='pdf') # Save the figure to a file
        # or_ci_plot = os.path.join(target_folder, f'{model_name}_OR_CI_Plot_{date_time}.pdf')

        self.OR_CI_Plot = io.BytesIO()
        plt.savefig(self.OR_CI_Plot, format='pdf')
        self.OR_CI_Plot.seek(0)

        plt.show()

        return self.OR_CI_Plot


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
    # df_file_path = input("Provide df file path: ")

    df_file_path = "/root/ctgan/synthetic_data/survaya_03062022_clean_n_output_model_CTGAN_3735_samples.csv"
    df = pd.read_csv(df_file_path, delimiter=',', encoding='latin1')

    # model = "rf"
    
    evaluate_data = EvaluateModel(df)    
    evaluate_data.Make_Conf()
    evaluate_data.Make_Corr()
    # evaluate_data.Make_ROC()
    # evaluate_data.Make_OR_CI() #FIXME: Recheck code
    evaluate_data.Make_Metrics()