import logging
import os 

from data_pre_processing import DataPreProcessing
from data_processing import DataProcessing
from evaluation_metrics import EvaluateModel

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, PageBreak
from reportlab.lib import colors
import PyPDF2 
from PyPDF2 import PdfMerger

from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# save folder
# save plots
# PDF for metrics, feature importances, and plots


class FileSave:
    def __init__(self, df: pd.DataFrame, algorithm: str):
        self.df = df.copy()
        self.algorithm = algorithm

        self.processor = DataProcessing(self.df, self.algorithm)
        self.processor.Smote_Class_Balance()
        self.processor.Create_Model()
        self.processor.Feature_Selection()
        self.processor.Parameter_Tuning()
        self.processor.Feature_Importance()

        self.orig_train_size = self.processor.orig_train_size
        self.class_weight_OG = self.processor.class_weight_OG
        self.post_smote_train_size = self.processor.post_smote_train_size
        self.class_weight_smote = self.processor.class_weight_smote
        
        if self.algorithm == 'log_reg':
            self.feature_importance_log_df = self.processor.feature_importance_log_df
            self.best_estimator_log = self.processor.best_estimator_log
        
        if self.algorithm == 'rf':
            self.feature_importance_rf_df = self.processor.feature_importance_rf_df
            self.best_estimator_rf = self.processor.best_estimator_rf
        
        if self.algorithm == 'mlp':
            self.feature_importance_mlp_df = self.processor.feature_importance_mlp_df
            self.best_estimator_mlp = self.processor.best_estimator_mlp

        self.evaluator = EvaluateModel(self.df, self.algorithm)
        self.evaluator.Make_Metrics()
        self.metrics_ft = self.evaluator.metrics_ft


    def Create_Save_Folder(self, df_file_path):

        """

        Function to create (or locate) a folder to save the evaluation plots and output attributes to.

        Output:
            - save_folder: a folder within the results folder and with the name of the algorithm 
                for model performance evaluations.

        """

        now = datetime.now()
        self.date_time = now.strftime("%d-%m_%H-%M")
        self.date = now.strftime("%d-%m")
        
        self.df_file_path = df_file_path
        self.path_parts = self.df_file_path.split(os.sep) # Split the file path by the directory separator
        print(self.path_parts)

        self.root_folder = os.path.join(os.sep, self.path_parts[1])
        self.model_folder = os.path.join(self.root_folder, self.path_parts[2])
        self.results_folder = 'results'

        if self.algorithm == 'log_reg':
            self.results_folder = os.path.join(self.results_folder, 'log_reg')
        elif self.algorithm == 'mlp':
            self.results_folder = os.path.join(self.results_folder, 'mlp')
        elif self.algorithm == 'rf':
            self.results_folder = os.path.join(self.results_folder, 'rf')
    
        self.save_folder = os.path.join(self.model_folder, self.results_folder, f"{self.date}_Outputs") # root > ctgan > results
        
        # Folder path checks
        print(self.save_folder)
        print(self.root_folder)
        print(self.model_folder)
        print(self.results_folder)
        print(self.save_folder)

        # Check if the folders exist and create them if they do not
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
            print(f"Created folder: {self.save_folder}")
        else:
            print(f"Folder already exists: {self.save_folder}")

        return self.save_folder
    
    def Save_Eval_Plots(self):

        """

        Function to save the evaluation plots (which have their own '_path') to save_folder.

        Evaluation lots:
            - Confusion Matrix
            - Correlation Matrix
            - ROC Curve
            - Odds Ratio and Confidence Interval (OR_CI) plot (Logistic Regression only)
            - Normalized Importance plot (Random Forest and MLP only)
        
        """

        # Save the confusion matrix plot
        self.Confusion_Matrix_Plot = self.evaluator.Make_Conf()
        self.conf_mat_path = os.path.join(self.save_folder, f'{self.path_parts[2]}_confusion_matrix_plot_{self.date_time}.pdf') # Define the path to save the confusion matrix plot
        if hasattr(self.Confusion_Matrix_Plot, 'savefig'): # Check if the object has a savefig method
            self.Confusion_Matrix_Plot.savefig(self.conf_mat_path, format='pdf', bbox_inches='tight') 
            plt.close(self.Confusion_Matrix_Plot)
            print(f"Confusion matrix plot saved to: {self.conf_mat_path}")
        else:
            logging.error("The object returned by Make_Conf() does not have a savefig method.")

        # Save the correlation matrix plot
        self.Correlation_Plot = self.evaluator.Make_Corr()
        self.corr_mat_path = os.path.join(self.save_folder, f'{self.path_parts[2]}_correlation_matrix_plot_{self.date_time}.pdf')
        if hasattr(self.Correlation_Plot, 'savefig'):
            self.Correlation_Plot.savefig(self.corr_mat_path, format='pdf', bbox_inches='tight')
            plt.close(self.Correlation_Plot)
            print(f"Correlation matrix plot saved to: {self.corr_mat_path}")
        else:
            logging.error("The object returned by Make_Corr() does not have a savefig method.")

        # Save the ROC curve plot
        self.ROC_Curve_plot = self.evaluator.Make_ROC() 
        self.roc_curve_path = os.path.join(self.save_folder, f'{self.path_parts[2]}_roc_curve_plot_{self.date_time}.pdf')
        if hasattr(self.ROC_Curve_plot, 'savefig'):
            self.ROC_Curve_plot.savefig(self.roc_curve_path, format='pdf', bbox_inches='tight')
            plt.close(self.ROC_Curve_plot)
            print(f"ROC curve plot saved to: {self.roc_curve_path}")
        else:
            logging.error("The object returned by Make_ROC() does not have a savefig method.")

        # Save the odds ratio and confidence interval plot
        if self.algorithm == 'log_reg':
            self.OR_CI_Plot = self.evaluator.Make_OR_CI() 
            self.or_ci_path = os.path.join(self.save_folder, f'{self.path_parts[2]}_or_ci_plot_{self.date_time}.pdf')
            if hasattr(self.OR_CI_Plot, 'savefig'):
                self.OR_CI_Plot.savefig(self.or_ci_path, format='pdf', bbox_inches='tight')
                plt.close(self.OR_CI_Plot)
                print(f"Odds ratio and confidence interval plot saved to: {self.or_ci_path}")
            else:
                logging.error("The object returned by Make_OR_CI() does not have a savefig method.")
        
        # Save the normalized importance plot
        if self.algorithm == 'rf':
            self.Normalized_Importance_rf_Plot = self.evaluator.Make_Normalized_Importance()
            self.normalized_importance_rf_path = os.path.join(self.save_folder, f'{self.path_parts[2]}_normalized_importance_rf_plot_{self.date_time}.pdf')
            if hasattr(self.Normalized_Importance_rf_Plot, 'savefig'):
                self.Normalized_Importance_rf_Plot.savefig(self.normalized_importance_rf_path, format='pdf', bbox_inches='tight')
                plt.close(self.Normalized_Importance_rf_Plot)
                print(f"Normalized importance plot saved to: {self.normalized_importance_rf_path}")
            else:
                logging.error("The object returned by Make_Normalized_Importance() does not have a savefig method.")

        if self.algorithm == 'mlp':
            self.Normalized_Importance_mlp_Plot = self.evaluator.Make_Normalized_Importance()
            self.normalized_importance_mlp_path = os.path.join(self.save_folder, f'{self.path_parts[2]}_normalized_importance_mlp_plot_{self.date_time}.pdf')
            if hasattr(self.Normalized_Importance_mlp_Plot, 'savefig'):
                self.Normalized_Importance_mlp_Plot.savefig(self.normalized_importance_mlp_path, format='pdf', bbox_inches='tight')
                plt.close(self.Normalized_Importance_mlp_Plot)
                print(f"Normalized importance plot saved to: {self.normalized_importance_mlp_path}")
            else:
                logging.error("The object returned by Make_Normalized_Importance() does not have a savefig method.")

    
    def Output_Attributes(self):

        """
        Function to save the textual and plotted output attributes of model/performance to save_folder in a merged PDF file.

        Input:
            - metric_data_text: textual evaluation data (dict). Also saved to a csv file for ease of access.
            - feature_data_text: textual selected feature importance data per model (dict). Also saved to a csv file for ease of access.
            - Plots saved to a temp file and then to merged file,

        """
        
        elements = []

        # Add a title to the metric PDF
        title = f"{self.path_parts[2]} Metrics Report"
        elements.append(Paragraph(title, getSampleStyleSheet()['Title']))

        self.metric_data_text = {
            'Generator Model': self.path_parts[2],
            'Date': self.date_time,
            'Algorithm': self.algorithm,
            'Train size before smote': str(self.orig_train_size),
            'Class balance before smote': str(self.class_weight_OG),
            'Train size after smote': str(self.post_smote_train_size),
            'Class balance after smote': str(self.class_weight_smote),
            'Accuracy Score': self.metrics_ft["Accuracy Score"],
            'Balanced Accuracy Score': self.metrics_ft["Balanced Accuracy Score"],
            'Precision Score': self.metrics_ft["Precision Score"],
            'Recall Score': self.metrics_ft["Recall Score"],
            'F1 Score': self.metrics_ft["F1 Score"],
            'Log Loss': self.metrics_ft["Log Loss"],
            'ROC AUC Score': self.metrics_ft["ROC AUC Score"],
            'Best Estimators': self.best_estimator_log if self.algorithm == 'log_reg' else self.best_estimator_mlp if self.algorithm == 'mlp' else self.best_estimator_rf
        }

        # Swap columns and rows in the table
        metric_data = [[key] + [value] for key, value in self.metric_data_text.items()]
        metric_table = Table(metric_data)
        metric_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.black),
              ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
              ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
              ('FONTNAME', (0, 0), (-1, 0), 'Courier-Bold'),
              ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
              ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
              ('LEFTPADDING', (0, 0), (-1, -1), 6),
              ('RIGHTPADDING', (0, 0), (-1, -1), 6),
              ('TOPPADDING', (0, 0), (-1, -1), 6),
              ('BOTTOMPADDING', (0, 0), (-1, -1), 6),\
              ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
              ]))
        elements.append(metric_table)
        elements.append(PageBreak())



        if self.algorithm == 'log_reg':
            self.feature_data_text = {
            'Feature': self.feature_importance_log_df['Feature'],
            'Importance': self.feature_importance_log_df['Importance'],
            'Odds Ratio': self.feature_importance_log_df['Odds Ratio'],
            'CI Lower': self.feature_importance_log_df['CI Lower'],
            'CI Upper': self.feature_importance_log_df['CI Upper'],
            'P Value': self.feature_importance_log_df['P Value']
            }
        elif self.algorithm == 'mlp':
            self.feature_data_text = {
            'Feature': self.feature_importance_mlp_df['Feature'],
            'Importance': self.feature_importance_mlp_df['Importance'],
            'Normalized Importance': self.feature_importance_mlp_df['Normalized Importance']
            }
        elif self.algorithm == 'rf':
            self.feature_data_text = {
            'Feature': self.feature_importance_rf_df['Feature'],
            'Importance': self.feature_importance_rf_df['Importance'],
            'Normalized Importance': self.feature_importance_rf_df['Normalized Importance']
            }

        # Add a title to the feature importance PDF
        title = f"{self.path_parts[2]} Feature Importances"
        elements.append(Paragraph(title, getSampleStyleSheet()['Title']))
        
        # add data_text as a table to the PDF
        feature_data = [list(self.feature_data_text.keys())] + list(map(list, zip(*self.feature_data_text.values())))
        feature_table = Table(feature_data)
        feature_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.black),
                  ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                  ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                  ('FONTNAME', (0, 0), (-1, 0), 'Courier-Bold'),
                  ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                  ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
                  ('LEFTPADDING', (0, 0), (-1, -1), 6),
                  ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                  ('TOPPADDING', (0, 0), (-1, -1), 6),
                #   ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                  ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
                  ]))
        elements.append(feature_table) #FIXME: Rotate 90
        elements.append(PageBreak())


        # Save metric_ and feature_ data_text to a csv file
        pd.DataFrame([self.metric_data_text]).to_csv(os.path.join(self.save_folder, f'{self.path_parts[2]}_Metrics_{self.date_time}.csv'), index=False)
        pd.DataFrame(self.feature_data_text).to_csv(os.path.join(self.save_folder, f'{self.path_parts[2]}_Feature_Importance_{self.date_time}.csv'), index=False)

        # Create a temporary PDF file to hold the report content
        temp_report_pdf_file = os.path.join(self.save_folder, f'temp_report_{self.date_time}.pdf')
        doc = SimpleDocTemplate(temp_report_pdf_file)
        doc.build(elements)

        # Merge the PLOT PDFs
        merger = PdfMerger()
        merger.append(temp_report_pdf_file)

        merger.append(self.roc_curve_path)
        merger.append(self.conf_mat_path)
        merger.append(self.corr_mat_path)

        if self.algorithm == 'log_reg':
            merger.append(self.or_ci_path)

        if self.algorithm == 'rf':
            merger.append(self.normalized_importance_rf_path)

        if self.algorithm == 'mlp':
            merger.append(self.normalized_importance_mlp_path)

        merged_pdf_path = os.path.join(self.save_folder, f'{self.path_parts[2]}_Merged_Report_{self.date_time}.pdf')
        merger.write(merged_pdf_path)
        merger.close()
        print(f"Merged PDF saved to: {merged_pdf_path}")

        # Remove the temporary report PDF file
        os.remove(temp_report_pdf_file)
        

if __name__ == "__main__":
    df_file_path = input("Provide df file path: ")
    df = pd.read_csv(df_file_path, delimiter=',', encoding='latin1')


    save_to_file = FileSave(df, algorithm= input("Choose algorithm: log_reg, rf, mlp: "))
    save_to_file.Create_Save_Folder(df_file_path)
    save_to_file.Save_Eval_Plots()
    save_to_file.Output_Attributes()
