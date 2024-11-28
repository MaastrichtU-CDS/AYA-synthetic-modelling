import os 
import csv

from data_pre_processing import DataPreProcessing
from data_processing import DataProcessing
from evaluation_metrics import EvaluateModel

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import pytz

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, PageBreak
from reportlab.lib import colors
from PyPDF2 import PdfMerger

from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# save folder
# save plots
# PDF for metrics, feature importances, and plots

class FileSave:
    def __init__(self):

        self.results_folder = None #f'{self.model_folder}/results/mlp'
        self.save_folder = None
        self.merged_folder = None
        self.plots_folder = None
        self.csv_folder = None
        self.file_reader = None
        
        self.model_folder = input("Choose model: ctgan, fast_ml, gc: ")
        self.algorithm = input("Choose algorithm: log_reg, rf, mlp: ")

        if self.algorithm == 'log_reg':
            self.results_folder = f'/root/{self.model_folder}/results/log_reg'
        
        if self.algorithm == 'rf':
            self.results_folder = f'/root/{self.model_folder}/results/rf'
        
        if self.algorithm == 'mlp':
            self.results_folder = f'/root/{self.model_folder}/results/mlp'

        self.data_files = [os.path.join(f'/root/{self.model_folder}/synthetic_data', file) 
                           for file in os.listdir(f'/root/{self.model_folder}/synthetic_data') if file.endswith('.csv')]

        for file in self.data_files:
            self.df = pd.read_csv(file, delimiter=',', encoding='latin1')
            self.df = pd.DataFrame(self.df.copy())

            amsterdam_tz = pytz.timezone('Europe/Amsterdam') # Set timezone to Amsterdam
            self.now = datetime.now(amsterdam_tz) # Get the current date and time in Amsterdam timezone
            self.date_time = self.now.strftime("%d%m-%H%M") # Format the date and time as DDMM-HHMM
            self.date = self.now.strftime("%d%m") 

            self.run = '_'.join(file.split('/')[-1].split('_')[0:2])

            self.preprocessing = DataPreProcessing(self.df)
            self.preprocessing.Rename_Cols(self.df)
            self.preprocessing.Cat_Encoder(self.df)
            self.preprocessing.Ttv_Split()
            self.chosen_split = self.preprocessing.chosen_split
            self.X = self.preprocessing.X

            self.processor = DataProcessing(self.df, self.algorithm)
            self.processor.Smote_Class_Balance()
            self.processor.Create_Model()
            self.processor.Feature_Selection()
            self.processor.Parameter_Tuning()
            self.processor.Feature_Importance()

            self.orig_train_size = self.processor.orig_train_size
            self.class_weight_OG = self.processor.class_weight_pre_smote
            self.post_smote_train_size = self.processor.post_smote_train_size
            self.class_weight_smote = self.processor.class_weight_post_smote
            self.X_train_ft = self.processor.X_train_ft
            

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

            self.now = datetime.now()
            print(self.now)
            self.date_time = self.now.strftime("%d%m_%H%M") # day month hour minute
            print(self.date_time)
            self.date = self.now.strftime("%d%m") # day month

            self.Create_Folders()
            self.Save_Eval_Plots()
            self.Output_Attributes()

    
    def Create_Folders(self):

        """

        Function to create (or locate) a folder to save the evaluation plots and output attributes to.

        Output:
            - save_folder: a folder within the results folder and with the name of the algorithm 
                for model performance evaluations.
            - merged_folder: a folder to save the merged PDFs of the evaluation plots and output attributes.
            - plots_folder: a folder to save the individual evaluation plots.
            - csv_folder: a folder to save the output attributes as CSV files.

        """

        self.save_folder = os.path.join(self.results_folder, "Outputs") # root > ctgan > results > model > Outputs
        self.merged_folder = os.path.join(self.save_folder, f'{self.date}_Merged_PDFs')
        self.plots_folder = os.path.join(self.save_folder, f'{self.date}_Individual_Plots')
        self.csv_folder = os.path.join(self.save_folder, f'{self.date}_CSVs')

        # Folder path checks
        print('save folder:', self.save_folder)
        print('merged folder:', self.merged_folder)
        print('plots older:', self.plots_folder)
        print('csv folder:', self.csv_folder) 

        # Check if the folders exist and create them if they do not
        for folder in [self.save_folder, self.merged_folder, self.plots_folder, self.csv_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"Created folder: {folder}")
            else:
                print(f"Folder already exists: {folder}")

        return self.save_folder, self.merged_folder, self.plots_folder, self.csv_folder
    
    def Save_Eval_Plots(self):

        """

        Function to save the evaluation plots (which have their own '_path') to save_folder.

        Evaluation plots:
            - Confusion Matrix
            - Correlation Matrix
            - ROC Curve
            - Odds Ratio and Confidence Interval (OR_CI) plot (Logistic Regression only)
            - Normalized Importance plot (Random Forest and MLP only)
        
        """

        # Save the confusion matrix plot
        self.Confusion_Matrix_Plot = self.evaluator.Make_Conf()
        self.conf_mat_path = os.path.join(self.plots_folder, f'{self.date_time}_Confusion_Matrix_Plot_{self.run}_{self.model_folder}.pdf') # Define the path to save the confusion matrix plot
        
        if hasattr(self.Confusion_Matrix_Plot, 'savefig'): # Check if the object has a savefig method
            self.Confusion_Matrix_Plot.savefig(self.conf_mat_path, format='pdf', bbox_inches='tight') 
            plt.close(self.Confusion_Matrix_Plot)
            print(f"Confusion matrix plot saved to: {self.conf_mat_path}")
        else:
            print("The object returned by Make_Conf() does not have a savefig method.")

        # Save the correlation matrix plot
        self.Correlation_Plot = self.evaluator.Make_Corr()
        self.corr_mat_path = os.path.join(self.plots_folder, f'{self.date_time}_Correlation_Matrix_Plot_{self.run}_{self.model_folder}.pdf')
        
        if hasattr(self.Correlation_Plot, 'savefig'):
            self.Correlation_Plot.savefig(self.corr_mat_path, format='pdf', bbox_inches='tight')
            plt.close(self.Correlation_Plot)
            print(f"Correlation matrix plot saved to: {self.corr_mat_path}")
        else:
            print("The object returned by Make_Corr() does not have a savefig method.")

        # Save the ROC curve plot
        self.ROC_Curve_plot = self.evaluator.Make_ROC() 
        self.roc_curve_path = os.path.join(self.plots_folder, f'{self.date_time}_ROC_Curve_Plot_{self.run}_{self.model_folder}.pdf')
        
        if hasattr(self.ROC_Curve_plot, 'savefig'):
            self.ROC_Curve_plot.savefig(self.roc_curve_path, format='pdf', bbox_inches='tight')
            plt.close(self.ROC_Curve_plot)
            print(f"ROC curve plot saved to: {self.roc_curve_path}")
        else:
            print("The object returned by Make_ROC() does not have a savefig method.")

        # Save the odds ratio and confidence interval plot
        if self.algorithm == 'log_reg':
            self.OR_CI_Plot = self.evaluator.Make_OR_CI() 
            self.or_ci_path = os.path.join(self.plots_folder, f'{self.date_time}_OR_CI_Plot_{self.run}_{self.model_folder}.pdf')
            
            if hasattr(self.OR_CI_Plot, 'savefig'):
                self.OR_CI_Plot.savefig(self.or_ci_path, format='pdf', bbox_inches='tight')
                plt.close(self.OR_CI_Plot)
                print(f"Odds ratio and confidence interval plot saved to: {self.or_ci_path}")
            else:
                print("The object returned by Make_OR_CI() does not have a savefig method.")
        
        # Save the normalized importance plot
        if self.algorithm == 'rf':
            self.Normalized_Importance_rf_Plot = self.evaluator.Make_Normalized_Importance()
            self.normalized_importance_rf_path = os.path.join(self.plots_folder, f'{self.date_time}_Normalized_Importance_RF_Plot_{self.run}_{self.model_folder}.pdf')
            
            if hasattr(self.Normalized_Importance_rf_Plot, 'savefig'):
                self.Normalized_Importance_rf_Plot.savefig(self.normalized_importance_rf_path, format='pdf', bbox_inches='tight')
                plt.close(self.Normalized_Importance_rf_Plot)
                print(f"Normalized importance plot saved to: {self.normalized_importance_rf_path}")
            else:
                print("The object returned by Make_Normalized_Importance() does not have a savefig method.")

        if self.algorithm == 'mlp':
            self.Normalized_Importance_mlp_Plot = self.evaluator.Make_Normalized_Importance()
            self.normalized_importance_mlp_path = os.path.join(self.plots_folder, f'{self.date_time}_Normalized_Importance_MLP_Plot_{self.run}_{self.model_folder}.pdf')
           
            if hasattr(self.Normalized_Importance_mlp_Plot, 'savefig'):
                self.Normalized_Importance_mlp_Plot.savefig(self.normalized_importance_mlp_path, format='pdf', bbox_inches='tight')
                plt.close(self.Normalized_Importance_mlp_Plot)
                print(f"Normalized importance plot saved to: {self.normalized_importance_mlp_path}")
            else:
                print("The object returned by Make_Normalized_Importance() does not have a savefig method.")
    
    def Output_Attributes(self):

        """

        Function to save the textual and plotted output attributes of model/performance to save_folder in a merged PDF (and/or csv) file.

        Input:
            - metric_data_text: textual evaluation data (dict). Also saved to a csv file for ease of access.
            - feature_data_text: textual selected feature importance data per model (dict). Also saved to a csv file for ease of access.
            - Plots saved to a temp file and then to merged file and folder.

        """
        
        elements = []

        # Add a title to the metric PDF
        title = f"Metrics Report [{self.model_folder}]"
        elements.append(Paragraph(title, getSampleStyleSheet()['Title']))

        self.metric_data_text = {
            'Generator Model': self.model_folder,
            'Date/Time': self.date_time,
            'Algorithm': self.algorithm,
            'Run': self.run,
            'Train, test, val split': str(self.chosen_split),
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
            'Best Estimators': self.best_estimator_log if self.algorithm == 'log_reg' else self.best_estimator_rf if self.algorithm == 'rf' else "N/A"
        }

        # Swap columns and rows in the table
        metric_data = [[key] + [value] for key, value in self.metric_data_text.items()]
        col_widths = [None, 250]
        row_heights = [20] * (len(metric_data) - 1) + [40]  # Set last row height to auto
        metric_table = Table(metric_data, rowHeights=row_heights, colWidths=col_widths)

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
        title = f"Feature Importances [{self.model_folder}]"
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
                  ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
                  ]))
        elements.append(feature_table)
        elements.append(PageBreak())

        # Save metric_ and feature_ data_text to a csv file
        pd.DataFrame([self.metric_data_text]).to_csv(os.path.join(self.csv_folder, f'{self.date_time}_Metrics_{self.run}_{self.model_folder}.csv'), index=False)
        pd.DataFrame(self.feature_data_text).to_csv(os.path.join(self.csv_folder, f'{self.date_time}_Feature_Importance_{self.run}_{self.model_folder}.csv'), index=False)

        # Create a temporary PDF file to hold the report content
        temp_report_pdf_file = os.path.join(self.save_folder, f'{self.date_time}_temp_report_{self.run}.pdf')
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

        merged_pdf_path = os.path.join(self.merged_folder, f'{self.date_time}_Merged_Eval_Report_{self.run}_{self.model_folder}.pdf')
        merger.write(merged_pdf_path)
        merger.close()
        print(f"Merged Evaluations PDF saved to: {merged_pdf_path}")

        # Remove the temporary report PDF file
        os.remove(temp_report_pdf_file)

if __name__ == "__main__":

    save_to_file = FileSave()



