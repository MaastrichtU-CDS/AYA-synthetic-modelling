import os 
import argparse

from data_pre_processing import DataPreProcessing
from data_processing import DataProcessing
from evaluate_visualize_model import EvaluateModel

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

class FileSave:
    """
    Class to save model performance and evaluation plots and metrics.
    """

    def __init__(self, architecture, algorithm, test_df):
        """
        Create a for loop to iterate through the synthetic data files as training data and using the original data as testing data.
        Preprocess the data, balance the data, initialize the model, select features, tune the parameters, and evaluate the model.
        Process the data, evaluate the model, and save the evaluation plots and output attributes to the results folder. 

        Parameters:
            architecture (str): the architecture used to generate the synthetic data (ctgan, fast_ml, gc).
            algorithm (str): the algorithm used to model the data (log_reg, rf, mlp).
            test_df (str): the path to the test data to use for evaluation.
        """

        self.architecture = architecture
        self.algorithm = algorithm
        self.test_df = test_df
        self.test_df = pd.read_csv(self.test_df, delimiter=',', encoding='latin1')
        self.test_df = pd.DataFrame(self.test_df.copy())

        self.results_folder = None
        self.save_folder = None
        self.train_merged_folder = None
        self.train_plots_folder = None
        self.train_csv_folder = None
        self.test_merged_folder = None
        self.test_plots_folder = None
        self.test_csv_folder = None

        # Initialize the file reader which will be used to read the synthetic data files
        self.file_reader = None

        # Get synthetic data files
        self.data_files = [os.path.join(f'/root/{self.architecture}/synthetic_data', file) 
                           for file in os.listdir(f'/root/{self.architecture}/synthetic_data') if file.endswith('.csv')]

        # For loop to iterate through the synthetic data files as training data and using the original data as testing data to evaluate model performance.
        for file in self.data_files:

            self.train_df = pd.read_csv(file, delimiter=',', encoding='latin1')
            self.train_df = pd.DataFrame(self.train_df.copy())

            amsterdam_tz = pytz.timezone('Europe/Amsterdam') # Set timezone to Amsterdam
            self.now = datetime.now(amsterdam_tz) # Get the current date and time in Amsterdam timezone
            self.date_time = self.now.strftime("%d%m-%H%M") # Format the date and time as DDMM-HHMM
            self.date = self.now.strftime("%d%m") 

            # SD files have a run number in file name so that number is extracted and used for tracking purposes.
            self.run = '_'.join(file.split('/')[-1].split('_')[0:2])

            self.preprocessing = DataPreProcessing(self.train_df, self.test_df)
            self.preprocessing.Rename_Cols()
            self.preprocessing.Cat_Encoder()

            self.processed_data = DataProcessing(self.algorithm, self.train_df, self.test_df)
            self.processed_data.Balance_Smote()
            self.processed_data.Initialize_Model()
            self.processed_data.Feature_Selection_Train()
            self.processed_data.Hyperparameter_Tuning()
            self.processed_data.Feature_Importance_Opt()

            self.feature_importance_df = self.processed_data.feature_importance_df
            self.orig_train_size = self.processed_data.orig_train_size
            self.class_weight_before_smote = self.processed_data.class_weight_before_smote
            self.post_smote_train_size = self.processed_data.post_smote_train_size
            self.class_weight_smote = self.processed_data.class_weight_post_smote
            self.best_estimator = self.processed_data.best_estimator 

            self.evaluate_data = EvaluateModel(self.algorithm, self.train_df, self.test_df)  
            self.evaluate_data.Model_Performance()  
            self.metrics_train = self.evaluate_data.metrics_train
            self.metrics_test = self.evaluate_data.metrics_test

            # Create folders and save files for model performance evaluation plots and metrics for train and test data.
            self.Create_Folders()
            self.Save_Eval_Plots()
            self.Train_Output_Attributes()
            self.Test_Output_Attributes()
        
        print('Pipeline complete.')

    
    def Create_Folders(self):
        """
        Create or indicate the folder to save the evaluation plots and output attributes to.

        Returns:
            save_folder (str): an overarching folder for train and test data within the results folder to save the evaluation plots and output attributes for each set of runs.
            train_merged_folder (str): a folder within save_folder for the merged PDFs containing both the plots and CSVs of training data.
            train_plots_folder (str): a folder within save_folder for the individual evaluation plots of training data.
            train_csv_folder (str): a folder within save_folder for the textual output attributes as CSV files of training data.
            train_merged_folder (str): a folder within save_folder for the merged PDFs containing both the plots and CSVs of testing data.
            train_plots_folder (str): a folder within save_folder for the individual evaluation plots of testing data.
            train_csv_folder (str): a folder within save_folder for the textual output attributes as CSV files of testing data.
        """

        self.results_folder = f'/root/{self.architecture}/results/{self.algorithm}'
        self.save_folder = os.path.join(self.results_folder, f'{self.date}_Outputs') # root > {architecture} > results > {algorithm} > Outputs
        
        if len(self.data_files) == 1000:
            for i in range(len(self.data_files)):
                self.train_merged_folder = os.path.join(self.save_folder, 'Train_Merged_PDFs_1000_runs')
                self.train_plots_folder = os.path.join(self.save_folder, 'Train_Individual_Plots_1000_runs_')
                self.train_csv_folder = os.path.join(self.save_folder, 'Train_CSVs_1000_runs')
                self.test_merged_folder = os.path.join(self.save_folder, 'Test_Merged_PDFs_1000_runs')
                self.test_plots_folder = os.path.join(self.save_folder, 'Test_Individual_Plots_1000_runs')
                self.test_csv_folder = os.path.join(self.save_folder, 'Test_CSVs_1000_runs')
                
        # for testing purposes only
        else:
            for i in range(len(self.data_files)):
                self.train_merged_folder = os.path.join(self.save_folder, 'Train_Merged_PDFs_X_runs')
                self.train_plots_folder = os.path.join(self.save_folder, 'Train_Individual_Plots_X_runs')
                self.train_csv_folder = os.path.join(self.save_folder, 'Train_CSV_X_runs')
                self.test_merged_folder = os.path.join(self.save_folder, 'Test_Merged_PDFs_X_runs')
                self.test_plots_folder = os.path.join(self.save_folder, 'Test_Individual_Plots_X_runs')
                self.test_csv_folder = os.path.join(self.save_folder, 'Test_CSV_X_runs')
      

        # Check if the folders exist and create them if they do not
        for folder in [self.results_folder, self.save_folder, self.train_merged_folder, self.train_plots_folder, self.train_csv_folder, self.test_merged_folder, self.test_plots_folder, self.test_csv_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"Created folder: {folder}")
            else:
                print(f"Folder already exists: {folder}")

        # Folder path checks
        print('Save folder: ', self.save_folder)
        print('Results folder: ', self.results_folder)
        print('Train Merged folder: ', self.train_merged_folder)
        print('Train Plots older: ', self.train_plots_folder)
        print('Train CSV folder: ', self.train_csv_folder) 
        print('Test Merged folder: ', self.test_merged_folder)
        print('Test Plots folder: ', self.test_plots_folder)
        print('Test CSV folder: ', self.test_csv_folder)

        return self.save_folder, self.results_folder, self.train_merged_folder, self.train_plots_folder, self.train_csv_folder, self.test_merged_folder, self.test_plots_folder, self.test_csv_folder
    
    def Save_Eval_Plots(self):
        """
        Function to save the training and testing evaluation plots (which have their own '_path' to plots_folder.

        Evaluation plots:
            - Training Confusion Matrix Plot (plt.figure) 
            - Training Correlation Matrix Plot (plt.figure)
            - Training ROC Curve Plot (plt.figure)

            - Testing Confusion Matrix Plot (plt.figure)
            - Testing Correlation Matrix Plot (plt.figure)
            - Testing ROC Curve Plot (plt.figure)
            - Testing Coef Error Plot (plt.figure): Coefficients of optimized model with standard error bar (Logistic Regression only)
        """

        # Save the Train confusion matrix plot
        self.evaluate_data.Make_Conf()
        self.Confusion_Matrix_Plot_Train = self.evaluate_data.Confusion_Matrix_Plot_Train
        self.conf_mat_path_train = os.path.join(self.train_plots_folder, f'{self.date_time}_Confusion_Matrix_Plot_Train_{self.algorithm}_{self.run}_{self.architecture}.pdf')
        self.Confusion_Matrix_Plot_Train.savefig(self.conf_mat_path_train, format='pdf', bbox_inches='tight') 
        plt.close(self.Confusion_Matrix_Plot_Train)
        print(f"Confusion matrix plot (Train) saved to: {self.conf_mat_path_train}")

        # Save the Test confusion matrix plot
        self.Confusion_Matrix_Plot_Test = self.evaluate_data.Confusion_Matrix_Plot_Test
        self.conf_mat_path_test = os.path.join(self.test_plots_folder, f'{self.date_time}_Confusion_Matrix_Plot_Test_{self.algorithm}_{self.run}_{self.architecture}.pdf')
        self.Confusion_Matrix_Plot_Test.savefig(self.conf_mat_path_test, format='pdf', bbox_inches='tight')
        plt.close(self.Confusion_Matrix_Plot_Test)
        print(f"Confusion matrix plot (Test) saved to: {self.conf_mat_path_test}")
        

        # Save the Train correlation matrix plot
        self.evaluate_data.Make_Corr()
        self.Correlation_Plot_Train = self.evaluate_data.Correlation_Plot_Train
        self.corr_mat_path_train = os.path.join(self.train_plots_folder, f'{self.date_time}_Correlation_Matrix_Plot_Train_{self.algorithm}_{self.run}_{self.architecture}.pdf')
        self.Correlation_Plot_Train.savefig(self.corr_mat_path_train, format='pdf', bbox_inches='tight')
        plt.close(self.Correlation_Plot_Train)
        print(f"Correlation matrix plot (Train) saved to: {self.corr_mat_path_train}")

        # Save the Test correlation matrix plot
        self.Correlation_Plot_Test = self.evaluate_data.Correlation_Plot_Test
        self.corr_mat_path_test = os.path.join(self.test_plots_folder, f'{self.date_time}_Correlation_Matrix_Plot_Test_{self.algorithm}_{self.run}_{self.architecture}.pdf')
        self.Correlation_Plot_Test.savefig(self.corr_mat_path_test, format='pdf', bbox_inches='tight')
        plt.close(self.Correlation_Plot_Test)
        print(f"Correlation matrix plot (Test) saved to: {self.corr_mat_path_test}")
        
        # Save the Train ROC curve plot
        self.evaluate_data.Make_ROC()
        self.Roc_Curve_Plot_Train = self.evaluate_data.Roc_Curve_Plot_Train
        self.roc_curve_path_train = os.path.join(self.train_plots_folder, f'{self.date_time}_Roc_Curve_Plot_Train_{self.algorithm}_{self.run}_{self.architecture}.pdf')
        self.Roc_Curve_Plot_Train.savefig(self.roc_curve_path_train, format='pdf', bbox_inches='tight')
        plt.close(self.Roc_Curve_Plot_Train)
        print(f"ROC curve plot (Train) saved to: {self.roc_curve_path_train}")

        # Save the Test ROC curve plot
        self.Roc_Curve_Plot_Test = self.evaluate_data.Roc_Curve_Plot_Test
        self.roc_curve_path_test = os.path.join(self.test_plots_folder, f'{self.date_time}_Roc_Curve_Plot_Test_{self.algorithm}_{self.run}_{self.architecture}.pdf')
        self.Roc_Curve_Plot_Test.savefig(self.roc_curve_path_test, format='pdf', bbox_inches='tight')
        plt.close(self.Roc_Curve_Plot_Test)
        print(f"ROC curve plot (Test) saved to: {self.roc_curve_path_test}")
        
        # Save a plot of the coefficients with standard error of the optimal model (only for logistic regression)
        # Only for test data
        if self.algorithm == 'log_reg':
            self.evaluate_data.Make_Coef_Error()
            self.Coef_Error_Plot = self.evaluate_data.Coef_Error_Plot
            self.coef_error_path = os.path.join(self.test_plots_folder, f'{self.date_time}_Coef_Error_Plot_{self.algorithm}_{self.run}_{self.architecture}.pdf')
            self.Coef_Error_Plot.savefig(self.coef_error_path, format='pdf', bbox_inches='tight')
            plt.close(self.Coef_Error_Plot)
            print(f"Coef error plot (Test) saved to: {self.coef_error_path}")

    
    def Train_Output_Attributes(self):
        """
        Save the textual (CSV) and plotted output attributes (PDF) of the train model/performance to save_folder 
        Create a PDF with evaluation plots and save to the train_merged_PDF folder to a train_merged_PDF file.

        Attributes:
            train_metric_data_text (dict): textual evaluation data of training performance. Also saved to a csv file for ease of access.
            feature_data_text (dict): textual selected feature importance data (training only) for specified model. Also saved to a csv file for ease of access.
            Plots saved to a temp file and then to train_merged_folder.
        """
        
        elements = []

        # Add a title to the metric PDF
        title = f"Metrics Report Train [{self.architecture}]"
        elements.append(Paragraph(title, getSampleStyleSheet()['Title']))

        self.train_metric_data_text = {
            'Generator Model': self.architecture,
            'Date/Time': self.date_time,
            'Algorithm': self.algorithm,
            'Data Type': 'Train',
            'Run': self.run,
            'Train size before smote': str(self.orig_train_size),
            'Class balance before smote': str(self.class_weight_before_smote),
            'Train size after smote': str(self.post_smote_train_size),
            'Class balance after smote': str(self.class_weight_smote),
            'Best Estimator': self.best_estimator,
            'Accuracy Score': self.metrics_train["Accuracy Score (Train)"],
            'Balanced Accuracy Score': self.metrics_train["Balanced Accuracy Score (Train)"],
            'Precision Score': self.metrics_train["Precision Score (Train)"],
            'Recall Score': self.metrics_train["Recall Score (Train)"],
            'F1 Score': self.metrics_train["F1 Score (Train)"],
            'Log Loss': self.metrics_train["Log Loss (Train)"],
            'ROC AUC Score': self.metrics_train["ROC AUC Score (Train)"],
            'Cross Validation Scores': self.metrics_train["Cross Validation Scores (Train)"],
            'Cross Validation Mean': self.metrics_train["Cross Validation Mean (Train)"],
            'Cross Validation Std': self.metrics_train["Cross Validation Std (Train)"],
        }

        # Swap columns and rows in the table
        train_metric_data = [[key] + [value] for key, value in self.train_metric_data_text.items()]
        col_widths = [None, 250]
        row_heights = [20] * (len(train_metric_data) - 1) + [40]  # Set last row height to auto
        train_metric_table = Table(train_metric_data, rowHeights=row_heights, colWidths=col_widths)

        # Indicate table design/layout
        train_metric_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.black),
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
        elements.append(train_metric_table)
        elements.append(PageBreak())

        self.feature_data_text = {
            'Feature': self.feature_importance_df['Feature'],
            'Importance': self.feature_importance_df['Importance'],
            'Normalized Importance': self.feature_importance_df['Normalized Importance'],
        }

        # Add a title to the feature importance PDF
        title = f"Feature Importances [{self.architecture}]"
        elements.append(Paragraph(title, getSampleStyleSheet()['Title']))
        
        # Add data_text as a table to the PDF
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
        pd.DataFrame([self.train_metric_data_text]).to_csv(os.path.join(self.train_csv_folder, f'{self.date_time}_Train_Metrics_{self.run}_{self.architecture}.csv'), index=False)
        pd.DataFrame(self.feature_data_text).to_csv(os.path.join(self.train_csv_folder, f'{self.date_time}_Train_Feature_Importance_{self.run}_{self.architecture}.csv'), index=False)

        # Create a temporary PDF file to hold the report content
        temp_report_pdf_file_train = os.path.join(self.save_folder, f'{self.date_time}_train_temp_report_{self.run}.pdf')
        doc = SimpleDocTemplate(temp_report_pdf_file_train)
        doc.build(elements)

        # Merge the PLOT PDFs
        merger = PdfMerger()
        merger.append(temp_report_pdf_file_train)
        merger.append(self.roc_curve_path_train)
        merger.append(self.conf_mat_path_train)
        merger.append(self.corr_mat_path_train)

        merged_pdf_path_train = os.path.join(self.train_merged_folder, f'{self.date_time}_Train_Merged_Eval_Report_{self.run}_{self.architecture}.pdf')
        merger.write(merged_pdf_path_train)
        merger.close()
        print(f"Merged Evaluations PDF saved to: {merged_pdf_path_train}")

        # Remove the temporary report PDF file
        os.remove(temp_report_pdf_file_train)

    def Test_Output_Attributes(self):
        """
        Save the textual (CSV) and plotted output attributes (PDF) of testing (& optimized) model/performance to save_folder in a merged PDF file.
        Create a PDF with evaluation plots and save to the train_merged_PDF folder to a test_merged_PDF file.
              
        Attributes:
            test_metric_data_text (dict): textual evaluation data of testing model and perormance. Also saved to a csv file for ease of access.
            Plots saved to a temp file and then to test_merged_folder as a file.
        """
        
        elements = []

        # Add a title to the metric PDF
        title = f"Metrics Report Test [{self.architecture}]"
        elements.append(Paragraph(title, getSampleStyleSheet()['Title']))

        self.test_metric_data_text = {
            'Generator Model': self.architecture,
            'Date/Time': self.date_time,
            'Algorithm': self.algorithm,
            'Data Type': 'Test',
            'Run': self.run,
            'Best Estimator': self.best_estimator,
            'Accuracy Score': self.metrics_test["Accuracy Score (Test)"],
            'Balanced Accuracy Score': self.metrics_test["Balanced Accuracy Score (Test)"],
            'Precision Score': self.metrics_test["Precision Score (Test)"],
            'Recall Score': self.metrics_test["Recall Score (Test)"],
            'F1 Score': self.metrics_test["F1 Score (Test)"],
            'Log Loss': self.metrics_test["Log Loss (Test)"],
            'ROC AUC Score': self.metrics_test["ROC AUC Score (Test)"],
            'Cross Validation Scores': self.metrics_test["Cross Validation Scores (Test)"],
            'Cross Validation Mean': self.metrics_test["Cross Validation Mean (Test)"],
            'Cross Validation Std': self.metrics_test["Cross Validation Std (Test)"],
            'Coefs of Opt Model': self.evaluate_data.coefs_opt_model,
            'Effect Sizes of Opt Model': self.evaluate_data.effect_sizes
        }

        # Swap columns and rows in the table
        test_metric_data = [[key] + [value] for key, value in self.test_metric_data_text.items()]
        col_widths = [None, 250]
        row_heights = [20] * (len(test_metric_data) - 1) + [40]  # Set last row height to auto
        test_metric_table = Table(test_metric_data, rowHeights=row_heights, colWidths=col_widths)

        # Indicate table design/layout
        test_metric_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.black),
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
        elements.append(test_metric_table)
        elements.append(PageBreak())

        # Save test metrics text to a csv file
        pd.DataFrame([self.test_metric_data_text]).to_csv(os.path.join(self.test_csv_folder, f'{self.date_time}_Test_Metrics_{self.run}_{self.architecture}.csv'), index=False)

        # Create a temporary PDF file to hold the report content
        temp_report_pdf_file_test = os.path.join(self.save_folder, f'{self.date_time}_test_temp_report_{self.run}.pdf')
        doc = SimpleDocTemplate(temp_report_pdf_file_test)
        doc.build(elements)

        # Merge the PLOT PDFs
        merger = PdfMerger()
        merger.append(temp_report_pdf_file_test)
        merger.append(self.roc_curve_path_test)
        merger.append(self.conf_mat_path_test)
        merger.append(self.corr_mat_path_test)

        if self.algorithm == 'log_reg':
            merger.append(self.coef_error_path)

        merged_pdf_path_test = os.path.join(self.test_merged_folder, f'{self.date_time}_Test_Merged_Eval_Report_{self.run}_{self.architecture}.pdf')
        merger.write(merged_pdf_path_test)
        merger.close()
        print(f"Merged Evaluations PDF saved to: {merged_pdf_path_test}")

        # Remove the temporary report PDF file
        os.remove(temp_report_pdf_file_test)

if __name__ == "__main__":

    # Create an argument parser for shell script (run.sh) to provide architecture, algorithm, and test_df
    parser = argparse.ArgumentParser(description='Provide algorithm and architecture.')
    parser.add_argument('--architecture', choices=['ctgan', 'fast_ml', 'gc'], required=True, help='Architecture to use (ctgan, fast_ml, gc)')
    parser.add_argument('--algorithm', choices=['log_reg', 'rf', 'mlp'], required=True, help='Algorithm to use (log_reg, rf, mlp)')
    parser.add_argument('--test_df', required=True, help='Path to test df to use')
    args = parser.parse_args()

    print(f'Architecture: {args.architecture}')
    print(f'Algorithm: {args.algorithm}')
    
    save_to_file = FileSave(architecture=args.architecture, algorithm=args.algorithm, test_df=args.test_df)

