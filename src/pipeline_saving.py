import os 
import argparse
import logging

from data_pre_processing import DataPreProcessing
from data_processing import DataProcessing
from evaluate_visualize_model import EvaluateModel

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, PageBreak
from reportlab.lib import colors
from PyPDF2 import PdfMerger

from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet

logging.basicConfig(filename='/root/log', filemode='a', encoding='utf-8', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        self.train_save_folder = None
        self.test_save_folder = None
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
        for i, file in enumerate(self.data_files):
            self.pipe_loop = i
            logging.info(f"\n Pipeline run: {self.pipe_loop}")

            self.train_df = pd.read_csv(file, delimiter=',', encoding='latin1')
            self.train_df = pd.DataFrame(self.train_df.copy())

            # SD files have a run number in file name so that number is extracted and used for tracking purposes.
            self.sd_run = '_'.join(file.split('/')[-1].split('_')[0:2])

            logging.info(f"Pre-processing data from SD run: {self.sd_run}")
            self.preprocessing = DataPreProcessing(self.train_df, self.test_df)
            self.preprocessing.Rename_Cols()
            self.preprocessing.Cat_Encoder()

            logging.info(f"Processing data from SD run: {self.sd_run}")
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
            self.best_params = self.processed_data.best_params

            logging.info(f"Evaluating data from SD run: {self.sd_run}")
            self.evaluate_data = EvaluateModel(self.algorithm, self.train_df, self.test_df)  
            self.evaluate_data.Model_Performance()  
            self.metrics_train = self.evaluate_data.metrics_train
            self.metrics_test = self.evaluate_data.metrics_test
            self.coefs_stdv = self.evaluate_data.coefs_stdv
            self.coefs_mean = self.evaluate_data.coefs_mean
            self.se = self.evaluate_data.se
            self.odds_ratios = self.evaluate_data.odds_ratios
            self.ci_lower_or = self.evaluate_data.ci_lower_or
            self.ci_upper_or = self.evaluate_data.ci_upper_or
            self.p_values = self.evaluate_data.p_values
            self.effect_mean = self.evaluate_data.effect_mean
            self.effect_stdv = self.evaluate_data.effect_stdv

            # Create folders and save files for model performance evaluation plots and metrics for train and test data.
            self.Create_Folders()
            self.Save_Eval_Plots()
            self.Train_Output_Attributes()
            self.Test_Output_Attributes()
        
        logging.info('Pipeline complete.')

    
    def Create_Folders(self):
        """
        Create or indicate the folder to save the evaluation plots and output attributes to.

        Returns:
            results_folder (str): a folder within the architecture and algorithm folders to save the evaluation plots and output attributes for each set of runs.
            
            train_save_folder (str): a folder within results folder for the evaluation plots and output attributes of training data.
            train_merged_folder (str): a folder within train_save_folder for the merged PDFs containing both the plots and CSVs of training data.
            train_plots_folder (str): a folder within train_save_folder for the individual evaluation plots of training data.
            train_csv_folder (str): a folder within train_save_folder for the textual output attributes as CSV files of training data.
            
            test_save_folder (str): a folder within results folder for the evaluation plots and output attributes of testing data.
            test_merged_folder (str): a folder within test_save_folder for the merged PDFs containing both the plots and CSVs of testing data.
            test_plots_folder (str): a folder within test_save_folder for the individual evaluation plots of testing data.
            test_csv_folder (str): a folder within test_save_folder for the textual output attributes as CSV files of testing data.
        """

        logging.info(f"Creating folders for {self.architecture}, {self.algorithm}")

        self.results_folder = f'/root/{self.architecture}/results/{self.algorithm}/pipeline_run/Outputs'
        
        # Create a folder for the results of 1000 runs
        if len(self.data_files) == 1000:
            for i in range(len(self.data_files)):
                self.train_save_folder = os.path.join(self.results_folder, 'Train_1000_runs')
                self.train_merged_folder = os.path.join(self.train_save_folder, 'Train_Merged_PDFs')
                self.train_plots_folder = os.path.join(self.train_save_folder, 'Train_Individual_Plots')
                self.train_csv_folder = os.path.join(self.train_save_folder, 'Train_CSVs')
                self.test_save_folder = os.path.join(self.results_folder, 'Test_1000_runs')
                self.test_merged_folder = os.path.join(self.test_save_folder, 'Test_Merged_PDFs')
                self.test_plots_folder = os.path.join(self.test_save_folder, 'Test_Individual_Plots')
                self.test_csv_folder = os.path.join(self.test_save_folder, 'Test_CSVs')
                
        # for testing purposes only
        # Create a folder for few runs used for pipeline testing purposes
        else:
            for i in range(len(self.data_files)):
                self.train_save_folder = os.path.join(self.results_folder, 'Train_X_runs')
                self.test_save_folder = os.path.join(self.results_folder, 'Test_X_runs')

        # Check if the folders exist and create them if they do not
        for folder in [self.results_folder, self.train_save_folder, self.test_save_folder, self.train_merged_folder, self.train_plots_folder, self.train_csv_folder, self.test_merged_folder, self.test_plots_folder, self.test_csv_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                logging.info(f"Created folder: {folder}")
            else:
                logging.info(f"Folder already exists: {folder}")

        return self.results_folder, self.train_save_folder, self.test_save_folder, self.train_merged_folder, self.train_plots_folder, self.train_csv_folder, self.test_merged_folder, self.test_plots_folder, self.test_csv_folder
    
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

        logging.info(f"Saving evaluation plots for {self.architecture}, {self.algorithm}")

        # Save the Train confusion matrix plot
        self.evaluate_data.Make_Conf()
        self.Confusion_Matrix_Plot_Train = self.evaluate_data.Confusion_Matrix_Plot_Train
        self.conf_mat_path_train = os.path.join(self.train_plots_folder, f'Confusion_Matrix_Plot_Train_{self.algorithm}_{self.sd_run}_{self.architecture}.pdf')
        self.Confusion_Matrix_Plot_Train.savefig(self.conf_mat_path_train, format='pdf', bbox_inches='tight') 
        plt.close(self.Confusion_Matrix_Plot_Train)
        logging.info(f"Confusion matrix plot (Train) saved to train_plots_folder.")

        # Save the Test confusion matrix plot
        self.Confusion_Matrix_Plot_Test = self.evaluate_data.Confusion_Matrix_Plot_Test
        self.conf_mat_path_test = os.path.join(self.test_plots_folder, f'Confusion_Matrix_Plot_Test_{self.algorithm}_{self.sd_run}_{self.architecture}.pdf')
        self.Confusion_Matrix_Plot_Test.savefig(self.conf_mat_path_test, format='pdf', bbox_inches='tight')
        plt.close(self.Confusion_Matrix_Plot_Test)
        logging.info(f"Confusion matrix plot (Test) saved to test_plots_folder.")
        
        # Save the Train correlation matrix plot
        self.evaluate_data.Make_Corr()
        self.Correlation_Plot_Train = self.evaluate_data.Correlation_Plot_Train
        self.corr_mat_path_train = os.path.join(self.train_plots_folder, f'Correlation_Matrix_Plot_Train_{self.algorithm}_{self.sd_run}_{self.architecture}.pdf')
        self.Correlation_Plot_Train.savefig(self.corr_mat_path_train, format='pdf', bbox_inches='tight')
        plt.close(self.Correlation_Plot_Train)
        logging.info(f"Correlation matrix plot (Train) saved to train_plots_folder.")

        # Save the Test correlation matrix plot
        self.Correlation_Plot_Test = self.evaluate_data.Correlation_Plot_Test
        self.corr_mat_path_test = os.path.join(self.test_plots_folder, f'Correlation_Matrix_Plot_Test_{self.algorithm}_{self.sd_run}_{self.architecture}.pdf')
        self.Correlation_Plot_Test.savefig(self.corr_mat_path_test, format='pdf', bbox_inches='tight')
        plt.close(self.Correlation_Plot_Test)
        logging.info(f"Correlation matrix plot (Test) saved to test_plots_folder.")
        
        # Save the Train ROC curve plot
        self.evaluate_data.Make_AUC_ROC()
        self.Roc_Curve_Plot_Train = self.evaluate_data.Roc_Curve_Plot_Train
        self.roc_curve_path_train = os.path.join(self.train_plots_folder, f'Roc_Curve_Plot_Train_{self.algorithm}_{self.sd_run}_{self.architecture}.pdf')
        self.Roc_Curve_Plot_Train.savefig(self.roc_curve_path_train, format='pdf', bbox_inches='tight')
        plt.close(self.Roc_Curve_Plot_Train)
        logging.info(f"ROC curve plot (Train) saved to train_plots_folder.")

        # Save the Test ROC curve plot
        self.Roc_Curve_Plot_Test = self.evaluate_data.Roc_Curve_Plot_Test
        self.roc_curve_path_test = os.path.join(self.test_plots_folder, f'Roc_Curve_Plot_Test_{self.algorithm}_{self.sd_run}_{self.architecture}.pdf')
        self.Roc_Curve_Plot_Test.savefig(self.roc_curve_path_test, format='pdf', bbox_inches='tight')
        plt.close(self.Roc_Curve_Plot_Test)
        logging.info(f"ROC curve plot (Test) saved to test_plots_folder.")
        
        # Save a plot of the coefficients with standard error of the optimal model (only for logistic regression)
        # Only for test data
        if self.algorithm == 'log_reg':
            self.evaluate_data.Make_Coef_Error()
            self.Coef_Error_Plot = self.evaluate_data.Coef_Error_Plot
            self.coef_error_path = os.path.join(self.test_plots_folder, f'Coef_Error_Plot_{self.algorithm}_{self.sd_run}_{self.architecture}.pdf')
            self.Coef_Error_Plot.savefig(self.coef_error_path, format='pdf', bbox_inches='tight')
            plt.close(self.Coef_Error_Plot)
            logging.info(f"Coef error plot (Test) saved to test_plots_folder.")

    
    def Train_Output_Attributes(self):
        """
        Save the textual (CSV) and plotted output attributes (PDF) of the train model/performance to save_folder 
        Create a PDF with evaluation plots and save to the train_merged_PDF folder to a train_merged_PDF file.

        Attributes:
            train_metric_data_text (dict): textual evaluation data of training performance. Also saved to a csv file for ease of access.
            feature_data_text (dict): textual selected feature importance data (training only) for specified model. Also saved to a csv file for ease of access.
            Plots saved to a temp file and then to train_merged_folder.
        """

        logging.info(f"Saving train output attributes for {self.architecture}, {self.algorithm}")
        
        elements = []

        # Add a title to the metric PDF
        title = f"Train Metrics Report [{self.architecture}]"
        elements.append(Paragraph(title, getSampleStyleSheet()['Title']))

        self.train_metric_data_text = {
            'Generator Model': self.architecture,
            'Algorithm': self.algorithm,
            'Data Type': 'Train',
            'SD Run': self.sd_run,
            'Pipeline Run': self.pipe_loop,
            'Train size before smote': str(self.orig_train_size),
            'Class balance before smote': str(self.class_weight_before_smote),
            'Train size after smote': str(self.post_smote_train_size),
            'Class balance after smote': str(self.class_weight_smote),
            'Best Params': self.best_params,
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
        col_widths = [None, 300]
        row_heights = [None] * len(train_metric_data)  # Set auto height for all rows
        train_metric_table = Table(train_metric_data, rowHeights=row_heights, colWidths=col_widths)

        # Indicate table design/layout
        train_metric_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.black),
              ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
              ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
              ('FONTNAME', (0, 0), (-1, 0), 'Courier-Bold'),
              ('FONTSIZE', (0, 0), (-1, 0), 12),
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

        pd.DataFrame([self.train_metric_data_text]).to_csv(os.path.join(self.train_csv_folder, f'Train_Metrics_{self.sd_run}_{self.algorithm}_{self.architecture}.csv'), index=False)

        # Create a temporary PDF file to hold the report content
        temp_report_pdf_file_train = os.path.join(self.train_save_folder, f'train_temp_report_{self.sd_run}.pdf')
        doc = SimpleDocTemplate(temp_report_pdf_file_train)
        doc.build(elements)

        # Merge the PLOT PDFs
        logging.info(f"Merging evaluation plots for {self.architecture}, {self.algorithm}")
        merger = PdfMerger()
        merger.append(temp_report_pdf_file_train)
        merger.append(self.roc_curve_path_train)
        merger.append(self.conf_mat_path_train)
        merger.append(self.corr_mat_path_train)

        merged_pdf_path_train = os.path.join(self.train_merged_folder, f'Train_Merged_Eval_Report_{self.sd_run}_{self.algorithm}_{self.architecture}.pdf')
        merger.write(merged_pdf_path_train)
        merger.close()
        logging.info(f"Merged Evaluations PDF saved to train_merged_folder.")

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

        logging.info(f"Saving test output attributes for {self.architecture}, {self.algorithm}")
        
        elements = []

        # Add a title to the metric PDF
        title = f"Test Metrics Report [{self.architecture}]"
        elements.append(Paragraph(title, getSampleStyleSheet()['Title']))

        self.test_metric_data_text = {
            'Generator Model': self.architecture,
            'Algorithm': self.algorithm,
            'Data Type': 'Test',
            'SD Run': self.sd_run,
            'Pipeline Run': self.pipe_loop,
            'Best Params': self.best_params,
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
            'Average Coef': self.coefs_mean if self.algorithm == 'log_reg' else 'N/A',
            'Stdv Coef': self.coefs_stdv if self.algorithm == 'log_reg' else 'N/A',
            'Standard Error': self.se if self.algorithm == 'log_reg' else 'N/A',
            'Odds Ratios': self.odds_ratios if self.algorithm == 'log_reg' else 'N/A',
            'CI Lower OR': self.ci_lower_or if self.algorithm == 'log_reg' else 'N/A',
            'CI Upper OR': self.ci_upper_or if self.algorithm == 'log_reg' else 'N/A',
            'P-Values': self.p_values if self.algorithm == 'log_reg' else 'N/A',
            'Effect Avg': self.effect_mean if self.algorithm == 'log_reg' else 'N/A',
            'Effect Stdv': self.effect_stdv if self.algorithm == 'log_reg' else 'N/A',
        }

        # Swap columns and rows in the table
        test_metric_data = [[key] + [value] for key, value in self.test_metric_data_text.items()]
        col_widths = [None, 300]
        row_heights = [None] * len(test_metric_data)  # Set standard height for all rows
        test_metric_table = Table(test_metric_data, rowHeights=row_heights, colWidths=col_widths)

        # Indicate table design/layout
        test_metric_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.black),
              ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
              ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
              ('FONTNAME', (0, 0), (-1, 0), 'Courier-Bold'),
              ('FONTSIZE', (0, 0), (-1, 0), 12),
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

        logging.info(f"Saving feature importance (test data) for {self.architecture}, {self.algorithm}")
        self.feature_data_opt_text = {
            'Feature': self.feature_importance_df['Feature'],
            'Importance': self.feature_importance_df['Importance'],
            'Normalized Importance': self.feature_importance_df['Normalized Importance'],
        }

        # Add a title to the feature importance PDF
        title = f"Feature Importances for optimized model [{self.architecture}]"
        elements.append(Paragraph(title, getSampleStyleSheet()['Title']))
        
        # Add data_text as a table to the PDF
        feature_data = [list(self.feature_data_opt_text.keys())] + list(map(list, zip(*self.feature_data_opt_text.values())))
        feature_table = Table(feature_data)
        feature_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.black),
                  ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                  ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                  ('FONTNAME', (0, 0), (-1, 0), 'Courier-Bold'),
                  ('FONTSIZE', (0, 0), (-1, 0), 12),
                  ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                  ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
                  ('LEFTPADDING', (0, 0), (-1, -1), 6),
                  ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                  ('TOPPADDING', (0, 0), (-1, -1), 6),
                  ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
                  ]))
        elements.append(feature_table)
        elements.append(PageBreak())

        # Save test metrics text to a csv file
        pd.DataFrame([self.test_metric_data_text]).to_csv(os.path.join(self.test_csv_folder, f'Test_Metrics_{self.sd_run}_{self.algorithm}_{self.architecture}.csv'), index=False)
        pd.DataFrame(self.feature_data_opt_text).to_csv(os.path.join(self.test_csv_folder, f'Opt_Feature_Importance_{self.sd_run}_{self.algorithm}_{self.architecture}.csv'), index=False)

        # Create a temporary PDF file to hold the report content
        temp_report_pdf_file_test = os.path.join(self.test_save_folder, f'test_temp_report_{self.sd_run}.pdf')
        doc = SimpleDocTemplate(temp_report_pdf_file_test)
        doc.build(elements)

        # Append the PLOT PDFs to the (textual) report PDF
        logging.info(f"Merging evaluation plots to text PDF for {self.architecture}, {self.algorithm}")
        merger = PdfMerger()
        merger.append(temp_report_pdf_file_test)
        merger.append(self.roc_curve_path_test)
        merger.append(self.conf_mat_path_test)
        merger.append(self.corr_mat_path_test)

        if self.algorithm == 'log_reg':
            merger.append(self.coef_error_path)

        merged_pdf_path_test = os.path.join(self.test_merged_folder, f'Test_Merged_Eval_Report_{self.sd_run}_{self.algorithm}_{self.architecture}.pdf')
        merger.write(merged_pdf_path_test)
        merger.close()
        logging.info(f"Merged (textual and plotted data) Evaluations PDF saved to test_merged_folder.")

        # Remove the temporary report PDF file
        os.remove(temp_report_pdf_file_test)

if __name__ == "__main__":

    # Create an argument parser for shell script (run.sh) to provide architecture, algorithm, train_df, and test_df
    parser = argparse.ArgumentParser(description='Provide algorithm and architecture.')
    parser.add_argument('--architecture', choices=['ctgan', 'fast_ml', 'gc', 'dp_cgan'], required=True, help='Architecture to use (ctgan, fast_ml, gc, dp_cgan)')
    parser.add_argument('--algorithm', choices=['log_reg', 'rf', 'mlp'], required=True, help='Algorithm to use (log_reg, rf, mlp)')
    parser.add_argument('--test_df', required=True, help='Path to test df to use')
    parser.add_argument('--train_df', required=False, help='Path to train df to use')
    args = parser.parse_args()

    save_to_file = FileSave(architecture=args.architecture, algorithm=args.algorithm, test_df=args.test_df)

