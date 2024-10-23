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


class FileSave:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def Create_Save_Folder(self, df_file_path):
        now = datetime.now()
        self.date_time = now.strftime("%d-%m_%H-%M")
        self.date = now.strftime("%d-%m")
        
        self.df_file_path = df_file_path
        self.path_parts = self.df_file_path.split(os.sep) # Split the file path by the directory separator
        print(self.path_parts)

        self.root_folder = self.path_parts[0]
        self.model_folder = self.path_parts[1]
        self.results_folder = '/results'
        self.save_folder = os.path.join(self.root_folder, self.model_folder, self.results_folder, f"{self.date}_Outputs") # root > ctgan > results
        print(self.save_folder)
    
    def Save_Evals(self):
        self.evaluations = EvaluateModel(self.df)
        self.Conf_Mat = self.evaluations.Make_Conf()
        conf_mat_path = os.path.join(self.save_folder, 'confusion_matrix_plot.pdf')
        self.Confusion_Matrix_Plot = self.evaluations.Confusion_Matrix_Plot

        plt.savefig(conf_mat_path, format='pdf')

        #Confusion_Matrix_Plot
        # self.file_path = os.path.join(self.save_folder, Conf_Mat, 'confusion_matrix.pdf')

        # # Generate the confusion matrix plot


        
        
        # self.Confusion_Matrix_Plot = self.evaluations.Confusion_Matrix_Plot
        
        # self.evaluations.Make_Conf(self.evaluations.y_val, self.evaluations.y_pred_ft)
        # self._Confusion_Matrix_Plot_ = self.evaluations._Confusion_Matrix_Plot_

        
        # self._Correlation_Plot_ = self.evaluations.Make_Conf(file_path = self.save_folder)
        # self._ROC_Plot_ = self.evaluations.Make_ROC()
        # self._OR_CI_Plot_ = self.saver.Make_OR_CI()



if __name__ == "__main__":
    df_file_path = input("Provide df file path: ")
    df = pd.read_csv(df_file_path, delimiter=',', encoding='latin1')

    save_to_file = FileSave(df)
    save_to_file.Create_Save_Folder(df_file_path)
    save_to_file.Save_Evals()

    # save_to_file.Get_Plots()

    # file_save.Save_PDF()
    # file_save.Merge_PDF()
        
    # print(f"PDF saved to: {pdf_file}")
    # print(f"PDF merged to: {os.path.join(target_folder, f'{model_name}_Report_{date_time}.
