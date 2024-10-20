import logging
import os 

from data_pre_processing import DataPreProcessing
from data_processing import DataProcessing
from evaluation_metrics import EvaluateModel

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, PageBreak
from reportlab.lib import colors
import PyPDF2 
from PyPDF2 import PdfMerger


class FileSave(self):
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        self.saver = EvaluateModel(self.df)
        self.saver.Make_Conf()
        self.saver.Make_Corr()
        self.saver.Make_ROC()
        self.saver.Make_PR()

        # self.pdf_file = os.path.join(target_folder, f'{model_name}_Report_{date_time}.pdf')
        # self.doc = SimpleDocTemplate(pdf_file, pagesize=A4)
        # self.elements = []

        # model_name = 


    def Save_PDF(self):
        # Add a title to the PDF
        title = f'{model_name} Report'
        elements.append(canvas.Canvas(pdf_file).drawString(40, 800, title))

        # Add a table to the PDF
        data = [list(self.df.columns)] + self.df.values.tolist()
        t = Table(data)
        t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                              ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                              ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                              ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                              ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                              ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                              ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
        elements.append(t)

        # Add a page break
        elements.append(PageBreak())

        # Build the PDF
        doc.build(elements)

        return pdf_file

    def Merge_PDF(self):
        # Merge the PDF files
        merger = PdfMerger()
        merger.append(pdf_file)
        merger.append(confusion_matrix_plot)
        merger.append(correlation_matrix_plot)
        merger.append(ft_impt_file_path)
        merger.write(os.path.join(target_folder, f'{model_name}_Report_{date_time}.pdf'))
        merger.close()

# # Get current date and time
# now = datetime.now()
# date_time = now.strftime("%d-%m_%H-%M")
# date = now.strftime("%d-%m")

# base_path = "../"
# target_folder_name = input('PDF (Destination) folder name: ')
# target_folder = os.path.join(base_path, target_folder_name, f"{date}_Outputs")
# os.makedirs(target_folder, exist_ok=True) 
# model_name = input("Please enter the model name: ") 
# # model_file_path = os.path.join(base_path, target_folder_name, f"{model_name}.csv")
# model_file_path = input("Please enter the model file path: ")

# path_parts = model_file_path.split(os.sep) # Split the file path by the directory separator
# file_name = path_parts[-1] # Extract the second to last part of the split path

if __name__ == "__main__":
    df_file_path = input("Provide df file path: ")
    df = pd.read_csv(df_file_path, delimiter=',', encoding='latin1')

    file_save = FileSave(df)
    file_save.Save_PDF()
    file_save.Merge_PDF()
        
    # print(f"PDF saved to: {pdf_file}")
    # print(f"PDF merged to: {os.path.join(target_folder, f'{model_name}_Report_{date_time}.