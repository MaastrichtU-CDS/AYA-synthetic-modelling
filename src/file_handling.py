import logging
import os 

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, PageBreak
from reportlab.lib import colors
import PyPDF2 
from PyPDF2 import PdfMerger

# Get current date and time
now = datetime.now()
date_time = now.strftime("%d-%m_%H-%M")
date = now.strftime("%d-%m")

base_path = "../"
target_folder_name = input('PDF (Destination) folder name: ')
target_folder = os.path.join(base_path, target_folder_name, f"{date}_Outputs")
os.makedirs(target_folder, exist_ok=True) 
model_name = input("Please enter the model name: ") 
# model_file_path = os.path.join(base_path, target_folder_name, f"{model_name}.csv")
model_file_path = input("Please enter the model file path: ")

df_og = pd.read_csv(model_file_path, delimiter=',', encoding='latin1')
df = pd.DataFrame(df_og.copy())
df.drop(columns='leeft', inplace=True)

path_parts = model_file_path.split(os.sep) # Split the file path by the directory separator
file_name = path_parts[-1] # Extract the second to last part of the split path