# from .data_pre_processing import DataPreProcessing
# from .data_processing import DataProcessing
# from .evaluation_metrics import EvaluateModel
# from .file_handling import FileSave

# __all__ = ['DataPreProcessing', 'DataProcessing', 'EvaluateModel', 'FileSave']


# perhaps add a function to run the whole pipeline in one go.
# request df
# src/__init__.py

# from file_handling import FileSave
# import pandas as pd

# df_file_path = input("Provide df file path: ")
# df = pd.read_csv(df_file_path, delimiter=',', encoding='latin1')


# save_to_file = FileSave(df, algorithm= input("Choose algorithm: log_reg, rf, mlp: "))
# save_to_file.Create_Save_Folder(df_file_path)
# save_to_file.Save_Eval_Plots()
# save_to_file.Output_Attributes()

# requirements file