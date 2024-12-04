import csv
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.text as mtext
import seaborn as sns

from datetime import datetime
import pytz

from data_processing import DataProcessing

class Compare_CSVs:
    def __init__(self):

        amsterdam_tz = pytz.timezone('Europe/Amsterdam') # Set timezone to Amsterdam
        self.now = datetime.now(amsterdam_tz) # Get the current date and time in Amsterdam timezone
        self.date_time = self.now.strftime("%d%m-%H%M") # Format the date and time as DDMM-HHMM
        self.date = self.now.strftime("%d%m") 

        self.model = input("Choose model: ctgan, fast_ml, gc: ")
        self.algorithm = input("Algorithm (log_reg, rf, mlp): ")

        self.csv_folder = f'/root/{self.model}/results/{self.algorithm}/{self.date}_Outputs/X_runs_CSV' # change for 1000 runs

        # self.processed = DataProcessing()
        # self.X_ft = self.processed.X_ft


         
    def Compare_Metrics(self):
        
        ''' 

        Function to concatenate metrics of all files in csv_metrics_files and keep 1st col as metric name.

        '''

        self.csv_metrics_files = [os.path.join(self.csv_folder, file) 
                 for file in os.listdir(self.csv_folder) if 'Metrics' in file]
        
        self.metric_names = []
        self.metric_values = []
  
        for file in self.csv_metrics_files:
            
            f = open(file, newline='')
            self.file_reader = list(csv.reader(f, dialect='excel'))
            self.file_reader = pd.DataFrame(self.file_reader)

            self.run = '_'.join(file.split('/')[-1].split('_')[0:2])

            self.metric_values.append(self.file_reader.iloc[1,:]) 
        
        self.metric_names.append(self.file_reader.iloc[0,:])

        self.con_met_names = pd.concat(self.metric_names, axis=0).reset_index(drop=True) 
        self.con_met_values = pd.concat(self.metric_values, axis=1).reset_index(drop=True) 
            
        # Combine metric names and values into a single DataFrame
        self.combined_metrics = pd.concat([self.con_met_names, self.con_met_values], axis=1, ignore_index=True)
        self.combined_metrics.columns = ['Metrics/Attributes'] + [f'{self.run}_{self.algorithm}' for i in range(len(self.csv_metrics_files))]
        
        # Save the combined metrics to a CSV file
        self.combined_metrics.to_csv(os.path.join(self.csv_folder, 'combined_metrics.csv'), index=False)
        

    def Features_Concat(self):

        '''
        Function to concatenate feature names and importance values across runs into a single data frame and csv file. 

        Input:
        - csv_ft_files: a list of file paths to the feature importance csv files.
        
        Output:
        - seen_features is a set of features that have been appeared more than once in the feature names across all files.

        '''

        self.csv_ft_files = [os.path.join(self.csv_folder, file) 
            for file in os.listdir(self.csv_folder) if 'Feature_Importance' in file] 
        
        self.feature_names_indiv = [] 
        self.feature_values = []
        self.file_name = []
        self.combined_ft_vals_list = []
        
        for file in self.csv_ft_files:
            self.run = '_'.join(file.split('/')[-1].split('_')[4:6])
            self.file_name.append(f'{self.run}_{self.model}_{self.algorithm}')

            f = open(file, newline='')
            self.file_reader = list(csv.reader(f, dialect='excel'))
            f.close()
            self.file_reader = pd.DataFrame(self.file_reader)
            
            # Add a blank row before each file's feature names and values
            self.feature_names_indiv.append(pd.Series([None]))
            self.feature_values.append(pd.Series([None])) 

            self.feature_names_indiv.append(self.file_reader.iloc[1:, 0])
            self.feature_values.append(self.file_reader.iloc[1:, 1])
            
        self.con_ft_names = pd.concat(self.feature_names_indiv, axis=0).reset_index(drop=False) #kept original index for reference
        self.con_ft_values = pd.concat(self.feature_values, axis=0).reset_index(drop=False)



        # Combine feature names and values into a single DataFrame
        for i, file_name in enumerate(self.file_name):
            if len(self.feature_names_indiv) > i*2+1 and len(self.feature_values) > i*2+1:
                feature_names = self.feature_names_indiv[i*2+1].reset_index(drop=True)
                feature_values = self.feature_values[i*2+1].reset_index(drop=True)
                self.combined_ft_vals_list.append(pd.DataFrame({
                    'File Name': [file_name] * len(feature_names), # lists are structured such that every second element (starting from index 1) corresponds to the features and values of the files.
                    'Feature Names': feature_names,
                    'Importance': feature_values
                }))

        # Concatenate all dataframes in the list
        self.all_ft_vals_df = pd.concat(self.combined_ft_vals_list, ignore_index=True)

        # Pivot the DataFrame to have file names as columns and feature names as rows
        self.importance_df = self.all_ft_vals_df.pivot(index='Feature Names', columns='File Name', values='Importance')

        # Reset the index to make 'Feature Names' a column again
        self.importance_df.reset_index(inplace=True)

        # print('\n Importance DF \n', self.importance_df)
        self.importance_df.to_csv(os.path.join(self.csv_folder, 'importance_df.csv'), index=False)

        return self.importance_df



    def Tally_Features(self):

        # Tally loop
        self.feature_names_whole = pd.read_csv('/root/X_cols.csv', delimiter=',', encoding='latin1', names=['Feature Names'])
        self.feature_names_whole['Tally'] = 0
        
        for feature_series in self.feature_names_indiv:
            for feature in feature_series.dropna().values:
                if feature in self.feature_names_whole['Feature Names'].values:
                    self.feature_names_whole.loc[self.feature_names_whole['Feature Names'] == feature, 'Tally'] += 1
           
        self.feature_names_whole.to_csv(os.path.join(self.csv_folder, 'tally.csv'), index=False)

        print('\n Ft Whole Tally \n ', self.feature_names_whole)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Feature Names', y='Tally', data=self.feature_names_whole.sort_values(by='Tally', ascending=False))
        plt.title('Feature Tally Across Models')
        plt.xlabel('Features')
        plt.ylabel('Tally')
        plt.xticks(rotation=90)
        plt.yticks(range(0, self.feature_names_whole['Tally'].max() + 1, 10))
        plt.tight_layout()
        plt.show()

        return self.feature_names_whole
        

    def Compare_Features(self):
        self.feature_metrics = pd.DataFrame()

        for _, row in self.importance_df.iterrows():
            self.ft_name = row['Feature Names']
            self.impt_vals = row.drop('Feature Names').dropna().astype(float)

            self.feature_sd = round(self.impt_vals.std(), 3)
            self.feature_min = self.impt_vals.min()
            self.feature_max = self.impt_vals.max()
            self.feature_range = round(self.feature_max - self.feature_min, 3)
            self.feature_mean = self.impt_vals.mean()
                 
            print(f'\n Feature: {self.ft_name}, \n SD: {self.feature_sd}, \n Min: {self.feature_min}, \n Max: {self.feature_max} \n Range: {self.feature_range}, \n Mean: {self.feature_mean}')
            
            # make new df with these values
            self.feature_metrics_row = pd.DataFrame({
                'Feature Name': [self.ft_name],
                'SD': [self.feature_sd],
                'Min': [self.feature_min],
                'Max': [self.feature_max],
                'Range': [self.feature_range],
                'Mean': [self.feature_mean],
            })
            self.feature_metrics_row.dropna(inplace=True)
            self.feature_metrics = pd.concat([self.feature_metrics, self.feature_metrics_row], ignore_index=True)

        self.feature_metrics.to_csv(os.path.join(self.csv_folder, 'feature_metrics.csv'), index=False)

        return self.feature_metrics
    
    def Visualize_Fts(self):
        
        melted_df = self.feature_metrics.melt(id_vars=['Feature Name'], var_name='Metric', value_name='Value')

        # Plotting the distribution of importance scores across models
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=melted_df, x='Feature Name', y='Value', hue='Feature Name')
        plt.title('Distribution of Importance Scores Across Models')
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__': 
    file_save = Compare_CSVs()
    file_save.Compare_Metrics()
    file_save.Features_Concat()
    file_save.Tally_Features()
    file_save.Compare_Features()
    file_save.Visualize_Fts()