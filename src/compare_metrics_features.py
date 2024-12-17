import csv
import os
import pandas as pd
import numpy as np
# from pipeline_saving import FileSave
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
import pytz

class Compare_CSVs:
    def __init__(self, architecture='ctgan', algorithm='log_reg'):

        amsterdam_tz = pytz.timezone('Europe/Amsterdam') # Set timezone to Amsterdam
        self.now = datetime.now(amsterdam_tz) # Get the current date and time in Amsterdam timezone
        self.date_time = self.now.strftime("%d%m-%H%M") # Format the date and time as DDMM-HHMM
        self.date = self.now.strftime("%d%m") 

        self.architecture = architecture
        self.algorithm = algorithm

        # FIXME: change for 1000 runs

        # Find the folder with 'Outputs' in its name
        base_folder = f'/root/{self.architecture}/results/{self.algorithm}/pipeline_run/Outputs' # change to: pipeline_run folder
        
        self.test_csv_folder = os.path.join(base_folder, 'Test_X_runs/Test_CSVs')
        self.train_csv_folder = os.path.join(base_folder, 'Train_X_runs/Train_CSVs')
        
        # Initialize the combined metrics vars for train and test as DataFrames
        self.train_combined_metrics = pd.DataFrame()
        self.test_combined_metrics = pd.DataFrame()
         
    def Train_Metrics(self):
        ''' 
        Combine metrics of the train (model performance) CSV files for data aggregation.
        '''

        self.train_csv_metrics_files = [os.path.join(self.train_csv_folder, file) 
                 for file in os.listdir(self.train_csv_folder) if 'Metrics_run' in file]
        
        train_metric_names = []
        train_metric_values = []
  
        for file in self.train_csv_metrics_files:
            
            f = open(file, newline='')
            self.file_reader = list(csv.reader(f, dialect='excel'))
            f.close()
            self.file_reader = pd.DataFrame(self.file_reader)

            train_metric_values.append(self.file_reader.iloc[1,:]) # Get the metric values from the second row of csv file

        train_metric_names.append(self.file_reader.iloc[0,:]) # Get the metric names from the first row of csv file       
        
        # combine metric names and values into a single dataframe
        train_con_met_names = pd.concat(train_metric_names, axis=0).reset_index(drop=True) 
        train_con_met_values = pd.concat(train_metric_values, axis=1).reset_index(drop=True) 

        # metrics_indices = [10, 11, 12, 13, 14, 15]
        # train_means = []
        # train_stds = []
        # for i in metrics_indices:
        #     train_means.append(round(train_con_met_values.iloc[i, :].astype(float).mean(), 2))
        #     train_stds.append(round(train_con_met_values.iloc[i, :].astype(float).std(), 2))
        
        # # Combine metric names and values into a single DataFrame
        self.train_combined_metrics = pd.concat([train_con_met_names, train_con_met_values], axis=1, ignore_index=True)
        self.train_combined_metrics.columns = ['Metrics/Attributes'] + [f'Run_{i+1}' for i in range(self.train_combined_metrics.shape[1] - 1)]

        # # Add train_means and train_stds to the combined metrics DataFrame as new columns
        # self.train_combined_metrics = pd.concat([self.train_combined_metrics, pd.DataFrame({
        #     'Mean': train_means,
        #     'Standard Deviation': train_stds
        # })], axis=1)

        # print(self.train_combined_metrics)
        # Save the combined metrics to a CSV file
        self.train_combined_metrics.to_csv(os.path.join(self.train_csv_folder, 'Train_Metrics.csv'), index=False)



    def Test_Metrics(self):
        ''' 
        Combine metrics of the test (model performance) CSV files for data aggregation.
        '''

        self.test_csv_metrics_files = [os.path.join(self.test_csv_folder, file) 
                 for file in os.listdir(self.test_csv_folder) if 'Metrics_run' in file]
        
        test_metric_names = []
        test_metric_values = []
  
        for file in self.test_csv_metrics_files:
            
            f = open(file, newline='')
            self.file_reader = list(csv.reader(f, dialect='excel'))
            f.close()
            self.file_reader = pd.DataFrame(self.file_reader)

            test_metric_values.append(self.file_reader.iloc[1,:]) # Get the metric values from the second row of csv file

        test_metric_names.append(self.file_reader.iloc[0,:]) # Get the metric names from the first row of csv file       
        
        # combine metric names and values into a single dataframe
        test_con_met_names = pd.concat(test_metric_names, axis=0).reset_index(drop=True) 
        test_con_met_values = pd.concat(test_metric_values, axis=1).reset_index(drop=True) 

        # 0: gen model; 5: acc, 6: bal acc, 7: precision, 8: recall, 9: f1, 10: log loss, 11: auc, 13: cv mean, 14: cvstdv, 16: coef avg, 17: coef stdv, 19: effects stdv
        metrics_indices = [5, 6, 7, 8 , 9, 10, 11, 13, 14, 16, 17, 19, 20]
        test_means = []
        test_stdv = []
      
        for i in metrics_indices:
            test_means.append(round(test_con_met_values.iloc[i, :].astype(float).mean(), 2))
            test_stdv.append(round(test_con_met_values.iloc[i, :].astype(float).std(), 2))
        
        # append test_means in line with proper indices as new column
        test_avg_stdv = pd.concat([test_con_met_values, pd.DataFrame({
            'Mean': test_means,
            'Standard Deviation': test_stdv
        })], axis=0)

        # Combine metric names and values into a single DataFrame
        self.test_combined_metrics = pd.concat([test_con_met_names, test_avg_stdv], axis=1, ignore_index=True)
        self.test_combined_metrics.columns = ['Metrics/Attributes'] + [f'Run_{i+1}' for i in range(self.test_combined_metrics.shape[1] - 1)]

        print(self.test_combined_metrics)

        # Save the combined metrics to a CSV file
        self.test_combined_metrics.to_csv(os.path.join(self.test_csv_folder, 'Test_Metrics.csv'), index=False)




        # TODO: add effect sizes (mean), p value, coef mean and update in evaluate model script
        # So for next meeting: how much detail do we want for the coefs and effect sizes? 
        # bc I think I should keep all values

        
    def Features_Metrics(self):

        '''
        Concatenate feature names and importance values across runs into a single data frame and csv file. 

        Attributes:
            csv_ft_files: a list of file paths to the feature importance csv files.
        
        Output:
            seen_features is a set of features that have been appeared more than once in the feature names across all files.

        '''

        # FIXME: Perhaps do this fx for both train & test so that not 2 separate cause essentially the same data minus 3 rows
        # Also recall that log reg has extra rows in the csv files
        # TODO: How to get coef representation from mlp and rf?

        self.csv_ft_files = [os.path.join(self.test_csv_folder, file) 
            for file in os.listdir(self.test_csv_folder) if 'Feature_Importance' in file] 
        
        self.feature_names_indiv = [] 
        self.feature_values = []
        self.file_name = []
        self.combined_ft_vals = []
        
        for i, file in enumerate(self.csv_ft_files):
            self.run = '_'.join(file.split('/')[-1].split('_')[4:6])

            f = open(file, newline='')
            self.file_reader = list(csv.reader(f, dialect='excel'))
            f.close()
            self.file_reader = pd.DataFrame(self.file_reader)
            
            # Add a blank row before each file's feature names and values
            self.feature_names_indiv.append(pd.Series([None]))
            self.feature_values.append(pd.Series([None])) 

            self.feature_names_indiv.append(self.file_reader.iloc[1:, 0])
            self.feature_values.append(self.file_reader.iloc[1:, 1])

            self.file_name.append(f'{self.run}_{self.architecture}_{self.algorithm}')
            
            # Combine feature names mentioned and individual values into a single DataFrame
            if 'Feature' in file:
                if len(self.feature_names_indiv) > i*2+1 and len(self.feature_values) > i*2+1:
                    feature_names = self.feature_names_indiv[i*2+1].reset_index(drop=True)
                    feature_values = self.feature_values[i*2+1].reset_index(drop=True)
                    self.combined_ft_vals.append(pd.DataFrame({
                        'File Name': [self.file_name[i]] * len(feature_names), # lists are structured such that every second element (starting from index 1) corresponds to the features and values of the files.
                        'Feature Names': feature_names,
                        'Importance': feature_values.astype(float),
                        # 'Mean': round(feature_values.astype(float).mean(), 2),
                        # 'Standard Deviation': round(feature_values.astype(float).std(), 2)
                    }))
        
        # Concatenate all dataframes in the list
        self.all_ft_vals_df = pd.concat(self.combined_ft_vals, ignore_index=True).drop_duplicates()

        # Pivot the DataFrame to have file names as columns and feature names as rows
        self.importance_df = self.all_ft_vals_df.pivot_table(index='Feature Names', columns='File Name', values='Importance', aggfunc='first')
        
        # Reset the index to make 'Feature Names' a column again
        self.importance_df.reset_index(inplace=True)
 
        # From importance scores across df for each feature get the mean and standard deviation
        self.importance_df['Mean'] = round(self.importance_df.iloc[:, 1:].mean(axis=1), 2)
        self.importance_df['Standard Deviation'] = round(self.importance_df.iloc[:, 1:].std(axis=1), 2)
        cols = self.importance_df.columns.tolist()
        
        # Reorder the columns
        cols = cols[:1] + cols[-2:] + cols[1:-2]
        self.importance_df = self.importance_df[cols]
        # print('IMPT DF \n')
        print(self.importance_df)
        self.importance_df.to_csv(os.path.join(self.test_csv_folder, 'Opt_Importances.csv'), index=False)

        # Tally loop
        self.feature_names_whole = pd.read_csv('/root/X_cols.csv', delimiter=',', encoding='latin1', names=['Feature Names'])
        self.feature_names_whole['Tally'] = 0
        
        for feature_series in self.feature_names_indiv:
            for feature in feature_series.dropna().values:
                if feature in self.feature_names_whole['Feature Names'].values:
                    self.feature_names_whole.loc[self.feature_names_whole['Feature Names'] == feature, 'Tally'] += 1

        self.feature_names_whole.to_csv(os.path.join(self.test_csv_folder, 'Opt Feature Tally.csv'), index=False)

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

        return self.importance_df

    
    # def Visualize_Fts(self):
        
    #     melted_df = self.feature_metrics.melt(id_vars=['Feature Name'], var_name='Metric', value_name='Value')

    #     # Plotting the distribution of importance scores across models
    #     plt.figure(figsize=(12, 8))
    #     sns.boxplot(data=melted_df, x='Feature Name', y='Value', hue='Feature Name')
    #     plt.title('Distribution of Importance Scores Across Models')
    #     plt.xlabel('Features')
    #     plt.ylabel('Importance Score')
    #     plt.xticks(rotation=90)
    #     plt.tight_layout()
    #     plt.show()

    # #TODO: get standard deviation across runs for all metrics (We have it for features but not as crucial)
    # def Compare_Models(self):
    #     # get all metrics from the combined_metrics file
    #     self.metrics = self.combined_metrics.iloc[:, 0]
    #     print(self.metrics)

    


if __name__ == '__main__': 

    # parser = argparse.ArgumentParser(description='Provide algorithm and architecture.')
    # parser.add_argument('--architecture', choices=['ctgan', 'fast_ml', 'gc'], required=True, help='Architecture to use (ctgan, fast_ml, gc)')
    # parser.add_argument('--algorithm', choices=['log_reg', 'rf', 'mlp'], required=True, help='Algorithm to use (log_reg, rf, mlp)')
    # args = parser.parse_args()

    # print(f'Architecture: {args.architecture}')
    # print(f'Algorithm: {args.algorithm}')

    file_save = Compare_CSVs()
    file_save.Train_Metrics()
    file_save.Test_Metrics()
    # file_save.Features_Metrics()

    # file_save.Compare_Models()
    # file_save.Visualize_Fts()