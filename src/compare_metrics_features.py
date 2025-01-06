import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
import pytz

class Compare_CSVs:
    def __init__(self, architecture='ctgan', algorithm='log_reg'):

        self.architecture = architecture
        self.algorithm = algorithm

        # Find the folder with 'Outputs' in its name
        base_folder = f'/root/{self.architecture}/results/{self.algorithm}/pipeline_run/Outputs'
        
        self.test_csv_folder = os.path.join(base_folder, 'Test_1000_runs/Test_CSVs')
        self.train_csv_folder = os.path.join(base_folder, 'Train_1000_runs/Train_CSVs')
        
        # Initialize the combined metrics vars for train and test as DataFrames
        self.train_metric_indices = []
        self.test_metric_indices = []

        self.all_metric_names = pd.DataFrame()
        self.all_metric_values = pd.DataFrame()

        self.metric_indices = []
        self.metric_names = []
         
    def Metrics_Calculations(self, train_test):
        ''' 
        Combine metrics of the train (model performance) CSV files for data aggregation.
        '''

        csv_metrics_files = [os.path.join(getattr(self, f'{train_test}_csv_folder'), file) 
                 for file in os.listdir(getattr(self, f'{train_test}_csv_folder')) if 'Metrics_run' in file]
        
        metric_names = []
        metric_values = []
  
        for file in csv_metrics_files:
            
            f = open(file, newline='')
            file_reader = list(csv.reader(f, dialect='excel'))
            f.close()
            file_reader = pd.DataFrame(file_reader)

            metric_values.append(file_reader.iloc[1,:]) # Get the metric values from the second row of csv file
            
            #split file name into parts and get the run name
            run_name = '_'.join(file.split('_')[7:9])

            # FIXME: Make run name idx

            # self.file_reader.columns[] = [run_name]
            # print(self.file_reader)

        metric_names.append(file_reader.iloc[0,:]) # Get the metric names from the first row of csv file       
        
        # combine metric names and values into a single dataframe
        self.all_metric_names = pd.concat(metric_names, axis=0).reset_index(drop=True) 
        self.all_metric_values = pd.concat(metric_values, axis=1).reset_index(drop=True)
  
        #FIXME: perhaps take out cv scores and just keep mean and std; would be nice to have mean and stdv appended with mean and stdv cols instead of as rows.

        if train_test == 'train':
            setattr(self, f'{train_test}_metric_indices', [10, 11, 12, 13, 14, 15, 16]) # log loss: 15
            self.metric_names = ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1', 'Log Loss', 'AUC-ROC']
        elif train_test == 'test': # and self.algorithm in ['rf', 'mlp']:
            setattr(self, f'{train_test}_metric_indices', [6, 7, 8, 9, 10, 11, 12]) # log loss: 11
            self.metric_names = ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1', 'Log Loss', 'AUC-ROC']
        # elif train_test == 'test' and self.algorithm == 'log_reg':
        #     setattr(self, f'{train_test}_metric_indices', [6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 22]) # log loss: 11
        #     self.metric_names = ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1', 'Log Loss', 'AUC-ROC', 'Standard Error', 'Odds Ratios', 'CI Lower', 'CI Upper', 'P-Values']

        #     # convert indicies 18, 19, 20, 21, 22 to float
        #     for x in [18, 19, 20, 21, 22]:
        #         # self.all_metric_values.iloc[x] = self.all_metric_values.iloc[x].astype(str).str.split().apply(lambda y: [float(i) for i in y[0].split()])

       
        # self.metric_names = ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1', 'Log Loss', 'AUC-ROC']

        self.all_metric_values['Mean'] = np.nan
        self.all_metric_values['Standard Deviation'] = np.nan
        self.all_metric_values['Max'] = np.nan
        self.all_metric_values['Min'] = np.nan
        self.all_metric_values['Range'] = np.nan
        self.all_metric_values['IQR'] = np.nan
        self.all_metric_values['Median'] = np.nan
        self.all_metric_values['Mode'] = np.nan

        self.metric_calculation_file = pd.DataFrame()

        # Calculate the mean and standard deviation of the metrics
        for idx, name in zip(getattr(self, f'{train_test}_metric_indices'), self.metric_names):
            metric_values = self.all_metric_values.iloc[idx].astype(float)
            mean = round(metric_values.mean(), 2)
            stdv = round(metric_values.std(), 2)
            min_val = round(metric_values.min(), 2)
            max_val = round(metric_values.max(), 2)
            median_val = round(metric_values.median(), 2)
            range_val = round(max_val - min_val, 2)
            mode_val = round(metric_values.mode()[0], 2)  # mode() returns a Series, take the first element
            iqr_val = round(metric_values.quantile(0.75) - metric_values.quantile(0.25), 2)

            self.metric_calculation_file = pd.concat([self.metric_calculation_file, pd.DataFrame({'Metric': [name], 'Mean': [mean], 'Standard Deviation': [stdv], 'Min': [min_val], 'Max': [max_val], 'Range': [range_val], 'Median': [median_val], 'Mode': [mode_val], 'IQR': [iqr_val]})], ignore_index=True)
        
        if self.algorithm == 'log_reg' and train_test == 'test':
            mean_log = [round(self.all_metric_values.iloc[14].astype(float).mean(), 2),
                round(self.all_metric_values.iloc[16].astype(float).mean(), 2), 
                round(self.all_metric_values.iloc[23].astype(float).mean(), 2)]
            stdv_log = [round(self.all_metric_values.iloc[15].astype(float).std(), 2),
                round(self.all_metric_values.iloc[17].astype(float).std(), 2),
                round(self.all_metric_values.iloc[24].astype(float).std(), 2)]
            
            self.metric_calculation_file = pd.concat([self.metric_calculation_file, pd.DataFrame({'Metric': ['Cross Validation', 'Coefficient', 'Effect Size'], 'Mean': mean_log, 'Standard Deviation': stdv_log})], ignore_index=True)

        self.metric_calculation_file.to_csv(f'/root/{self.architecture}/results/{self.algorithm}/pipeline_run/Outputs/{train_test}_{self.algorithm}_Calculated_Metrics.csv', index=False)

        return self.metric_calculation_file
    

    def Visualize_Metrics(self):
        '''
        Visualize the metrics
        '''

        #get name of metric in self.metric_means_stdv
        
        name = [i[0] for i in self.metric_means_stdv]
        mean = [i[1] for i in self.metric_means_stdv]
        stdv = [i[2] for i in self.metric_means_stdv]
        
        fig, ax = plt.subplots()
        ax.errorbar(name, mean, yerr=stdv, fmt='o', color='blue', ecolor='red', capsize=5)
        ax.set_title('Train Metrics')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Mean')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        
    def Features_Metrics(self):

        '''
        Concatenate feature names and importance values across runs into a single data frame and csv file. 

        Attributes:
            csv_ft_files: a list of file paths to the feature importance csv files.
        
        Output:
            seen_features is a set of features that have been appeared more than once in the feature names across all files.

        '''

        # Recall that log reg has extra rows in the csv files

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
        self.importance_df.to_csv(os.path.join(self.test_csv_folder, 'Opt_Importances.csv'), index=False)

        # Tally loop
        self.feature_names_whole = pd.read_csv('/root/Features.csv', delimiter=',', encoding='latin1', names=['Feature Names'])
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

    # parser = argparse.ArgumentParser(description='Provide algorithm and architecture.')
    # parser.add_argument('--architecture', choices=['ctgan', 'fast_ml', 'gc'], required=True, help='Architecture to use (ctgan, fast_ml, gc)')
    # parser.add_argument('--algorithm', choices=['log_reg', 'rf', 'mlp'], required=True, help='Algorithm to use (log_reg, rf, mlp)')
    # args = parser.parse_args()

    # print(f'Architecture: {args.architecture}')
    # print(f'Algorithm: {args.algorithm}')

    file_save = Compare_CSVs()
    file_save.Metrics_Calculations('test')
    # file_save.Visualize_Metrics()
    # file_save.Features_Metrics()
    # file_save.Visualize_Fts()