import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Compare_CSVs:
    def __init__(self, architecture, algorithm):
        '''
        This class compares the performance metrics and feature values (importance score, tally, log_reg values) 
            across multiple runs (hence, averaging) of a pipeline.
        Initializes the necessary folders and files for (compiled) scores across architectures and generators.

        Below is a summary of the documents produced.
        - Feature_Importance_Values
        - Feature_Importance_Tally (csv & plot)
        - Feature_Importance_Metrics_with_Tally
        - Train_Avg_Calculated_Metrics
        - Test_Avg_Calculated_Metrics
        - Logistic_Feature_Vals_Metrics_With_Tally

        Parameters:
        - architecture (str): The architecture used in the pipeline.
        - algorithm (str): The algorithm used in the pipeline.
        '''

        self.architecture = architecture
        self.algorithm = algorithm

        # Set the base folder path to fetch data for the results
        self.base_folder = f'/root/{self.architecture}/results/{self.algorithm}/pipeline_run/Outputs'
        
        self.test_csv_folder = os.path.join(self.base_folder, 'Test_1000_runs/Test_CSVs')
        self.train_csv_folder = os.path.join(self.base_folder, 'Train_1000_runs/Train_CSVs')

        self.csv_ft_files = [os.path.join(self.test_csv_folder, file) 
            for file in os.listdir(self.test_csv_folder) if 'Feature_Importance' in file] 

        # Set the output folder path to save the compiled results
        self.compiled_folder = os.path.join(self.base_folder, 'Compiled_Results')

        # Create the output folder if it does not exist
        if not os.path.exists(self.compiled_folder):
            os.makedirs(self.compiled_folder)
        
        # Initialize the combined metrics vars for train and test as DataFrames
        self.all_test_metric_values = pd.DataFrame()
        self.all_train_metric_values = pd.DataFrame()
        self.train_metric_calculations = pd.DataFrame()
        self.test_metric_calculations = pd.DataFrame()
        self.metric_names = ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1', 'Log Loss', 'AUC-ROC']

        # Initialize for feature importance scores (all models)
        self.combined_ft_vals = pd.DataFrame()
        self.feature_importance_tally = pd.DataFrame(columns=['Feature Names', 'Tally'])
        self.feature_importance_calculations = pd.DataFrame()
        
        # Initalize for LR scores
        self.LR_feature_val_calculations = pd.DataFrame()
        self.feature_logistic_values = ()
        self.log_ft_all = pd.DataFrame()

    def Feature_Importance_Values_Tally(self):

        '''
        Concatenate feature names and importance values across runs into a single data frame and csv file. 
        Create csv file with tally of selected features.
        '''

        print('Doing Feature Importance Metrics')
        
        feature_values = []
        file_name = []
        feature_names_indiv = []

        # Loop to get the feature names and importance scores from the csv files
        for i, file in enumerate(self.csv_ft_files):
            run = '_'.join(file.split('/')[-1].split('_')[4:6])

            f = open(file, newline='')
            file_reader = list(csv.reader(f, dialect='excel'))
            f.close()
            file_reader = pd.DataFrame(file_reader)
            
            feature_names_indiv.append(file_reader.iloc[1:, 0])
            feature_values.append(file_reader.iloc[1:, 1].astype(float))

            file_name.append(f'{run}_{self.architecture}_{self.algorithm}')
            
        # Combine feature names mentioned and individual values into a single DataFrame
        feature_names = pd.concat(feature_names_indiv)
        feature_values = pd.concat(feature_values)

        ft_df = pd.DataFrame({
            'File Name': [file_name[i]] * len(feature_names), # lists are structured such that every second element (starting from index 1) corresponds to the features and values of the files.
            'Feature Names': feature_names,
            'Importance': feature_values.astype(float),
        })
        self.combined_ft_vals = pd.concat([self.combined_ft_vals, ft_df], ignore_index=True)    

        self.combined_ft_vals.to_csv(f'{self.compiled_folder}/{self.architecture}_{self.algorithm}_Feature_Importance_All.csv', index=False)

        '''
        Tally the number of times each feature appears in the feature importance values across runs.
        Sort the feature importance tally by the number of times each feature appears.
        Save the Tally DataFrame to a csv file.
        Plot the feature importance tally.
        '''

        print('Doing Feature Tally')

        # Tally the number of times each feature appears in the feature importance values
        feature_counts = feature_names.value_counts()
        self.feature_importance_tally = pd.DataFrame({
            'Feature Names': feature_counts.index,
            'Tally': feature_counts.values
        }).sort_values(by='Tally', ascending=False)

        print(self.feature_importance_tally)

        # Save the Tally DataFrame to a csv file
        self.feature_importance_tally.to_csv(f'{self.compiled_folder}/{self.architecture}_{self.algorithm}_Feature_Importance_Tally.csv', index=False)
    
        # Plot the feature importance tally
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Feature Names', y='Tally', data=self.feature_importance_tally)
        plt.title(f'{self.algorithm} {self.architecture} Feature Tally')
        plt.xlabel('Features')
        plt.ylabel('Tally')
        plt.xticks(rotation=90)
        plt.yticks(range(0, int(self.feature_importance_tally['Tally'].max()) + 1, 100))
        plt.tight_layout()
        plt.savefig(f'{self.compiled_folder}/{self.architecture}_{self.algorithm}_Feature_Importance_Tally_Histo.pdf')
        plt.show()

        print('Feature Importance Metrics & Tally Done')
        return self.feature_importance_tally, self.combined_ft_vals

    def Feature_Importance_Calculations(self):
        '''
        Calculate the mean, standard deviation, min, max, median, mode, and IQR of the feature importance values.
        Save the calculated metrics to a csv file.
        '''

        print('Doing Feature Importance Calculations')
    
        # Calculate the mean, standard deviation, min, max, median, mode, and IQR of the feature importance values
        for feature in self.feature_importance_tally['Feature Names']:
            # get importance score for each feature across runs and calculate mean, std dev, min, max, median, mode, iqr
            feature_data = self.combined_ft_vals[self.combined_ft_vals['Feature Names'] == feature]['Importance']
            stats = {
                'Feature Names': [feature],
                'Importance Mean': [round(feature_data.mean(), 2)],
                'Importance Std Dev': [round(feature_data.std(), 2)],
                'Importance Min': [round(feature_data.min(), 2)],
                'Importance Max': [round(feature_data.max(), 2)],
                'Importance Median': [round(feature_data.median(), 2)],
                'Importance Mode': [round(feature_data.mode().iloc[0], 2) if not feature_data.mode().empty else np.nan],
                'Importance IQR': [round(feature_data.quantile(0.75) - feature_data.quantile(0.25), 2)]
            }
            self.feature_importance_calculations = pd.concat([self.feature_importance_calculations, pd.DataFrame(stats)], ignore_index=True)
        
        self.feature_importance_calculations.dropna(inplace=True)

        # add tally to feature_importaance_calculations where feature names matches the feature names in feature_importance_tally
        self.feature_importance_calculations = self.feature_importance_calculations.merge(self.feature_importance_tally, on='Feature Names', how='left')
        self.feature_importance_calculations = self.feature_importance_calculations[['Feature Names', 'Tally', 'Importance Mean', 'Importance Std Dev', 'Importance Min', 'Importance Max', 'Importance Median', 'Importance Mode', 'Importance IQR']]

        self.feature_importance_calculations.to_csv(f'{self.compiled_folder}/{self.architecture}_{self.algorithm}_Feature_Importance_Metrics_with_Tally.csv', index=False)

        print('Feature Importance Metrics Done')
        return self.feature_importance_calculations
         
    def Train_Metrics_Calculations(self):
        ''' 
        Combine performance metrics of the train CSV files for data aggregation.
        Calculate the mean, standard deviation, min, max, median, mode, and IQR of the metrics.
        Save the calculated metrics to a csv file.
        '''

        print('Doing Train Metrics Calculations')

        train_csv_metrics_files = [os.path.join(getattr(self, 'train_csv_folder'), file) 
                 for file in os.listdir(getattr(self, 'train_csv_folder')) if 'Metrics_run' in file]
        
        train_metric_values = []
       
        # Loop to get the metric names and values from the csv files
        for file in train_csv_metrics_files:

            f = open(file, newline='')
            file_reader = list(csv.reader(f, dialect='excel'))
            f.close()
            file_reader = pd.DataFrame(file_reader)

            train_metric_values.append(file_reader.iloc[1,:]) # Get the metric values from the second row of csv file
        
        # combine metric names and values into a single dataframe
        self.all_train_metric_values = pd.concat(train_metric_values, axis=1).reset_index(drop=True)
        # Initialize columns for the calculated metrics
        self.all_train_metric_values[['Mean', 'Standard Deviation', 'Max', 'Min', 'Range', 'IQR', 'Median', 'Mode']] = np.nan
        train_metric_indices = [10, 11, 12, 13, 14, 15, 16]

        # Calculate the mean and standard deviation of the metrics and append to the dataframe
        train_metric_calculations_list = []
        for idx, name in zip(train_metric_indices, self.metric_names):
            metric_values = self.all_train_metric_values.iloc[idx].astype(float)
            mean = round(metric_values.mean(), 2)
            stdv = round(metric_values.std(), 2)
            min_val = round(metric_values.min(), 2)
            max_val = round(metric_values.max(), 2)
            median_val = round(metric_values.median(), 2)
            range_val = round(max_val - min_val, 2)
            mode_val = round(metric_values.mode()[0], 2)  # mode() returns a Series, take the first element
            iqr_val = round(metric_values.quantile(0.75) - metric_values.quantile(0.25), 2)

            train_metric_calculations_list.append(pd.DataFrame({'Metric': [name], 'Mean': [mean], 'Standard Deviation': [stdv], 'Min': [min_val], 'Max': [max_val], 'Range': [range_val], 'Median': [median_val], 'Mode': [mode_val], 'IQR': [iqr_val]}))
        
        self.train_metric_calculations = pd.concat(train_metric_calculations_list, ignore_index=True)
        self.train_metric_calculations.to_csv(f'/{self.compiled_folder}/{self.architecture}_{self.algorithm}_Train_Avg_Calculated_Metrics.csv', index=False)

        print('Train Metrics Calculations Done')
        return self.train_metric_calculations, self.all_train_metric_values
    
    def Test_Metrics_Calculations(self):
        '''
        Combine performance metrics of the test CSV files for data aggregation.
        Calculate the mean, standard deviation, min, max, median, mode, and IQR of the metrics.
        Save the calculated metrics to a csv file.
        '''

        print('Doing Test Metrics Calculations')

        test_metric_values = []

        test_csv_metrics_files = [os.path.join(getattr(self, 'test_csv_folder'), file) 
                 for file in os.listdir(getattr(self, 'test_csv_folder')) if 'Metrics_run' in file]
        
        # Loop to get the metric names and values from the csv files
        for file in test_csv_metrics_files:
            
            f = open(file, newline='')
            file_reader = list(csv.reader(f, dialect='excel'))
            f.close()
            file_reader = pd.DataFrame(file_reader)

            test_metric_values.append(file_reader.iloc[1,:]) # Get the metric values from the second row of csv file

        # combine metric names and values into a single dataframe
        self.all_test_metric_values = pd.concat(test_metric_values, axis=1).reset_index(drop=True)
        # Initialize columns for the calculated metrics
        self.all_test_metric_values[['Mean', 'Standard Deviation', 'Max', 'Min', 'Range', 'IQR', 'Median', 'Mode']] = np.nan
        test_metric_indices = [6, 7, 8, 9, 10, 11, 12]

        # Calculate the mean and standard deviation of the metrics
        test_metric_calculations_list = []
        for idx, name in zip(test_metric_indices, self.metric_names):
            metric_values = self.all_test_metric_values.iloc[idx].astype(float)
            mean = round(metric_values.mean(), 2)
            stdv = round(metric_values.std(), 2)
            min_val = round(metric_values.min(), 2)
            max_val = round(metric_values.max(), 2)
            median_val = round(metric_values.median(), 2)
            range_val = round(max_val - min_val, 2)
            mode_val = round(metric_values.mode()[0], 2)
            iqr_val = round(metric_values.quantile(0.75) - metric_values.quantile(0.25), 2)

            test_metric_calculations_list.append(pd.DataFrame({'Metric': [name], 'Mean': [mean], 'Standard Deviation': [stdv], 'Min': [min_val], 'Max': [max_val], 'Range': [range_val], 'Median': [median_val], 'Mode': [mode_val], 'IQR': [iqr_val]}))

        self.test_metric_calculations = pd.concat(test_metric_calculations_list, ignore_index=True)
        self.test_metric_calculations.to_csv(f'/{self.compiled_folder}/{self.architecture}_{self.algorithm}_Test_Avg_Calculated_Metrics.csv', index=False)

        print('Test Metrics Calculations Done')
        return self.test_metric_calculations, self.all_test_metric_values
    
    def Log_Reg_Scores(self):
        """
        Processes CSV files containing logistic regression metrics, calculates statistical summaries for features and saves the results to CSV files.
        This function performs the following steps:
        1. Reads CSV files from the specified folder that contain 'Feature_Values' in their filenames.
        2. Extracts feature names and their corresponding logistic regression values from the CSV files.
        3. Identifies features that appear more than once across the CSV files.
        4. For each selected feature that appears more than once, calculates statistical summaries (mean, standard deviation, min, max, median, mode, IQR) 
        for each logistic regression metric (coefficients, ORs, p-val, effect size).
        5. Saves the calculated summaries of logistic metrics to a CSV file.
    
        CSV File Columns: Feature, coefficient, CI Lower Coef, CI Upper Coef, Standard Error, Odds Ratios, CI Lower OR, CI Upper OR, P-Value, Effect Size.
        Statistical Summaries: mean, median, mode, min, max, standard deviation, IQR.
        
        Output:
            - feature_logistic_values: CSV file containing the combined feature names and logistic regression values.
            - LR_feature_val_calculations: CSV file containing the calculated statistical summaries for duplicated features.
        """

        print('Doing Logistic Regression Calculations')

        csv_metrics_files = [os.path.join(getattr(self, 'test_csv_folder'), file) 
                for file in os.listdir(getattr(self, 'test_csv_folder')) if 'Feature_Values' in file]
        
        variables = []
        ft_log_values = []
        value_names = []
        run_names = []
        
        # Loop to get the feature names and values from the csv files
        for file in csv_metrics_files:
            
            f = open(file, newline='')
            file_reader = list(csv.reader(f, dialect='excel'))
            f.close()
            file_reader = pd.DataFrame(file_reader)

            run_name = file.split('_')[-4]

            for index, row in file_reader.iterrows():
                if index == 0:
                    value_names = row[1:].values
                else:
                    variables.append(row[0]) # feature names
                    ft_log_values.append(row[1:].astype(float).values) # log_reg feature values
                    run_names.append(run_name) 

        # Combine feature names and values into a single DataFrame
        self.log_ft_all = (run_names, variables, value_names, ft_log_values) # tuple of all log_reg feature values
        # Create a DataFrame from the tuple
        self.feature_logistic_values = pd.concat(
            [pd.DataFrame(self.log_ft_all[3][i].reshape(1, -1), columns=self.log_ft_all[2], index=[self.log_ft_all[1][i]]).assign(Run_Name=self.log_ft_all[0][i]) for i in range(len(self.log_ft_all[1]))],
            axis=0
        )
        # Set index name to 'Feature'
        self.feature_logistic_values.index.name = 'Feature'
        self.feature_logistic_values.reset_index(inplace=True)
        self.feature_logistic_values.to_csv(f'{self.compiled_folder}/{self.architecture}_logistic_feature_vals_across_runs.csv', index=False)

        # Identify features that appear more than once across the CSV files
        for t in self.feature_importance_tally['Feature Names']:
            # get logistic values (eg. coefficients, ORs) and calculate mean, std dev, min, max, median, mode, iqr across values for said feature
            feature_data = self.feature_logistic_values[self.feature_logistic_values['Feature'] == t]
            for i, value_name in enumerate(value_names):
                stats = {
                    'Feature': [t],
                }
                for value_name in value_names:
                    stats[f'Mean_{value_name}'] = [round(feature_data[value_name].mean(), 2)]
                    stats[f'Std Dev_{value_name}'] = [round(feature_data[value_name].std(), 2)]
                    stats[f'Min_{value_name}'] = [round(feature_data[value_name].min(), 2)]
                    stats[f'Max_{value_name}'] = [round(feature_data[value_name].max(), 2)]
                    stats[f'Median_{value_name}'] = [round(feature_data[value_name].median(), 2)]
                    stats[f'Mode_{value_name}'] = [round(feature_data[value_name].mode().iloc[0], 2)]
                    stats[f'IQR_{value_name}'] = [round(feature_data[value_name].quantile(0.75) - feature_data[value_name].quantile(0.25), 2)]

            # Add tally to stats
            stats['Tally'] = [self.feature_importance_tally[self.feature_importance_tally['Feature Names'] == t]['Tally'].values[0]]
            # Make tally the second column
            stats = {k: stats[k] for k in ['Feature', 'Tally'] + [key for key in stats if key not in ['Feature', 'Tally']]}
        
            # Append stats to the DataFrame
            self.LR_feature_val_calculations = pd.concat([self.LR_feature_val_calculations, pd.DataFrame(stats)], ignore_index=True)
        
        self.LR_feature_val_calculations.to_csv(f'{self.compiled_folder}/{self.architecture}_{self.algorithm}_LR_Feature_Vals_Metrics_With_Tally.csv', index=False)

        print('Logistic Regression Calculations Done')
        return self.LR_feature_val_calculations, self.feature_logistic_values, self.log_ft_all

if __name__ == '__main__': 

    algorithm = 'log_reg'
    architecture = 'gc'

    file_save = Compare_CSVs(architecture=architecture, algorithm=algorithm)
    file_save.Feature_Importance_Values_Tally()
    file_save.Feature_Importance_Calculations()
    file_save.Train_Metrics_Calculations()
    file_save.Test_Metrics_Calculations()
    if algorithm == 'log_reg':
        file_save.Log_Reg_Scores()

