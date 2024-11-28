import csv
import os
import pandas as pd
import datetime
import pytz

from file_handling import FileSave 

class Compare_CSVs:
    def __init__(self):

        # self.alg = input("Algorithm (log_reg, rf, mlp): ")

        # if self.alg == "log_reg":
        self.csv_folder = '/root/fast_ml/results/log_reg/Outputs/2711_CSVs'
        # self.model_folder = input("Choose model: ctgan, fast_ml, gc: ")
        self.model_folder = 'fast_ml'

        
        self.algorithm = 'log_reg'

        # elif self.alg == "rf":
        #     self.csv_folder = '/root/fast_ml/results/rf/Outputs/2611_CSVs'
        # elif self.alg == "mlp":
        #     self.csv_folder = '/root/fast_ml/results/mlp/Outputs/2611_CSVs'

         
    def Compare_Metrics(self):
        
        ''' 

        Function to concatenate metrics of all files in csv_metrics_files and keep 1st col as metric name.

        '''

        self.csv_metrics_files = [os.path.join(self.csv_folder, file) 
                 for file in os.listdir(self.csv_folder) if 'Metrics' in file]
        
        self.metric_names = []
        self.metric_values = []
  
        for file in self.csv_metrics_files:
            # self.file_reader = pd.read_csv(file)
            
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
        self.combined_metrics.to_csv(os.path.join(self.csv_folder, 'combined_metricssss.csv'), index=False)
        

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

        # Combine file names, ft names and ft values into a single DataFrame
        self.file_name = pd.Series()
        for file in self.csv_ft_files:
            self.run = '_'.join(file.split('/')[-1].split('_')[4:6])
            self.file_name = self.file_name.append(pd.Series(f'{self.run}_{self.model_folder}_{self.algorithm}'))

            f = open(file, newline='')
            self.file_reader = list(csv.reader(f, dialect='excel'))
            f.close()
            self.file_reader = pd.DataFrame(self.file_reader)
            
            # Add a blank row before each file's feature names and values
            self.feature_names_indiv.append(pd.Series([self.file_name]))
            self.feature_values.append(pd.Series([None])) 

            self.feature_names_indiv.append(self.file_reader.iloc[1:, 0])
            self.feature_values.append(self.file_reader.iloc[1:, 1])
            
        self.con_ft_names = pd.concat(self.feature_names_indiv, axis=0).reset_index(drop=True)
        self.con_ft_values = pd.concat(self.feature_values, axis=0).reset_index(drop=True)

        # Combine feature names and values into a single DataFrame
        self.combined_ft_vals = pd.concat([self.file_name, self.con_ft_names, self.con_ft_values], axis=1)
        self.combined_ft_vals.columns = ['File Name', 'Feature Names', 'Importance']

        # Save the combined metrics to a CSV file
        self.combined_ft_vals.to_csv(os.path.join(self.csv_folder, 'combined_ft_vals.csv'), index=False)


        # FIXME: seen_features is not working as expected.

        # self.seen_features = []

        # if con_ft_names contains multiple instances of the same feature names, print the feature and add to list of seen features
        # if self.combined_ft_vals["Feature Names"].duplicated().any():
        #     duplicated_features = self.combined_ft_vals[self.combined_ft_vals["Feature Names"].duplicated()]["Feature Names"]
        #     for feature in duplicated_features:
        #         importance_list = self.combined_ft_vals[self.combined_ft_vals["Feature Names"] == feature]["Importance"].tolist()
        #         file_name = self.combined_ft_vals[self.combined_ft_vals["Feature Names"] == feature]["File Name"].values[0]
        #         self.seen_features.append({
        #             'File Name': file_name,
        #             'Feature Names': feature,

        if not self.combined_ft_vals["Feature Names"].duplicated().any():
            return []

        duplicated_features = self.combined_ft_vals[self.combined_ft_vals["Feature Names"].duplicated()]["Feature Names"]
        
        # Create a dictionary to store feature data instead of appending to list
        self.feature_data = {}

        for feature in duplicated_features:
            importance_list = self.combined_ft_vals[
                self.combined_ft_vals["Feature Names"] == feature
                ]["Importance"].tolist()
            file_name = self.combined_ft_vals[self.combined_ft_vals["Feature Names"] == feature]["File Name"].values[0]
            
            if feature not in self.feature_data:
                self.feature_data[feature] = {
                    'File Name': file_name,
                    'Feature Names': feature,
                    'Importance': importance_list
                }

        self.seen_features = list(self.feature_data.values())
        

        print('Ici \n', self.seen_features)
        
        return self.seen_features
     


        

        


                

        # self.seen_features = pd.DataFrame(self.seen_features)
        # self.seen_features.to_csv(os.path.join(self.csv_folder, 'seen_features_ICI.csv'), index=False)
        
        # print('Ici \n', self.seen_features)



    def Tally_Features(self):

        # Tally loop
        self.feature_names_whole = pd.read_csv('/root/X_cols.csv', delimiter=',', encoding='latin1', names=['Feature Names'])
        self.feature_names_whole['Tally'] = 0
        
        for feature_series in self.feature_names_indiv:
            for feature in feature_series.dropna().values:
                if feature in self.feature_names_whole['Feature Names'].values:
                    self.feature_names_whole.loc[self.feature_names_whole['Feature Names'] == feature, 'Tally'] += 1
                else:
                    pass

        self.feature_names_whole.to_csv(os.path.join(self.csv_folder, 'tallyyy.csv'), index=False)
        
    # TODO: variation of scores in fts. ranges. Would have to consider how to think about which one outperforms like how's that indicated. 
        
    # print('\n Ft Whole Tally \n ', self.feature_names_whole) # indent into for loop to get tally counter incrementing.
    # print('vals: ', self.feature_values)
    # print('name', self.feature_names_indiv)
    # print('\n rd\n ', self.file_reader)
    # print('\n con\n ', self.con_ft_values)



if __name__ == '__main__': 
    file_save = Compare_CSVs()
    file_save.Compare_Metrics()
    file_save.Features_Concat()
    file_save.Tally_Features()



    # TODO:
        # Then will want to add this maybe by 
            # a) comparing all the files in the folder, but ideally as a final loop/code of 
                # eg. the generated data loop which we need to make a non-static var for 
        # Maybe later down the line will want to compare those metric values by each other so then the number is like a difference score --> Would have to consider how to think about which one outperforms like how's that indicated. 

