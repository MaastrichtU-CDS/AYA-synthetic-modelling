import csv
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import histplot
from calculate_metrics import Compare_CSVs

logging.basicConfig(filename='/root/Comparing_Metrics_Log', filemode='a', encoding='utf-8', level=logging.INFO, format=' %(levelname)s - %(message)s')

class Visualize_Compare_Metrics:
    def __init__(self, architecture, algorithm):
        '''
        Initialize the Visualize_Compare_Metrics class with the architecture and algorithm parameters.
        
        Below is a summary of the function calls that can be made with this class:
        - Visualize_AUC_Num_Matrix(): A heatmap of the mean AUC scores with standard deviation for each architecture and algorithm.
        - Visualize_AUC_Hist_Matrix(): A 4x3 grid of subplots of the AUC scores for each architecture and algorithm.
        - One_Test_Model_Metrics(): A plot of the test model metrics with error bars for a single architecture and algorithm.
        - Visualize_ARCH_Metrics(): A 2x2 grid of subplots of the test model metrics for each architecture.
        - Visualize_ALG_Metrics(): A 3x1 grid of subplots of the test model metrics for each algorithm.
        - Visualize_LR_OR(): A forest plot of the odds ratios with 95% confidence intervals for logistic regression.
        - Tally_Bins_Imp(): A visualization of the tally bins for importance scores for all architectures and algorithms.
        - Tally_Bins_OR(): A visualization of the tally bins for odds ratios for all architectures for logistic regression.

        Parameters:
        - architecture (str): The architecture used in the pipeline.
        - algorithm (str): The algorithm used in the pipeline.
        '''

        # Set architecture to parameter input
        self.architecture = architecture
        self.algorithm = algorithm

        # Create mapping dictionaries for consistent naming across functions
        self.arch_map = {
            'fast_ml': 'Fast ML',
            'gc': 'GC',
            'ctgan': 'CTGAN',
            'dp_cgan': 'DP-CGAN'
        }

        self.alg_map = {
            'log_reg': 'LR',
            'rf': 'RF',
            'mlp': 'MLP'
        }

        # Set the display title for current architecture and algorithm
        self.arch_title = self.arch_map[self.architecture]
        self.alg_title = self.alg_map[self.algorithm]

        # Create list of architectures and algorithms for iteration
        self.arch_list = list(self.arch_map.keys())
        self.alg_list = list(self.alg_map.keys())
        
        # Get the display names for labels in the visualizations
        self.arch_labels = [self.arch_map[arch] for arch in self.arch_list]
        self.alg_labels = [self.alg_map[alg] for alg in self.alg_list]

        # Set the path to the compiled results folder & create the visualize output directory
        self.compiled_folder = f'/root/{self.architecture}/results/{self.algorithm}/pipeline_run/Outputs/Compiled_Results'
        self.visualize_output_dir = os.path.join(self.compiled_folder, 'Visualize')
        if not os.path.exists(self.visualize_output_dir):
            os.makedirs(self.visualize_output_dir)

        # Load the feature importance metrics with tally doc
        self.feature_metrics = pd.read_csv(f'{self.compiled_folder}/{self.architecture}_{self.algorithm}_Feature_Importance_Metrics_With_Tally.csv')
        self.feature_names = self.feature_metrics['Feature Names']
        self.feature_importances = self.feature_metrics['Importance Mean']
        self.feature_importance_std_devs = self.feature_metrics['Importance Std Dev']
        self.tally_count = self.feature_metrics['Tally']

        # Create a DataFrame for the feature data of current architecture and algorithm
        self.feature_data = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.feature_importances,
            'StdDev': self.feature_importance_std_devs,
            'Tally': self.tally_count
        })

    
    def Visualize_AUC_Num_Matrix(self):
        '''
        Visualize the AUC mean and standard deviation scores for each architecture and algorithm in a heatmap.

        '''

        logging.info('Visualizing AUC Number Matrix for all architectures and algorithms')
        print('Visualizing AUC Number Matrix for all architectures and algorithms')

        sns.set_context('poster', font_scale=1.0)

        _, ax = plt.subplots(figsize=(12, 10))
        
        # enumerate through the architectures and algorithms
        mean_auc_matrix = np.zeros((len(self.arch_list), len(self.alg_list)))
        std_auc_matrix = np.zeros((len(self.arch_list), len(self.alg_list)))

        auc_scores = pd.DataFrame()

        for i, architecture in enumerate(self.arch_list):
            # for each architecture, iterate through the algorithms
            for j, algorithm in enumerate(self.alg_list):

                # get the avg AUC scores across runs for each architecture & algorithm
                test_csv_mets = pd.read_csv(f'/root/{architecture}/results/{algorithm}/pipeline_run/Outputs/Compiled_Results/{architecture}_{algorithm}_Test_Avg_Calculated_Metrics.csv')
                mean_auc_matrix[i, j] = test_csv_mets.iloc[6, 1]
                std_auc_matrix[i, j] = test_csv_mets.iloc[6, 2]

                # Log the average AUC scores for each architecture and algorithm
                logging.info('-' * 80)
                logging.info(f'{self.arch_map[architecture]}-{self.alg_map[algorithm]} Average AUC: {mean_auc_matrix[i, j]:.2f} ± {std_auc_matrix[i, j]:.2f}')
                logging.info('-' * 80)

                # Add to dataframe for CSV export
                new_row = pd.DataFrame({
                    'Architecture': [self.arch_map[architecture]],
                    'Algorithm': [self.alg_map[algorithm]],
                    'Mean_AUC': [mean_auc_matrix[i, j]],
                    'Std_AUC': [std_auc_matrix[i, j]]
                })
                auc_scores = pd.concat([auc_scores, new_row], ignore_index=True)

        # save AUC scores to a csv file
        auc_scores.to_csv(f'/root/AUC_Scores.csv')

        # Create a heatmap of the mean AUC scores with standard deviation
        sns.heatmap(mean_auc_matrix, annot=False, fmt=".2f", cmap="GnBu", 
               cbar_kws={'label': 'AUC Score'}, vmin=0.5, vmax=1.0)
        ax.set_xticklabels(self.alg_labels)
        ax.set_yticklabels(self.arch_labels)
        
        # Add text to the heatmap to state the mean and standard deviation of the AUC scores
        for i in range(len(self.arch_list)):
            for j in range(len(self.alg_list)):
                # Format text to be in the center of each cell
                ax.text(j + 0.5, i + 0.5, f"{mean_auc_matrix[i, j]:.2f}\n±{std_auc_matrix[i, j]:.2f}", ha='center', va='center', color='black')
            
        plt.suptitle('Average AUC (Std Dev) Across 1000 runs', y=0.95) # y = 0.95 to give enough space between title and figure
        plt.xlabel('Algorithm', labelpad=20, fontsize=25) 
        plt.ylabel('Architecture', labelpad=20, fontsize=25)
        plt.savefig('/root/AUC_Scores_Across_Architectures.pdf')
        plt.show()
        plt.close()

    def Visualize_AUC_Hist_Matrix(self):
        """
        Visualize the AUC scores (histogram matrix) for each architecture and algorithm in a 4x3 grid of subplots.
        """

        print('Visualizing AUC Histogram Matrix')
        logging.info('Visualizing AUC Histogram Matrix for all architectures and algorithms')

        # Set the plotting style to 'poster'
        sns.set_style('white')
        sns.set_context('poster', font_scale=1.0)

        # Create a 4x3 grid of subplots
        fig, axs = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(15, 20))
        fig.suptitle('AUC Score Frequency Distribution for each Architecture-Algorithm Across Runs')
        axs = axs.flatten()

        bin_start = 0.5  # Minimum AUC score
        bin_end = 1.0    # Maximum AUC score
        bin_width = 0.02 # Fixed bin width
        bins = np.arange(bin_start, bin_end + bin_width, bin_width)  # Create array of bin edges

        # iterate through the architectures 
        for i, architecture in enumerate(self.arch_list):

            #for each architecture, iterate through the algorithms
            for j, algorithm in enumerate(self.alg_list):

                # get the AUC scores for each run
                auc_scores = []
                test_csv_folder = f'/root/{architecture}/results/{algorithm}/pipeline_run/Outputs/Test_1000_runs/Test_CSVs'
                test_csv_metrics_files = [os.path.join(test_csv_folder, file) 
                            for file in os.listdir(test_csv_folder) if 'Metrics_run' in file]
                
                # iterate through the test CSV files and extract the AUC scores
                for file in test_csv_metrics_files:
                    with open(file, newline='') as f:
                        file_reader = list(csv.reader(f, dialect='excel'))
                    file_reader = pd.DataFrame(file_reader)

                    # get AUC score & append to list 
                    auc_score = float(file_reader.iloc[1, 12])
                    auc_scores.append(auc_score)

                # plot the histogram of the AUC scores by architecture and algorithm
                ax = axs[i * 3 + j]

                # Calculate the mean and standard deviation of the AUC scores for each architecture and algorithm
                mean_auc = np.mean(auc_scores)
                std_auc = np.std(auc_scores)

                # Use a vertical dashed line to indicate the mean AUC score
                ax.axvline(mean_auc, color='black', linestyle='dashed', linewidth=1.5)

                # Add text to the plot to state the mean and standard deviation of the AUC scores on the right of the dashed line
                ax.text(0.97, 0.95, f'Mean: {mean_auc:.2f}\n Std Dev: {std_auc:.2f}', color='black', va='top', ha='right', transform=ax.transAxes, fontsize=16)

                # Plot the histogram of the AUC scores
                ax.hist(auc_scores, bins=bins, color='black', alpha=0.8, density=False)
                
                # Add a kernel density estimate (KDE) curve (hyperimposed curve) to the histogram
                kde_x = np.linspace(bin_start, bin_end, 1000)
                kde = sns.kdeplot(data=auc_scores, ax=ax, color='blue', lw=1.2, label='KDE')
                
                # Scale the KDE curve to match the histogram height
                if kde.lines:  # Check if there are lines in the plot
                    y_vals = kde.lines[-1].get_ydata()
                    x_vals = kde.lines[-1].get_xdata()
                    max_count = np.max(np.histogram(auc_scores, bins=bins)[0])
                    scale_factor = max_count / np.max(y_vals)
                    kde.lines[-1].set_ydata(y_vals * scale_factor)

                # Plot the histogram of the AUC scores
                ax.set_title(f'{self.arch_map[architecture]}-{self.alg_map[algorithm]}')
                xticks = np.arange(0.5, 1.0, 0.1)
                ax.tick_params(axis='x', rotation=45)
                xticks = np.append(xticks, 1.0)
                ax.set_xticks(xticks)
                ax.set_yticks(np.arange(0, 1001, 100)) 
                ax.set_ylabel('Frequency', visible=False)

        fig.supxlabel('AUC Score')
        fig.supylabel('Frequency')

        plt.tight_layout()
        plt.savefig('/root/AUC_Scores_All_Datasets_Subplots.pdf')
        plt.show()
        plt.close()

    def Visualize_One_Test_Model_Metrics(self):
        '''
        Visualize the test model performance metrics with error bars for a single architecture and algorithm.
        Metrics: Accuracy, Precision, Recall, F1, and Log Loss
        '''

        # Load the test model metrics
        test_model_metrics = pd.read_csv(f'{self.compiled_folder}/{self.architecture}_{self.algorithm}_Test_Avg_Calculated_Metrics.csv')        
        
        # Drop balanced accuracy and AUC from the metrics. 
        test_model_metrics = test_model_metrics.drop([1])
        test_model_metrics = test_model_metrics.drop([6])
        
        test_model_metrics_names = test_model_metrics.iloc[:, 0]
        test_model_means = test_model_metrics.iloc[:, 1]
        test_model_std_devs = test_model_metrics.iloc[:, 2]

        print('Visualizing Singular Test Performance Metrics')
        logging.info(f'Visualizing Singular Test Performance Metrics for {self.arch_title}-{self.alg_title}')
        
        # Plot the test metrics with error bars
        sns.set_context('poster')
        fig = plt.figure(figsize=(8, 4))
        fig.suptitle(f'{self.arch_title}-{self.alg_title} Average Test Metrics Across Runs', y=1.2)
        ax = fig.add_axes([0, 0, 1, 1])

        # Create a scatter plot with error bars
        ax.errorbar(
            x=test_model_metrics_names, 
            y=test_model_means, 
            yerr=test_model_std_devs, 
            fmt='o',  
            markersize=10,
            capsize=5,
            color='dodgerblue',
            ecolor='black',
            linewidth=2
        )
        
        ax.set_xlabel('Metric')
        ax.set_ylabel('Average')
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45)

        plt.savefig(f'/root/{self.architecture}/results/{self.algorithm}/pipeline_run/Outputs/Compiled_Results/Visualize/{self.architecture}_{self.algorithm}_Test_Model_Metrics.pdf')
        plt.show()
        plt.close()

    def Visualize_ARCH_Metrics(self):
        '''
        Visualize the test model performance metrics with error bars. Grouping by architecture.
        Metrics: Accuracy, Precision, Recall, F1, and Log Loss.
        '''

        print('Visualizing Performance Metrics by architecture')
        logging.info('Visualizing Performance Metrics by architecture')

        # Set the plotting style to 'poster'
        sns.set_context('poster')

        # Create a 2x2 grid of subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 16), gridspec_kw={'hspace': 0.5, 'wspace':0.4})
        axs = axs.flatten()
        fig.suptitle('Average Test Metrics Across Runs by Architecture', y=0.95)

        # For each architecture, plot the test model metrics (for each algorithm)
        for i, architecture in enumerate(self.arch_list):
            for algorithm in self.alg_list:

                test_model_metrics = pd.read_csv(f'/root/{architecture}/results/{algorithm}/pipeline_run/Outputs/Compiled_Results/{architecture}_{algorithm}_Test_Avg_Calculated_Metrics.csv')        
                test_model_metrics = test_model_metrics.drop([1]) # drop balanced accuracy
                test_model_metrics = test_model_metrics.drop([6]) # drop AUC
                test_model_metrics_names = test_model_metrics.iloc[:, 0]
                test_model_means = test_model_metrics.iloc[:, 1]
                test_model_std_devs = test_model_metrics.iloc[:, 2]
                
                axs[i].errorbar(x=test_model_metrics_names, y=test_model_means, yerr=test_model_std_devs, 
                    fmt='o', label=f'{self.alg_map[algorithm]}', 
                    markersize=7, linewidth=1.5)

                axs[i].set_title(f'{self.arch_map[architecture]}', y=1.01)
                axs[i].set_xlabel('Metric')
                axs[i].set_ylabel('Average', labelpad=15)
                axs[i].set_ylim(0, 1)
                axs[i].tick_params(axis='x', rotation=45)
                axs[i].legend(title='Algorithm', loc='upper right', 
                             ncol=1, fontsize=15, title_fontsize=15,
                             borderaxespad=0.5, handletextpad=0.5)

        plt.savefig('/root/Test_Model_Metrics_Arch.pdf')
        plt.show()
        plt.close()

    def Visualize_ALG_Metrics(self):
        '''
        Visualize the test model performance metrics with error bars. Grouping by algorithm.
        Metrics: Accuracy, Precision, Recall, F1, and Log Loss.
        '''

        sns.set_context('poster')

        print('Visualizing Metrics by Algorithm')
        logging.info('Visualizing Metrics by Algorithm')

        # Create a 3x1 grid of subplots
        fig, axs = plt.subplots(1, 3, figsize=(25, 8))
        axs = axs.flatten()

        fig.suptitle('Average Test Metrics Across Runs by Algorithm')

        # For each algorithm, plot the test model metrics (for each architecture)
        for i, algorithm in enumerate(self.alg_list):
            for architecture in self.arch_list:
                test_model_metrics = pd.read_csv(f'/root/{architecture}/results/{algorithm}/pipeline_run/Outputs/Compiled_Results/{architecture}_{algorithm}_Test_Avg_Calculated_Metrics.csv')        
                test_model_metrics = test_model_metrics.drop([1])
                test_model_metrics = test_model_metrics.drop([6])

                test_model_metrics_names = test_model_metrics.iloc[:, 0]
                test_model_means = test_model_metrics.iloc[:, 1]
                test_model_std_devs = test_model_metrics.iloc[:, 2]
                
                axs[i].errorbar(x=test_model_metrics_names, y=test_model_means, yerr=test_model_std_devs, 
                    fmt='o', label=f'{self.arch_map[architecture]}', 
                    markersize=7, linewidth=1.5)

                axs[i].set_title(f'{self.alg_map[algorithm]}', y=1.01)
                axs[i].set_xlabel('Metric')
                axs[i].set_ylabel('Average', labelpad=15)
                axs[i].set_ylim(0, 1)
                axs[i].tick_params(axis='x', rotation=45)
                axs[i].legend(title='Architecture', loc='upper right', 
                             ncol=1, fontsize=10, title_fontsize=10,
                             borderaxespad=0.5, handletextpad=0.5)
            
            axs[i].set_yticks(np.arange(0, 1.1, 0.1))

        plt.tight_layout()
        plt.savefig('/root/Test_Model_Metrics_Alg.pdf')
        plt.show()
        plt.close()

    def Visualize_Importance_Scores(self):
        '''
        Visualize the feature importance scores with error bars in a forest plot.
        '''
    
        print(f'Visualizing Feature Importances for {self.arch_title}-{self.alg_title}')
        logging.info(f'Visualizing Feature Importances for {self.arch_title}-{self.alg_title}')

        # Create a bar plot of the mean feature importances with error bars using seaborn
        sns.set_context(context='poster')
        fig,ax = plt.subplots(figsize=(12, 12))

        # Plot using seaborn
        bars = sns.barplot(data=self.feature_data, x='Importance', y='Feature', color='dodgerblue', ax=ax)

        # Add error bars with a parameter for line size
        for bar, std in zip(bars.patches, self.feature_data['StdDev'].values): ax.errorbar(bar.get_width(), bar.get_y() + bar.get_height() / 2, 
            xerr=std, color='black', capsize=5, linewidth=2)

        # Add tally count to the y-axis labels
        ax.set_yticklabels([f'{name} ({count})' for name, count in zip(self.feature_data['Feature'], self.feature_data['Tally'])])

        # Center title and pad x and y labels
        fig.suptitle(f'{self.arch_title}-{self.alg_title} Average Feature Importances', x=0.3)
        ax.set_xlabel('Mean Importance Score', labelpad=25)
        ax.set_ylabel('Selected Feature (Tally)', labelpad=25)

        plt.savefig(f'/root/{self.architecture}/results/{self.algorithm}/pipeline_run/Outputs/Compiled_Results/Visualize/{self.architecture}_{self.algorithm}_Feature_Importances.pdf')
        plt.show()
        plt.close()

    def Visualize_LR_OR(self):
        '''
        Create a forest plot of the odds ratios with 95% confidence intervals for logistic regression.
        '''

        print(f'Visualizing LR OR(95% CI) for {self.arch_title}-{self.alg_title}')
        logging.info(f'Visualizing LR OR(95% CI) for {self.arch_title}-{self.alg_title}')

        # Load the Logistic Regression Odds Ratio & Confidence Interval data
        lr_data = pd.read_csv(f'/root/{self.architecture}/results/log_reg/pipeline_run/Outputs/Compiled_Results/{self.architecture}_log_reg_LR_Feature_Vals_Metrics_With_Tally.csv')

        # Extract the feature names, mean odds ratios, and 95% confidence intervals
        feature_name = lr_data['Feature']
        mean_OR = lr_data['Mean_Odds Ratio']
        mean_upper_CI_OR = lr_data['Mean_CI Upper OR']
        mean_lower_CI_OR = lr_data['Mean_CI Lower OR']
        lr_tally_count = lr_data['Tally']
        
        sns.set_context('poster')
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # values reversed because plot naturally ordered with first index at bottom. First index in self.tally_count is the highest tally.
        ax.errorbar(mean_OR[::-1], [f'{name} ({count})' for name, count in zip(feature_name[::-1], lr_tally_count[::-1])], 
            xerr=(mean_OR[::-1] - mean_lower_CI_OR[::-1], mean_upper_CI_OR[::-1] - mean_OR[::-1]), 
            fmt='o', color='black', ecolor='black', capsize=5)
        ax.axvline(x=1.0, color='black', linestyle='--')
        ax.set_xticks(np.arange(0, 3.5, 0.5)) # May want to change range for different algorithms or architectures

        fig.suptitle(f'{self.arch_title}-{self.alg_title} Forest Plot of Average Odds Ratios (95% CI)')
        ax.set_xlabel('Mean Odds Ratio', labelpad=25)
        ax.set_ylabel('Selected Feature (Tally)', labelpad=25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add text annotations for OR (lowerCI - upperCI) in line with each feature name
        for i, (or_val, lower_ci, upper_ci) in enumerate(zip(mean_OR[::-1], mean_lower_CI_OR[::-1], mean_upper_CI_OR[::-1])):
            ax.text(ax.get_xlim()[1] + 1, i, f'{or_val:.2f} ({lower_ci:.2f} - {upper_ci:.2f})', va='center', ha='right', color='black') # May want to adjust text position so no overlap with figure

        plt.savefig(f'/root/{self.architecture}/results/log_reg/pipeline_run/Outputs/Compiled_Results/Visualize/{self.architecture}_log_reg_LR_OR_Forest_Plot.pdf')
        plt.show()
        plt.close()

        # Create a DataFrame of the feature names and odds ratios with 95% confidence intervals
        Lr_Or_table = pd.DataFrame({
            'Feature': feature_name[::-1],
            'Odds Ratio (95% CI)': [f'{or_val:.2f} ({lower_ci:.2f} - {upper_ci:.2f})' for or_val, lower_ci, upper_ci in zip(mean_OR[::-1], mean_lower_CI_OR[::-1], mean_upper_CI_OR[::-1])]
        })

        # Save the DataFrame to a CSV file
        Lr_Or_table.to_csv(f'{self.visualize_output_dir}/{self.architecture}_{self.algorithm}_LR_OR_Table.csv', index=False)

    def Tally_Bins_Imp(self):
        '''
        Visualize the tally bins for importance scores for all architectures and algorithms.
        Bins: 900-1000, 500-900, 100-500, 0-100
        '''

        print('Visualizing Tally Bins for Importance Scores: all architectures and algorithms')
        logging.info('Visualizing Tally Bins for Importance Scores: all architectures and algorithms')

        results = pd.DataFrame()

        # Iterate through theall architectures and algorithms
        for arch_index, architecture in enumerate(self.arch_list):
            for alg_index, algorithm in enumerate(self.alg_list):
                
                bin_900_1000 = 0
                bin_500_900 = 0
                bin_100_500 = 0
                bin_0_100 = 0

                # Get feature metrics from each arch-alg pair
                feature_metrics = pd.read_csv(f'/root/{architecture}/results/{algorithm}/pipeline_run/Outputs/Compiled_Results/{architecture}_{algorithm}_Feature_Importance_Metrics_With_Tally.csv')
                tally_count = feature_metrics['Tally']
                avg_imp = feature_metrics['Importance Mean'].mean() # Mean importance score for all features across the document

                imp_ft = feature_metrics['Importance Mean'] # Importance score of individual feature
                feature_names = feature_metrics['Feature Names']
                
                print(f"\n {self.arch_map[architecture]}-{self.alg_map[algorithm]} Average Importance: {avg_imp:.2f}")
                logging.info(f"\n {self.arch_map[architecture]}-{self.alg_map[algorithm]} Average Importance: {avg_imp:.2f}")
                
                for tally_index in range(len(tally_count)):
                    if tally_count[tally_index] >= 900:
                        bin_900_1000 += 1
                    elif tally_count[tally_index] >= 500:
                        bin_500_900 += 1
                    elif tally_count[tally_index] >= 100:
                        bin_100_500 += 1
                    else:
                        bin_0_100 += 1

                # Print the feature names with importance greater or less than the average importance
                for ft, imp in zip(feature_names, imp_ft):
                    if imp >= avg_imp:
                        print(f'{ft} has importance {imp} which is greater than the average importance of {avg_imp:.2f}')
                        logging.info(f'{ft} has importance {imp} which is greater than the average importance of {avg_imp:.2f}')
                    else:
                        print(f'{ft} has importance {imp} which is less than the average importance of {avg_imp:.2f}')
                        logging.info(f'{ft} has importance {imp} which is less than the average importance of {avg_imp:.2f}')

                # Create lists to store the feature names with importance greater than the average importance
                ft_above_avg_900 = []
                ft_above_avg_500 = []
                ft_above_avg_100 = []
                ft_above_avg_0 = []

                # Create counters to store the number of features with importance greater than the average importance
                ft_above_avg_900_counter = 0
                ft_above_avg_500_counter = 0
                ft_above_avg_100_counter = 0
                ft_above_avg_0_counter = 0

                # for a feature with tally count of eg. 900-1000, check if importance is greater than average importance. If so, add tally to above tally list.
                # This is done for each bin range
                for tally_index in range(len(tally_count)):
                    if tally_count[tally_index] >= 900:
                        if imp_ft[tally_index] >= avg_imp:
                            ft_above_avg_900.append(feature_names[tally_index])
                            ft_above_avg_900_counter += 1
                    elif tally_count[tally_index] >= 500:
                        if imp_ft[tally_index] >= avg_imp:
                            ft_above_avg_500.append(feature_names[tally_index])
                            ft_above_avg_500_counter += 1
                    elif tally_count[tally_index] >= 100:
                        if imp_ft[tally_index] >= avg_imp:
                            ft_above_avg_100.append(feature_names[tally_index])
                            ft_above_avg_100_counter += 1
                    else:
                        if imp_ft[tally_index] >= avg_imp:
                            ft_above_avg_0.append(feature_names[tally_index])
                            ft_above_avg_0_counter += 1
                
                # Create tuples for each bin range
                A = (bin_900_1000, ft_above_avg_900, ft_above_avg_900_counter)
                B = (bin_500_900, ft_above_avg_500, ft_above_avg_500_counter)
                C = (bin_100_500, ft_above_avg_100, ft_above_avg_100_counter)
                D = (bin_0_100, ft_above_avg_0, ft_above_avg_0_counter)

                # Create a new row for each bin for dataframe structure
                new_row_A = pd.DataFrame({
                    'Architecture': [self.arch_map[architecture]],
                    'Algorithm': [self.alg_map[algorithm]],
                    'Bin': ['900-1000'],
                    'Count': [bin_900_1000],
                    'Num_Imp_Above_Avg': [ft_above_avg_900_counter], 
                    'Imp_Features_Above_Avg': [', '.join(ft_above_avg_900) if ft_above_avg_900 else ''],
                })
                
                new_row_B = pd.DataFrame({
                    'Architecture': [self.arch_map[architecture]],
                    'Algorithm': [self.alg_map[algorithm]],
                    'Bin': ['500-900'],
                    'Count': [bin_500_900],
                    'Num_Imp_Above_Avg': [ft_above_avg_500_counter], 
                    'Imp_Features_Above_Avg': [', '.join(ft_above_avg_500) if ft_above_avg_500 else ''],
                })
                
                new_row_C = pd.DataFrame({
                    'Architecture': [self.arch_map[architecture]],
                    'Algorithm': [self.alg_map[algorithm]],
                    'Bin': ['100-500'],
                    'Count': [bin_100_500],
                    'Num_Imp_Above_Avg': [ft_above_avg_100_counter], 
                    'Imp_Features_Above_Avg': [', '.join(ft_above_avg_100) if ft_above_avg_100 else ''],
                })
                
                new_row_D = pd.DataFrame({
                    'Architecture': [self.arch_map[architecture]],
                    'Algorithm': [self.alg_map[algorithm]],
                    'Bin': ['0-100'],
                    'Count': [bin_0_100],
                    'Num_Imp_Above_Avg': [ft_above_avg_0_counter],
                    'Imp_Features_Above_Avg': [', '.join(ft_above_avg_0) if ft_above_avg_0 else ''],
                })

                # Concatenate the new rows to the results DataFrame
                results = pd.concat([results, new_row_A, new_row_B, new_row_C, new_row_D], ignore_index=True)
        
                print(f'\n Architecture: {self.arch_map[architecture]}, Algorithm: {self.alg_map[algorithm]}')
                print(f'Features with tally 900-1000 and importance greater than average importance, counter: {ft_above_avg_900}, {ft_above_avg_900_counter}')
                print(f'Features with tally 500-900 and importance greater than average importance, counter: {ft_above_avg_500}, {ft_above_avg_500_counter}')
                print(f'Features with tally 100-500 and importance greater than average importance, counter: {ft_above_avg_100}, {ft_above_avg_100_counter}')
                print(f'Features with tally 0-100 and importance greater than average importance, counter: {ft_above_avg_0}, {ft_above_avg_0_counter}')
                print('-' * 80)

                logging.info(f' \n Architecture: {self.arch_map[architecture]}, Algorithm: {self.alg_map[algorithm]}')
                logging.info(f'Features with tally 900-1000 and importance greater than average importance, counter: {ft_above_avg_900}, {ft_above_avg_900_counter}')
                logging.info(f'Features with tally 500-900 and importance greater than average importance, counter: {ft_above_avg_500}, {ft_above_avg_500_counter}')
                logging.info(f'Features with tally 100-500 and importance greater than average importance, counter: {ft_above_avg_100}, {ft_above_avg_100_counter}')
                logging.info(f'Features with tally 0-100 and importance greater than average importance, counter: {ft_above_avg_0}, {ft_above_avg_0_counter}')
                logging.info('-' * 80)

        # print(results)

        # Log the results with proper formatting
        logging.info("\n=== Importance Score Tally Bins Analysis Results ===\n")
        for arch in self.arch_list:
            for alg in self.alg_list:
                arch_display_name = self.arch_map[arch]
                alg_display_name = self.alg_map[alg]
                
                logging.info(f"\n--- {arch_display_name}-{alg_display_name} Results ---")
                subset = results[(results['Architecture'] == arch_display_name) & (results['Algorithm'] == alg_display_name)]
                
                for _, row in subset.iterrows():
                    logging.info(f"Bin: {row['Bin']} - Count: {row['Count']}")
                    logging.info(f"  Importance > Avg: {row['Num_Imp_Above_Avg']} features")
                    if row['Imp_Features_Above_Avg']:
                        logging.info(f"    Features: {row['Imp_Features_Above_Avg']}")

        # Save the results to a CSV file
        results.to_csv(f'/root/Tally_Bins_Imp.csv', index=False)

    def Tally_Bins_OR(self):
        '''
        Visualize the tally bins for logistic regression odds ratios (ORs) for all architectures and algorithms.
        Bins: 900-1000, 500-900, 100-500, 0-100
        '''

        print('\n Visualizing Tally Bins for ORs: all architectures for LR')
        logging.info('-' * 80)
        logging.info('\n Visualizing Tally Bins for ORs: all architectures for LR')

        # initialize new results dataframe for OR information
        results = pd.DataFrame()

        # Only run for log_reg
        if algorithm == 'log_reg':
            # Iterate through all architectures 
            for arch_index, architecture in enumerate(self.arch_list):   
                # Initialize bin counter             
                bin_900_1000 = 0
                bin_500_900 = 0
                bin_100_500 = 0
                bin_0_100 = 0

                # initialize bin list
                avg_or_900 = []
                avg_or_500 = []
                avg_or_100 = []
                avg_or_0 = []   

                #Get logistic regression OR data for each architecture 
                or_metrics = pd.read_csv(f'/root/{architecture}/results/log_reg/pipeline_run/Outputs/Compiled_Results/{architecture}_log_reg_LR_Feature_Vals_Metrics_With_Tally.csv')
                avg_or = or_metrics['Mean_Odds Ratio'].mean() # Get the average OR across runs
                or_ft = or_metrics['Mean_Odds Ratio'] # Get the inidvidual OR of features
                tally_count = or_metrics['Tally']
                feature_names = or_metrics['Feature']
                
                print(f"\n {self.arch_map[architecture]}-{self.alg_map[algorithm]} Average OR: {avg_or:.2f}")
                logging.info(f"\n {self.arch_map[architecture]}-{self.alg_map[algorithm]} Average OR: {avg_or:.2f}")
                
                # Check if OR is greater or less than the average OR
                for tally_index in range(len(tally_count)):
                    if tally_count[tally_index] >= 900:
                        bin_900_1000 += 1
                    elif tally_count[tally_index] >= 500:
                        bin_500_900 += 1
                    elif tally_count[tally_index] >= 100:
                        bin_100_500 += 1
                    else:
                        bin_0_100 += 1

                # Print the feature names with OR greater or less than the average OR
                for ft, odds in zip(feature_names, or_ft):
                    if odds >= avg_or:
                        print(f'{ft} has odds ratio (OR) of {odds} which is greater than the average OR of {avg_or:.2f}')
                        logging.info(f'{ft} has odds ratio (OR) of {odds} which is greater than the average OR of {avg_or:.2f}')
                    else:
                        print(f'{ft} has odds ratio (OR) {odds} which is less than the average OR of {avg_or:.2f}')
                        logging.info(f'{ft} has odds ratio (OR) {odds} which is less than the average OR of {avg_or:.2f}')

                # Initialize lists to store the feature names with OR greater than the average OR
                or_above_avg_900 = []
                or_above_avg_500 = []
                or_above_avg_100 = []
                or_above_avg_0 = []

                # Initialize counters to store the number of features with OR greater than the average OR
                or_above_avg_900_counter = 0
                or_above_avg_500_counter = 0
                or_above_avg_100_counter = 0
                or_above_avg_0_counter = 0

                # for a feature with tally count of eg. 900-1000, check if importance is greater than average OR then add tally to above tally list 
                # This is done for each bin range
                for tally_index in range(len(tally_count)):
                    if tally_count[tally_index] >= 900:
                        if or_ft[tally_index] >= avg_or:
                            or_above_avg_900.append(feature_names[tally_index])
                            or_above_avg_900_counter += 1
                    elif tally_count[tally_index] >= 500:
                        if or_ft[tally_index] >= avg_or:
                            or_above_avg_500.append(feature_names[tally_index])
                            or_above_avg_500_counter += 1
                    elif tally_count[tally_index] >= 100:
                        if or_ft[tally_index] >= avg_or:
                            or_above_avg_100.append(feature_names[tally_index])
                            or_above_avg_100_counter += 1
                    else:
                        if or_ft[tally_index] >= avg_or:
                            or_above_avg_0.append(feature_names[tally_index])
                            or_above_avg_0_counter += 1

                # Create tuples for each bin range
                A = (bin_900_1000, or_above_avg_900, or_above_avg_900_counter)
                B = (bin_500_900, or_above_avg_500, or_above_avg_500_counter)
                C = (bin_100_500, or_above_avg_100, or_above_avg_100_counter)
                D = (bin_0_100, or_above_avg_0, or_above_avg_0_counter)

                # Create a new row for each bin for dataframe structure
                new_row_A = pd.DataFrame({
                    'Architecture': [self.arch_map[architecture]],
                    'Algorithm': [self.alg_map[algorithm]],
                    'Bin': ['900-1000'],
                    'Count': [bin_900_1000],
                    'Num_OR_Above_Avg': [or_above_avg_900_counter],
                    'OR_Features_Above_Avg': [', '.join(or_above_avg_900) if or_above_avg_900 else '']
                })
                
                new_row_B = pd.DataFrame({
                    'Architecture': [self.arch_map[architecture]],
                    'Algorithm': [self.alg_map[algorithm]],
                    'Bin': ['500-900'],
                    'Count': [bin_500_900],
                    'Num_OR_Above_Avg': [or_above_avg_500_counter],
                    'OR_Features_Above_Avg': [', '.join(or_above_avg_500) if or_above_avg_500 else '']
                })
                
                new_row_C = pd.DataFrame({
                    'Architecture': [self.arch_map[architecture]],
                    'Algorithm': [self.alg_map[algorithm]],
                    'Bin': ['100-500'],
                    'Count': [bin_100_500],
                    'Num_OR_Above_Avg': [or_above_avg_100_counter],
                    'OR_Features_Above_Avg': [', '.join(or_above_avg_100) if or_above_avg_100 else '']
                })
                
                new_row_D = pd.DataFrame({
                    'Architecture': [self.arch_map[architecture]],
                    'Algorithm': [self.alg_map[algorithm]],
                    'Bin': ['0-100'],
                    'Count': [bin_0_100],
                    'Num_OR_Above_Avg': [or_above_avg_0_counter],
                    'OR_Features_Above_Avg': [', '.join(or_above_avg_0) if or_above_avg_0 else '']
                })

                # Concatenate the new rows to the results DataFrame
                results = pd.concat([results, new_row_A, new_row_B, new_row_C, new_row_D], ignore_index=True)
        
                print(f'\n Architecture: {self.arch_map[architecture]}, Algorithm: {self.alg_map[algorithm]}')
                print(f'Features with tally 900-1000 and OR greater than average OR, counter: {or_above_avg_900}, {or_above_avg_900_counter}')
                print(f'Features with tally 500-900 and OR greater than average OR, counter: {or_above_avg_500}, {or_above_avg_500_counter}')
                print(f'Features with tally 100-500 and OR greater than average OR, counter: {or_above_avg_100}, {or_above_avg_100_counter}')
                print(f'Features with tally 0-100 and OR greater than average OR, counter: {or_above_avg_0}, {or_above_avg_0_counter}')
                print('-' * 80)

                logging.info(f' \n Architecture: {self.arch_map[architecture]}, Algorithm: {self.alg_map[algorithm]}')
                logging.info(f'Features with tally 900-1000 and OR greater than average OR, counter: {or_above_avg_900}, {or_above_avg_900_counter}')
                logging.info(f'Features with tally 500-900 and OR greater than average OR, counter: {or_above_avg_500}, {or_above_avg_500_counter}')
                logging.info(f'Features with tally 100-500 and OR greater than average OR, counter: {or_above_avg_100}, {or_above_avg_100_counter}')
                logging.info(f'Features with tally 0-100 and OR greater than average OR, counter: {or_above_avg_0}, {or_above_avg_0_counter}')
                logging.info('-' * 80)

            # print(results)

            # Log the results with proper formatting
            print("\n=== Odds Ratios (OR) Tally Bins Analysis Results ===\n")
            logging.info("\n=== Odds Ratios (OR) Tally Bins Analysis Results ===\n")
            for arch in self.arch_list:
                arch_display_name = self.arch_map[arch]
                
                print(f"\n--- {arch_display_name}-LR Results ---")
                logging.info(f"\n--- {arch_display_name}-LR Results ---")
                subset = results[(results['Architecture'] == arch_display_name)]
                
                for _, row in subset.iterrows():
                    print(f"Bin: {row['Bin']} - Count: {row['Count']}")
                    print(f"  OR > Avg: {row['Num_OR_Above_Avg']} features")
                    logging.info(f"Bin: {row['Bin']} - Count: {row['Count']}")
                    logging.info(f"  OR > Avg: {row['Num_OR_Above_Avg']} features")
                    if row['OR_Features_Above_Avg']:
                        print(f"    Features: {row['OR_Features_Above_Avg']}")
                        logging.info(f"    Features: {row['OR_Features_Above_Avg']}")

        # Save the results to a CSV file
        results.to_csv(f'/root/Tally_Bins_OR.csv', index=False)

    

if __name__ == '__main__':

    algorithm = 'log_reg'
    architecture = 'dp_cgan'
    
    vis = Visualize_Compare_Metrics(architecture, algorithm)
    vis.Visualize_One_Test_Model_Metrics()
    vis.Visualize_Importance_Scores()
    if algorithm == 'log_reg':
        vis.Visualize_LR_OR()

    # Only to run after all comparisons from compare_metrics_features done
    vis.Visualize_AUC_Num_Matrix()
    vis.Visualize_AUC_Hist_Matrix()
    vis.Visualize_ARCH_Metrics()
    vis.Visualize_ALG_Metrics()
    vis.Tally_Bins_Imp()
    vis.Tally_Bins_OR()

    print("Visualizations Done")