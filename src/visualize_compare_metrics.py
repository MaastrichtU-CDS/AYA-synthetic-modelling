import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import histplot
from matplotlib.backends.backend_pdf import PdfPages
from compare_metrics_features import Compare_CSVs

class Visualize_Compare_Metrics:
    def __init__(self, architecture, algorithm):
        '''
        Initialize the Visualize_Compare_Metrics class with the architecture and algorithm parameters.
        
        Below is a summary of the function calls that can be made with this class:
        - AUC_Num_Matrix: A heatmap of the mean AUC scores with standard deviation for each architecture and algorithm.
        - AUC_Sub_Matrix: A 4x3 grid of subplots of the AUC scores for each architecture and algorithm.
        - One_Test_Model_Metrics: A plot of the test model metrics with error bars for a single architecture and algorithm.
        - ALL_Model_Metrics: A plot of the test model metrics with error bars for all architectures and algorithms.
        - ARCH_Model_Metrics: A 2x2 grid of subplots of the test model metrics for each architecture.
        - ALG_Model_Metrics: A 3x1 grid of subplots of the test model metrics for each algorithm.
        - Importances_histo_error: A histogram of the mean feature importances with error bars.
        - Importances_Forest: A bar plot of the mean feature importances with error bars.
        - LR_OR: A forest plot of the odds ratios with 95% confidence intervals for logistic regression.

        Parameters:
        - architecture (str): The architecture used in the pipeline.
        - algorithm (str): The algorithm used in the pipeline.
        '''

        # Set architecture to parameter input
        self.architecture = architecture
        # Create list of architectures (arch) for the (meta-)visualizations
        self.arch = ['fast_ml', 'gc', 'ctgan','dp_cgan']
        # Set algorithm to parameter input
        self.algorithm = algorithm
        # Create list of algorithms (alg) for the (meta-)visualizations
        self.alg = ['log_reg', 'rf', 'mlp']

        # Set the path to the compiled results folder
        self.compiled_folder = f'/root/{self.architecture}/results/{self.algorithm}/pipeline_run/Outputs/Compiled_Results'

        # Create a folder to store the visualizations
        self.visualize_output_dir = os.path.join(self.compiled_folder, 'Visualize')
        if not os.path.exists(self.visualize_output_dir):
            os.makedirs(self.visualize_output_dir)

        # Load the feature importance metrics with tally
        self.feature_metrics = pd.read_csv(f'{self.compiled_folder}/{self.architecture}_{self.algorithm}_Feature_Importance_Metrics_With_Tally.csv')
        self.feature_names = self.feature_metrics['Feature Names']
        self.feature_importances = self.feature_metrics['Importance Mean']
        self.feature_importance_std_devs = self.feature_metrics['Importance Std Dev']
        self.tally_count = self.feature_metrics['Tally']

    
    def Visualize_AUC_Num_Matrix(self):
        '''
        Visualize the AUC mean and standard deviation scores for each architecture and algorithm in a heatmap.
        '''

        print('Doing AUC Num Visualization')

        _, ax = plt.subplots(figsize=(8, 6))
        
        # enumerate through the architectures and algorithms
        mean_auc_matrix = np.zeros((len(self.arch), len(self.alg)))
        std_auc_matrix = np.zeros((len(self.arch), len(self.alg)))

        for i, architecture in enumerate(self.arch):
            # for each architecture, iterate through the algorithms
            for j, algorithm in enumerate(self.alg):

                # get the avg AUC scores across runs
                test_csv_mets = pd.read_csv(f'/root/{architecture}/results/{algorithm}/pipeline_run/Outputs/Compiled_Results/{architecture}_{algorithm}_Test_Avg_Calculated_Metrics.csv')
                mean_auc_matrix[i, j] = test_csv_mets.iloc[6, 1]
                std_auc_matrix[i, j] = test_csv_mets.iloc[6, 2]

                # Check to ensure proper values are being stored
                print(f'{architecture} {algorithm} {mean_auc_matrix[i, j]}')
                print(f'{architecture} {algorithm} {std_auc_matrix[i, j]}')

        # Create a heatmap of the mean AUC scores with standard deviation
        sns.heatmap(mean_auc_matrix, annot=False, fmt=".2f", cmap="Blues", xticklabels=self.alg, yticklabels=self.arch, cbar_kws={'label': 'AUC Score'})
        for i in range(len(self.arch)):
            for j in range(len(self.alg)):
                text_color = 'black' if mean_auc_matrix[i, j] < 0.8 else 'white'
                ax.text(j + 0.5, i + 0.5, f"{mean_auc_matrix[i, j]:.2f}\n±{std_auc_matrix[i, j]:.2f}", ha='center', va='center', color=text_color, fontsize=12)
            
        plt.title('Mean AUC (Standard Deviation) of 1000 Datasets Across Algorithms and Architectures')
        plt.xlabel('Algorithm')
        plt.ylabel('Architecture')
        plt.savefig('/root/AUC_Scores_Across_Gens.pdf')
        plt.show()
        plt.close()

    def Visualize_ACCURACY_Num_Matrix(self):
        '''
        Visualize the AUC mean and standard deviation scores for each architecture and algorithm in a heatmap.
        '''

        print('Doing Accuracy Num Visualization')

        _, ax = plt.subplots(figsize=(8, 6))
        
        # enumerate through the architectures and algorithms
        mean_ac_matrix = np.zeros((len(self.arch), len(self.alg)))
        std_ac_matrix = np.zeros((len(self.arch), len(self.alg)))

        for i, architecture in enumerate(self.arch):
            # for each architecture, iterate through the algorithms
            for j, algorithm in enumerate(self.alg):

                # get the avg AUC scores across runs
                test_csv_mets = pd.read_csv(f'/root/{architecture}/results/{algorithm}/pipeline_run/Outputs/Compiled_Results/{architecture}_{algorithm}_Test_Avg_Calculated_Metrics.csv')
                mean_ac_matrix[i, j] = test_csv_mets.iloc[0, 1]
                std_ac_matrix[i, j] = test_csv_mets.iloc[0, 2]

                # Check to ensure proper values are being stored
                print(f'{architecture} {algorithm} {mean_ac_matrix[i, j]}')
                print(f'{architecture} {algorithm} {std_ac_matrix[i, j]}')

        # Create a heatmap of the mean AUC scores with standard deviation
        sns.heatmap(mean_ac_matrix, annot=False, fmt=".2f", cmap="Blues", xticklabels=self.alg, yticklabels=self.arch, cbar_kws={'label': 'Accuracy Score'})
        for i in range(len(self.arch)):
            for j in range(len(self.alg)):
                text_color = 'black' if mean_ac_matrix[i, j] < 0.8 else 'white'
                ax.text(j + 0.5, i + 0.5, f"{mean_ac_matrix[i, j]:.2f}\n±{std_ac_matrix[i, j]:.2f}", ha='center', va='center', color=text_color, fontsize=12)
            
        plt.title('Mean Accuracy (Standard Deviation) of 1000 Datasets Across Algorithms and Architectures')
        plt.xlabel('Algorithm')
        plt.ylabel('Architecture')
        plt.savefig('/root/ACCURACY_Scores_Across_Gens.pdf')
        plt.show()
        plt.close()

    def Visualize_AUC_Sub_Matrix(self):
        """
        Visualize the AUC scores (histogram matrix) for each architecture and algorithm in a 4x3 grid of subplots.
        FIXME: Change title of graph
        """

        print('Doing AUC Sub Matrix Visualization')

        # Create a 4x3 grid of subplots
        fig, axs = plt.subplots(4, 3, figsize=(15, 20))
        axs = axs.flatten()

        bin_start = 0.5  # Minimum AUC score
        bin_end = 1.0    # Maximum AUC score
        bin_width = 0.02 # Fixed bin width
        bins = np.arange(bin_start, bin_end + bin_width, bin_width)  # Create array of bin edges

        # enemurate through the architectures and algorithms
        for i, architecture in enumerate(self.arch):

            #for each architecture, iterate through the algorithms
            for j, algorithm in enumerate(self.alg):

                # get the AUC scores for each run
                AUC_scores = []
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
                    AUC_scores.append(auc_score)

                # plot the histogram of the AUC scores by architecture and algorithm
                ax = axs[i * 3 + j]
                # Calculate the mean and standard deviation of the AUC scores for each architecture and algorithm
                mean_auc = np.mean(AUC_scores)
                std_auc = np.std(AUC_scores)

                # Use a vertical line to indicate the mean with dashed line
                ax.axvline(mean_auc, color='b', linestyle='dashed', linewidth=1)

                # Add text to the plot to state the mean and standard deviation of the AUC scores
                ax.text(0.95, 0.95, f'Mean: {mean_auc:.2f}\n Std: {std_auc:.2f}', color='b', va='top', ha='right', transform=ax.transAxes, fontsize=15)

                # Plot the histogram of the AUC scores
                ax.hist(AUC_scores, bins=bins, color='black', alpha=0.8, density=False)

                if architecture == 'fast_ml':
                    arch = 'Fast ML'
                elif architecture == 'gc':
                    arch = 'GC'
                elif architecture == 'ctgan':
                    arch = 'CTGAN'
                elif architecture == 'dp_cgan':
                    arch = 'DP-CGAN'
            
                if algorithm == 'log_reg':
                    alg = 'LR'
                elif algorithm == 'rf':
                    alg = 'RF'
                elif algorithm == 'mlp':
                    alg = 'MLP'

                # Plot the histogram of the AUC scores
                ax.set_title(f'{arch}-{alg}', fontsize=15)
                ax.set_xticks(np.arange(0.5, 1.0, 0.05)) # Min score doesn't go lower than 0.5. 
                ax.set_yticks(np.arange(0, 800, 100)) # Max frequency doesn't go higher than 500.
                ax.set_xlabel('AUC Score')
                ax.set_ylabel('Frequency')

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
        #Drop balanced accuracy and AUC from the metrics. 
        test_model_metrics = test_model_metrics.drop([1])
        test_model_metrics = test_model_metrics.drop([6])
        
        test_model_metrics_names = test_model_metrics.iloc[:, 0]
        test_model_means = test_model_metrics.iloc[:, 1]
        test_model_std_devs = test_model_metrics.iloc[:, 2]

        print('Doing 1 Test Model Metrics Visualization')
        
        # Plot the test model metrics with error bars
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(x=test_model_metrics_names, y=test_model_means, yerr=test_model_std_devs, fmt='o')
        ax.set_title(f'{self.architecture} {self.algorithm} Test Model Metrics Means and Standard Deviation as Error Bars')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Mean')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.visualize_output_dir}/{self.architecture}_{self.algorithm}_Test_Model_Metrics.pdf')
        plt.show()
        plt.close()

    def Visualize_ALL_Model_Metrics(self):
        '''
        Visualize the test model performance metrics with error bars for a all architectures and algorithms.
        Metrics: Accuracy, Precision, Recall, F1, and Log Loss
        '''

        print('Doing All Model Metrics Visualization')

        _, ax = plt.subplots(figsize=(12, 15))

        # for each architecture and algorithm, plot the test model metrics with error bars
        for architecture in self.arch:
            for algorithm in self.alg:
                test_model_metrics = pd.read_csv(f'/root/{architecture}/results/{algorithm}/pipeline_run/Outputs/Compiled_Results/{architecture}_{algorithm}_Test_Avg_Calculated_Metrics.csv')
                test_model_metrics.drop([1], inplace=True)
                test_model_metrics.drop([6], inplace=True)
                test_model_metrics_names = test_model_metrics.iloc[:, 0]
                test_model_means = test_model_metrics.iloc[:, 1]
                test_model_std_devs = test_model_metrics.iloc[:, 2]
                
                ax.errorbar(x=test_model_metrics_names, y=test_model_means, yerr=test_model_std_devs, fmt='o', label=f'{architecture} {algorithm}')


        ax.set_title('Test Model Metrics Means and Standard Deviation as Error Bars')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Mean')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('/root/All_Test_Model_Metrics.pdf')
        plt.show()
        plt.close()

    def Visualize_ARCH_Model_Metrics(self):
        '''
        Visualize the test model performance metrics with error bars. Grouping by architecture.
        Metrics: Accuracy, Precision, Recall, F1, and Log Loss
        '''

        print('Doing Model Metrics_Arch Visualization')

        # Create a 2x2 grid of subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        axs = axs.flatten()

        # For each architecture, plot the test model metrics (for each algorithm)
        for i, architecture in enumerate(self.arch):
            for algorithm in self.alg:
                test_model_metrics = pd.read_csv(f'/root/{architecture}/results/{algorithm}/pipeline_run/Outputs/Compiled_Results/{architecture}_{algorithm}_Test_Avg_Calculated_Metrics.csv')        
                test_model_metrics = test_model_metrics.drop([1])
                test_model_metrics = test_model_metrics.drop([6])
                test_model_metrics_names = test_model_metrics.iloc[:, 0]
                test_model_means = test_model_metrics.iloc[:, 1]
                test_model_std_devs = test_model_metrics.iloc[:, 2]
                
                axs[i].errorbar(x=test_model_metrics_names, y=test_model_means, yerr=test_model_std_devs, fmt='o', label=f'{architecture} {algorithm}')

                axs[i].set_title(f'{architecture} Test Model Metrics')
                axs[i].set_xlabel('Metric')
                axs[i].set_ylabel('Mean')
                axs[i].set_ylim(0, 1)
                axs[i].tick_params(axis='x', rotation=45)
            
            # Create a legend for each subplot
            handles, labels = axs[i].get_legend_handles_labels()
            by_label = {label.split()[1]: handle for handle, label in zip(handles, labels)}
            axs[i].legend(by_label.values(), by_label.keys(), loc='lower right')

        plt.tight_layout()
        plt.savefig('/root/Test_Model_Metrics_Arch.pdf')
        plt.show()
        plt.close()

    def Visualize_ALG_Model_Metrics(self):
        '''
        Visualize the test model performance metrics with error bars. Grouping by algorithm.
        Metrics: Accuracy, Precision, Recall, F1, and Log Loss
        '''
        # #FIXME: Not well displayed
        # Commented out  as precaution.

        # print('Doing Model Metrics_Alg Visualization')
        sns.set_context('poster')

        # fig, axs = plt.subplots(1, len(self.alg), figsize=(12, 6))
        

        # for i, algorithm in enumerate(self.alg):
        #     all_metrics = []
            
        #     for architecture in self.arch:
        #         # Load test model metrics
        #         test_model_metrics = pd.read_csv(f'/root/{architecture}/results/{algorithm}/pipeline_run/Outputs/Compiled_Results/{architecture}_{algorithm}_Test_Avg_Calculated_Metrics.csv')        
        #         test_model_metrics = test_model_metrics.drop([1])  # Drop rows if necessary
        #         test_model_metrics = test_model_metrics.drop([6])

        #         test_model_metrics_names = test_model_metrics.iloc[:, 0]  # Metric names
        #         test_model_means = test_model_metrics.iloc[:, 1]  # Mean values
        #         test_model_std_devs = test_model_metrics.iloc[:, 2]  # Std values

        #         for name, mean, std in zip(test_model_metrics_names, test_model_means, test_model_std_devs):
        #             all_metrics.append([architecture, name, mean, std])

        #     # Convert to DataFrame
        #     metrics_df = pd.DataFrame(all_metrics, columns=['Architecture', 'Metric', 'Mean', 'Std'])

        #     # Get unique metrics
        #     metrics = metrics_df['Metric'].unique()
            
        #     # Prepare data for `matplotlib`'s `ax.boxplot`
        #     box_data = []
        #     positions = []
        #     labels = []
            
        #     for j, metric in enumerate(metrics):
        #         for k, architecture in enumerate(self.arch):
        #             subset = metrics_df[(metrics_df['Metric'] == metric) & (metrics_df['Architecture'] == architecture)]
                    
        #             if not subset.empty:
        #                 mean = subset['Mean'].values[0]
        #                 std = subset['Std'].values[0]
                        
        #                 # Define box plot stats
        #                 Q1 = mean - 0.5 * std  # Approximate Q1
        #                 Q3 = mean + 0.5 * std  # Approximate Q3
        #                 whisker_low = mean - std
        #                 whisker_high = mean + std
        #                 median = mean  # Median is the mean itself

        #                 box_data.append([whisker_low, Q1, median, Q3, whisker_high])
        #                 positions.append(j + k * 0.2)  # Space architectures apart
        #                 labels.append(f"{metric}\n{architecture}")

        #     # Create box plots for each metric score (ie. architecture) per algorithm
        #     colors = ['blue', 'green', 'purple', 'yellow']
        #     for k, architecture in enumerate(self.arch):
        #         arch_positions = [pos + k * 0.2 for idx, pos in enumerate(positions) if idx % len(self.arch) == k]
        #         axs[i].boxplot(
        #             [box_data[m] for m in range(len(box_data)) if m % len(self.arch) == k],
        #             positions=arch_positions,
        #             widths=0.2,
        #             patch_artist=True,
        #             boxprops=dict(facecolor=colors[k])
        #         )

        #     # Create a custom legend
        #     legend_handles = [plt.Line2D([0], [0], color='w', marker='s', markersize=10, markerfacecolor=color, label=arch) for color, arch in zip(colors, self.arch)]
        #     axs[i].legend(handles=legend_handles, title='Architecture', loc='upper right')
            
        #     axs[i].set_xticks(range(len(metrics)))
        #     axs[i].set_xticklabels(metrics, rotation=45)
        #     axs[i].set_title(f'{algorithm} Test Model Metrics (Mean ± Std Dev)')
        #     axs[i].set_ylim(0, 1)
        #     axs[i].set_ylabel("Value")

        # plt.tight_layout()
        # plt.savefig('/root/All_Test_Model_Metrics_Alg.pdf')
        # plt.show()
        # plt.close()

        # def Visualize_ALG_Model_Metrics(self):

        print('Doing Model Metrics_Alg Visualization')

        # Create a 3x1 grid of subplots
        fig, axs = plt.subplots(1, 3, figsize=(25, 8))
        axs = axs.flatten()

        # For each algorithm, plot the test model metrics (for each architecture)
        for i, algorithm in enumerate(self.alg):
            for architecture in self.arch:
                test_model_metrics = pd.read_csv(f'/root/{architecture}/results/{algorithm}/pipeline_run/Outputs/Compiled_Results/{architecture}_{algorithm}_Test_Avg_Calculated_Metrics.csv')        
                test_model_metrics = test_model_metrics.drop([1])
                test_model_metrics = test_model_metrics.drop([6])

                test_model_metrics_names = test_model_metrics.iloc[:, 0]
                test_model_means = test_model_metrics.iloc[:, 1]
                test_model_std_devs = test_model_metrics.iloc[:, 2]
                
                axs[i].errorbar(x=test_model_metrics_names, y=test_model_means, yerr=test_model_std_devs, fmt='o', markersize=8, label=f'{architecture} {algorithm}')

                axs[i].set_title(f'{algorithm} Test Model Metrics')
                axs[i].set_xlabel('Metric')
                axs[i].set_ylabel('Mean')
                axs[i].set_ylim(0, 1)
                axs[i].tick_params(axis='x', rotation=45)
            
            # Create a legend for each subplot
            handles, labels = axs[i].get_legend_handles_labels()
            by_label = {label.split()[0]: handle for handle, label in zip(handles, labels)}
            axs[i].legend(by_label.values(), by_label.keys(), loc='upper right', title='Architecture', fontsize='14', title_fontsize='16')

            axs[i].set_yticks(np.arange(0, 1.0, 0.1))


        plt.tight_layout()
        plt.savefig('/root/All_Test_Model_Metrics_Alg.pdf')
        plt.show()
        plt.close()

    def Visualize_Importances_histo_error(self):
        '''
        Visualize the feature importances with error bars in a histogram.
        Bars are colored by selected feature.
        '''

        print('Doing Model Feature Importances (A) Visualization')
        
        # Create a histogram of the mean feature importances with error bars
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.Blues(np.linspace(0.3, 1, len(self.feature_names)))
        bars = ax.bar(range(len(self.feature_names)), self.feature_importances, yerr=self.feature_importance_std_devs, color=colors, ecolor='black', capsize=5)
        ax.set_title(f'{self.architecture} {self.algorithm} Feature Importances with Mean and Standard Deviation as Error Bars')
        ax.set_xlabel('Selected Feature')
        ax.set_ylabel('Mean Importance')
        plt.xticks(visible=False)
        plt.tight_layout()

        # Create a continuous color bar legend
        sm = plt.cm.ScalarMappable(cmap='Blues_r', norm=plt.Normalize(vmin=0, vmax=len(self.feature_names)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_ticks(np.linspace(0.5, len(self.feature_names) - 0.5, len(self.feature_names)))
        
        # Add tally count of each feature
        cbar.set_ticklabels([f'{name} ({count})' for name, count in zip(self.feature_names[::-1], self.tally_count[::-1])])
        cbar.set_label('Selected Features')

        plt.savefig(f'{self.visualize_output_dir}/{self.architecture}_{self.algorithm}_Feature_Importances_Histo.pdf')
        plt.show()
        plt.close()

    def Visualize_Importances_Forest(self):
        '''
        Visualize the feature importances with error bars in a forest plot.
        '''
        
        print('Doing Model Feature Importances (B) Visualization')

        # Create a bar plot of the mean feature importances with error bars
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(range(len(self.feature_names)), self.feature_importances[::-1], xerr=self.feature_importance_std_devs[::-1], color='skyblue', ecolor='black', capsize=5)
        ax.set_yticks(range(len(self.feature_names)))
        
        # Add tally count of each feature
        ax.set_yticklabels([f'{name} ({count})' for name, count in zip(self.feature_names[::-1], self.tally_count[::-1])])
        
        ax.set_title(f'{self.architecture} {self.algorithm} Feature Importances with Mean and Standard Deviation as Error Bars')
        ax.set_xlabel('Mean Importance')
        ax.set_ylabel('Selected Feature')
        plt.tight_layout()
        plt.savefig(f'{self.visualize_output_dir}/{self.architecture}_{self.algorithm}_Feature_Importances.pdf')
        plt.show()
        plt.close()

    def Visualize_LR_OR(self):
        '''
        Create a forest plot of the odds ratios with 95% confidence intervals for logistic regression.
        '''

        print('Doing Logistic Regression Odds Ratio Visualization')

        # Load the Logistic Regression Odds Ratio & Confidence Interval data
        LR_data = pd.read_csv(f'{self.compiled_folder}/{self.architecture}_{self.algorithm}_LR_Feature_Vals_Metrics_With_Tally.csv')

        feature_name = LR_data['Feature']
        mean_OR = LR_data['Mean_Odds Ratio']
        mean_upper_CI_OR = LR_data['Mean_CI Upper OR']
        mean_lower_CI_OR = LR_data['Mean_CI Lower OR']
        lr_tally_count = LR_data['Tally']

        # Append forest plot directly to PDF
        with PdfPages(f'{self.visualize_output_dir}/{self.architecture}_{self.algorithm}_Forest_Plot.pdf') as pdf:
            fig, ax = plt.subplots(figsize=(12, 8))
            # values reversed because plot naturally ordered with first index at bottom. First index in self.tally_count is the highest tally.
            ax.errorbar(mean_OR[::-1], [f'{name} ({count})' for name, count in zip(feature_name[::-1], lr_tally_count[::-1])], 
                xerr=(mean_OR[::-1] - mean_lower_CI_OR[::-1], mean_upper_CI_OR[::-1] - mean_OR[::-1]), 
                fmt='o', color='black', ecolor='black', capsize=5)
            ax.axvline(x=1.0, color='black', linestyle='--')
            ax.set_title(f'{self.architecture} {self.algorithm} Forest Plot of Odds Ratios with 95% Confidence Intervals')
            ax.set_ylabel('Selected Feature')
            ax.set_xlabel('Mean Odds Ratio')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Add text annotations for OR (lowerCI - upperCI) in line with each feature name
            for i, (or_val, lower_ci, upper_ci) in enumerate(zip(mean_OR[::-1], mean_lower_CI_OR[::-1], mean_upper_CI_OR[::-1])):
                ax.text(ax.get_xlim()[1] + 0.3, i, f'{or_val:.2f} ({lower_ci:.2f} - {upper_ci:.2f})', va='center', ha='right', color='black')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.show()
            plt.close()

        # Create a DataFrame of the feature names and odds ratios with 95% confidence intervals
        Lr_Or_table = pd.DataFrame({
            'Feature': feature_name[::-1],
            'Odds Ratio (95% CI)': [f'{or_val:.2f} ({lower_ci:.2f} - {upper_ci:.2f})' for or_val, lower_ci, upper_ci in zip(mean_OR[::-1], mean_lower_CI_OR[::-1], mean_upper_CI_OR[::-1])]
        })

        # Save the DataFrame to a CSV file
        Lr_Or_table.to_csv(f'{self.visualize_output_dir}/{self.architecture}_{self.algorithm}_OR_TABLE.csv', index=False)


    def All_Hist_AUC2(self):
        """
        Visualize the AUC scores (histogram matrix) for each architecture and algorithm in a 4x3 grid of subplots.
        FIXME: Change smooth line to blue
        """

        print('Doing AUC Sub Matrix Visualization')

        # Set the plotting style to 'poster'
        sns.set_style('white')
        sns.set_context('poster')

        # Create a 4x3 grid of subplots
        fig, axs = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(15, 20))
        fig.suptitle('Frequency Distribution of AUC Scores Across Algorithms and Architectures')
        axs = axs.flatten()

        # Create equally spaced bins for the AUC scores
        bin_start = 0.5  # Minimum AUC score
        bin_end = 1.0    # Maximum AUC score
        bin_width = 0.02 # Fixed bin width
        bins = np.arange(bin_start, bin_end + bin_width, bin_width)  # Create array of bin edges

        # enumerate through the architectures and algorithms
        for i, architecture in enumerate(self.arch):

            # for each architecture, iterate through the algorithms
            for j, algorithm in enumerate(self.alg):

                # get the AUC scores for each run
                AUC_scores = []
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
                    AUC_scores.append(auc_score)

                # plot the histogram of the AUC scores by architecture and algorithm
                ax = axs[i * 3 + j]   

                # Calculate the mean and standard deviation of the AUC scores for each architecture and algorithm
                mean_auc = np.mean(AUC_scores)
                std_auc = np.std(AUC_scores)

                # Use a vertical line to indicate the mean with dashed line
                ax.axvline(mean_auc, color='b', linestyle='dashed', linewidth=1)

                # Add text to the plot to state the mean and standard deviation of the AUC scores
                ax.text(0.95, 0.95, f'Mean: {mean_auc:.2f}\n Std: {std_auc:.2f}', color='black', va='top', ha='right', transform=ax.transAxes, fontsize=15)

                # Plot the histogram of the AUC scores using seaborn
                sns.histplot(AUC_scores, bins=bins, ax=ax, kde=True, alpha=0.7, color='black')
                ax.set_ylim(0, max(np.histogram(AUC_scores, bins=bins)[0]) + 50)

                # # Calculate the maximum value of the kde plot
                # kde = sns.kdeplot(AUC_scores, ax=ax, color='blue')
                # kde_max = kde.get_lines()[-1].get_data()[1].max()

                # # Set the y-axis limit to the maximum of the histogram, kde plot, or 800 plus some padding
                # ax.set_ylim(0, min(max(np.histogram(AUC_scores, bins=bins)[0].max(), kde_max) + 50, 800))

                if architecture == 'fast_ml':
                    arch = 'Fast ML'
                elif architecture == 'gc':
                    arch = 'GC'
                elif architecture == 'ctgan':
                    arch = 'CTGAN'
                elif architecture == 'dp_cgan':
                    arch = 'DP-CGAN'
                
                if algorithm == 'log_reg':
                    alg = 'LR'
                elif algorithm == 'rf':
                    alg = 'RF'
                elif algorithm == 'mlp':
                    alg = 'MLP'

                # Plot the histogram of the AUC scores
                ax.set_title(f'{arch}-{alg}') 
                xticks = np.arange(0.5, 1.0, 0.1)
                xticks = np.append(xticks, 1.0)
                ax.set_xticks(xticks)

                ax.tick_params(axis='x', rotation=45)
                ax.set_yticks(np.arange(0, 800, 100)) 
                ax.set_ylabel('Frequency', visible=False)
        fig.supxlabel('AUC Score')
        fig.supylabel('Frequency')

        plt.tight_layout()
        plt.savefig('/root/AUC_Scores_All_Datasets_Subplots_NEW.pdf')
        plt.show()
        plt.close()

        

if __name__ == '__main__':

    algorithm = 'mlp'
    architecture = 'ctgan'
    
    vis = Visualize_Compare_Metrics(architecture, algorithm)
    # vis.Visualize_One_Test_Model_Metrics()
    # vis.Visualize_Importances_histo_error()
    # vis.Visualize_Importances_Forest()
    # if algorithm == 'log_reg':
    #     vis.Visualize_LR_OR()

    ## Only to run after all comparisons from compare_metrics_features done
    # vis.Visualize_AUC_Num_Matrix()
    # vis.Visualize_ACCURACY_Num_Matrix()
    # vis.Visualize_AUC_Sub_Matrix()
    # vis.Visualize_ALL_Model_Metrics()
    # vis.Visualize_ARCH_Model_Metrics()
    # vis.Visualize_ALG_Model_Metrics()
    # vis.All_Hist_AUC2()

    # print('Visualizations done')

