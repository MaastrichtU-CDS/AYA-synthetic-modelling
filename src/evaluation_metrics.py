import logging
import os 

from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, log_loss, classification_report

#TODO: PVALUE, ERROR

def Evaluations(y_val, y_pred_ft, y_proba_ft, X_train_ft):

    """ 

    Function to evaluate the model fitted on selected features.

    Input:
    - y_val (pd.Series): The validation target variable.
    - y_pred_ft (np.ndarray): The predicted labels for the validation set.
    - X_train_ft (pd.DataFrame): The transformed training feature set with selected features.
    - y_proba_ft (np.ndarray): The predicted probabilities for the validation set.
    
    Output: Evaluates the model using various metrics:
    - Confusion Matrix: compare actual vs predicted values.
    - Correlation matrix: Correlation coefficient between each pair of variables. Pearson's r: -1 to 1. 0 = no correlation. Linear relationship.
    - ROC Curve and AUC: AUC = 0.5 (random) to 1 (perfect).
        - ROC AUC Score: The area under the ROC curve.
    Metrics:
    - Balanced accuracy: It is defined as the average of recall obtained on each class.
    - Accuracy Score: The proportion of true results among the total number of cases examined.
    - Precision Score: The proportion of true positive results among the positive results.
    - Recall Score: The proportion of true positive results among the true positive and false negative results.
    - F1 Score: The weighted average of precision and recall
    - Log Loss: The loss function for logistic regression.

    """

    global metrics_ft, conf_matrix, correlation_matrix, auc, fpr, tpr, confusion_matrix_plot, correlation_matrix_plot, roc_curve_plot, model_name, date_time, coef, ci_lower_or, ci_upper_or, odds_ratios, feature_importance, feature_importance_df, or_ci_plot

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_val, y_pred_ft)
    plt.figure(figsize=(10, 10)) 
    sns.set(font_scale=1.3)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix ({model_name})', fontsize=13)
    
    plt.savefig(os.path.join(target_folder, f'{model_name}_Confusion_Matrix_Plot_{date_time}.pdf'), format='pdf') # Save the figure to a file
    confusion_matrix_plot = os.path.join(target_folder, f'{model_name}_Confusion_Matrix_Plot_{date_time}.pdf')

    plt.show()
    plt.close()

    # Correlation matrix
    correlation_matrix = X_train_ft.corr(method="spearman")
    plt.figure(figsize=(15, 15))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    sns.heatmap(correlation_matrix, annot=True, cmap="RdYlGn", linewidths=.5)
    plt.title(f"Correlation Matrix Heatmap of All Variables ({model_name})", fontsize=25)

    plt.savefig(os.path.join(target_folder, f'{model_name}_Correlation_Matrix_Plot_{date_time}.pdf'), format='pdf') # Save the figure to a file
    correlation_matrix_plot = os.path.join(target_folder, f'{model_name}_Correlation_Matrix_Plot_{date_time}.pdf')

    plt.show()
    plt.close()

    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y_val, y_proba_ft[:, 1])
    auc = round(roc_auc_score(y_val, y_proba_ft[:, 1]), 3)
    plt.figure(figsize=(15, 15))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.tight_layout()
    plt.title(f'ROC Curve ({model_name})', fontsize=20)
    plt.legend(loc='best')
    
    plt.savefig(os.path.join(target_folder, f'{model_name}_ROC_Curve_Plot_{date_time}.pdf'), format='pdf') # Save the figure to a file
    roc_curve_plot = os.path.join(target_folder, f'{model_name}_ROC_Curve_Plot_{date_time}.pdf')

    plt.show()
    plt.close()

    # Plot OR & CI
    plt.figure(figsize=(10, 10))
    plt.errorbar(feature_importance_df['Odds Ratio'], feature_importance_df['Feature'],
                xerr=[feature_importance_df['Odds Ratio'] - feature_importance_df['CI Lower'],
                    feature_importance_df['CI Upper'] - feature_importance_df['Odds Ratio']],
                fmt='o', ecolor='r', capsize=5, capthick=2)
    plt.axvline(x=1, color='gray', linestyle='--')
    plt.yticks(rotation=0, fontsize=11)
    plt.ylabel('Features', fontsize=11)
    plt.xlabel('Odds Ratio', fontsize=11)
    plt.title(f'Odds Ratios with 95% Confidence Intervals ({model_name})', fontsize=13)
    plt.tight_layout()

    plt.savefig(os.path.join(target_folder, f'{model_name}_OR_CI_Plot_{date_time}.pdf'), format='pdf') # Save the figure to a file
    or_ci_plot = os.path.join(target_folder, f'{model_name}_OR_CI_Plot_{date_time}.pdf')


    # Print metrics
    for metric, value in metrics_ft:
        print(f'{metric}: {value}')

    # Metrics
    metrics_ft = [
        ['Accuracy Score', round(accuracy_score(y_val, y_pred_ft), 3)],
        ['Balanced Accuracy Score', round(balanced_accuracy_score(y_val, y_pred_ft), 3)],
        ['Precision Score', round(precision_score(y_val, y_pred_ft), 3)],
        ['Recall Score', round(recall_score(y_val, y_pred_ft), 3)],
        ['F1 Score', round(f1_score(y_val, y_pred_ft), 3)],
        ['Log Loss', round(log_loss(y_val, y_proba_ft), 3)],
        ['ROC AUC Score', round(roc_auc_score(y_val, y_proba_ft[:, 1]), 3)]
    ]

    return conf_matrix, correlation_matrix, auc, metrics_ft 