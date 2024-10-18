import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, log_loss, classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import os
import reportlab
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
model_name = input("Please enter the model name: ") #FIXME: not modelname anmyore
# model_file_path = os.path.join(base_path, target_folder_name, f"{model_name}.csv")
model_file_path = input("Please enter the model file path: ")


df_og = pd.read_csv(model_file_path, delimiter=',', encoding='latin1')
df = pd.DataFrame(df_og.copy())
df.drop(columns='leeft', inplace=True)

path_parts = model_file_path.split(os.sep) # Split the file path by the directory separator
file_name = path_parts[-1] # Extract the second to last part of the split path


def Data_Exploration(df):
    print("\n Visualize original data before processing \n")
    print(df.head())
    print(df.info())
    print(df.describe())

def Rename_Col(df):

    global X, y

    columns_to_rename = { 
        "ql": "Global Health Status",
        "pf": "Physical Functioning",
        "sf": "Social Functioning",
        "ef": "Emotional Functioning",
        "cf": "Cognitive Functioning",
        "rf": "Role Functioning",
        "maladaptive": "Maladaptive Coping Style",
        "nsurv94dich": "Sexual Attractiveness", # self-perception of own sexual attractiveness
        "Targetnew": "Targeted Therapy",
        "Hormnew": "Hormone Therapy",
        "RTnew": "Radiation Therapy",
        "chemonew": "Chemotherapy",
        "partner": "Partner",
        "alg_v1b": "Biological Sex",
        "stadium_opgeschoond": "Tumour Stage",
        "tumsoort_hoofd_new2": "Tumour Type",
        "edu": "Education Level",
        "tijddiagngroep": "Time Since Diagnosis",
        "BMIgroups2": "BMI Group",
        "BIneg_new": "Negative Body Image"
    }

    df.rename(columns=columns_to_rename, inplace=True)
    X = df.drop(columns=["Negative Body Image"])
    y = df["Negative Body Image"]
    print(df.info())

    return df




def Cat_Encoder():

    """

    Function to transform categorical columns using a OneHotEncoder.

    1. Uses a dictionary (refs) to indicate categorical columns and their reference values to drop using a OneHotEncoder.
    2. Transforms and refits the model with the encoded data.


    Global Variables:
    - refs (dict): A dictionary containing categorical columns and their reference values to drop.
    - df_encoded (pd.DataFrame): The (new) DataFrame containing the encoded data.
    Returns:
    - df_encoded (pd.DataFrame): The DataFrame with categorical columns encoded and other columns unchanged.
   
    # Binary features are not encoded.

    """

    global refs, df_encoded, X

    refs = {
        "Education Level": 3.0, #FIXME: Given data, no 2.0.
        "Tumour Type": 1.0,
        "Tumour Stage": 1.0,
        "Time Since Diagnosis": 3.0,
        "BMI Group": 1.0,
    }

    # Extract categorical columns and their reference values to drop
    cat_columns = list(refs.keys()) # Get the categorical columns
    refs_to_drop = [refs[col] for col in cat_columns] # Get the reference values to drop  
    print("DROP SPECS:", refs_to_drop)
    
    # Create a OneHotEncoder with drop specifications
    ohe = OneHotEncoder(drop=refs_to_drop, sparse_output=False, handle_unknown='ignore')
    
    # Apply the encoder to the specified categorical columns
    column_transformer = ColumnTransformer(
        transformers=[
            ('ohe', ohe, cat_columns)  
        ], remainder='passthrough'  # Keep other columns unchanged
    )
    
    # Fit and transform the DataFrame
    df_transformed = column_transformer.fit_transform(df)
    print("DF (X) TRANSFORMED:", df_transformed)
    
    # Get the transformed column names
    ohe_feature_names = column_transformer.named_transformers_['ohe'].get_feature_names_out(cat_columns)
    print("FT NAMES: ", ohe_feature_names)
    
    # Convert the result back to a DataFrame
    df_encoded = pd.DataFrame(df_transformed, columns=np.append(ohe_feature_names, df.drop(columns=cat_columns).columns))
    print("\n Encoded df:", df_encoded.info())

    return df_encoded


# Cat_Encoder()

#TODO: PVALUE, ERROR


def Ttv_Split():
    """

    Train Test Val (Ttv) Split Function
    
    1. Split the Data into training, validation, and test sets. 
        Split:, 80% training, 20% validation, and 10% test.
    2. Scale the data.
    3. Put data into dataframe type.

    """

    global X, y, X_train, X_test, X_val, y_val, y_train, y_test

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    print(f'\n X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, \n X_test shape: {X_test.shape}, y_test shape: {y_test.shape}, \n X_val shape: {X_val.shape}, y_val shape: {y_val.shape}')

    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_val = pd.DataFrame(X_val, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)

    return  X_train, X_test, X_val, y_val, y_train, y_test


def Create_Model():

    """

    Function to

        1. Initialize logistic regression model with elastic net regularization: log_reg_e
            The l1_ratio parameter controls the mix of L1 and L2 regularization: 0.5   
            C controls the sparsity: the smaller C the fewer features selected: 0.1
        2. Resample the training data using SMOTE to handle class imbalance.
        3. Fits the model on the training data.
    
    """

    global X_train, y_train, log_reg_e

    log_reg_e = LogisticRegression(penalty='elasticnet', solver ='saga', l1_ratio=0.5, C = 0.1)
    
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    
    log_reg_e.fit(X_train, y_train)

    return log_reg_e


def Feature_Selection():

    """

    Function to select features for the model.

    Inputs
        1. SelectFromModel: Selects features based on the fitted model.
            -  Threshold parameter: If None and if the estimator has a parameter penalty set to l1, 
                    either explicitly or implicitly (e.g, Lasso), the threshold used is 1e-5.
            #TODO: Implement?: max_features int, callable, default=None.
                If an integer, then it specifies the maximum number of features to allow. 
                If a calable, then it specifies how to calculate the maximum number of features allowed by using the output of max_features(X).
                To only select based on max_features, set threshold=-np.inf.
        2. selected_features_mask: A mask indicating which features were selected.
        3. selected_features: The names of the selected features from mask
        4. Transforms the training and validation datasets to `X_train_ft` and `X_val_ft`) to include only the selected features.
    
    """

    global X_train, X_val, X_val_ft, log_reg_e, X_train_ft, selected_features
    
    model_log_features = SelectFromModel(log_reg_e, prefit=True)
    selected_features_mask = model_log_features.get_support()
    selected_features = X_train.columns[selected_features_mask]

    X_train_ft = pd.DataFrame(model_log_features.transform(X_train), columns=selected_features)
    X_val_ft = pd.DataFrame(model_log_features.transform(X_val), columns=selected_features)

    return X_train, X_val, X_val_ft, log_reg_e, X_train_ft, selected_features


def Parameter_Tuning():

    """

    Function to tune hyperparameters and refit model on selected features.
 
    1. Computes class weights to handle class imbalance.
    2. Performs a grid search to find the best hyperparameters for the logistic regression model.
    3. Refits the logistic regression model with the best hyperparameters on the selected features.
    4. Makes predictions on the validation data.

    """

    global  y_train, X_train_ft, X_val_ft, log_reg_e, log_reg_e_ft, y_proba_ft, y_pred_ft

    #compute weights for class imbalance based on frequency in training data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    param_grid_elas = {
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],  # Range of l1_ratio values to search
        'C': [0.01, 0.1, 1, 10, 100],  # Range of regularization strengths. Smaller values, stronger regularization. 0.01 resulted in 'no predicted samples' for precision and would have to initiate the zero_division parameter.
            }
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    grid_search = GridSearchCV(log_reg_e, param_grid=param_grid_elas, scoring=scoring_metrics, refit='accuracy', cv=5) 
    grid_search.fit(X_train_ft, y_train) 
    best_params = grid_search.best_params_
    print("Best parameters found: ", best_params)
    print("Best score found: ", grid_search.best_score_)

    # Refit the logistic regression model on X_train_ft using balanced classes and best params.
    log_reg_e_ft = LogisticRegression(class_weight=class_weight_dict, **best_params).fit(X_train_ft, y_train)
    print('Class weight', class_weight_dict)

    # Predictions
    y_pred_ft = log_reg_e_ft.predict(X_val_ft)
    y_proba_ft = log_reg_e_ft.predict_proba(X_val_ft)

    return y_train, X_train_ft, X_val_ft, log_reg_e, log_reg_e_ft, y_proba_ft, y_pred_ft


def Feature_Importance():

    """ 
    Fucntion to:

    1. Compute feature importance using coefficients, and odds ratios (of selected features).
    2. Calculate standard errors and confidence intervals for coefficients.
    3. Create a DataFrame of features, coefficients, and odds ratios.
    4. Plot the odds ratios with 95% confidence intervals.

    """

    global log_reg_e_ft, selected_features, X_train_ft, coef, odds_ratios, ci_lower_or, ci_upper_or, feature_importance, or_ci_plot, model_name, date_time, feature_importance_df

    feature_importance = pd.Series(log_reg_e_ft.coef_.flatten(), index=selected_features)
    feature_importance = feature_importance.round(3)

    coef = log_reg_e_ft.coef_.flatten()
    coef = np.sort(coef, axis=None)[::-1] # to sort df by coef
    coef = np.round(coef, 3)

    # Calculate standard errors
    se = np.sqrt(np.diag(np.linalg.inv(np.dot(X_train_ft.T, X_train_ft))))

    # Calculate CIs for coefficients
    z = 1.96  # for 95% confidence interval
    ci_lower = coef - z * se
    ci_upper = coef + z * se

    # Convert coefficients to CIs
    ci_lower_or = np.exp(ci_lower)
    ci_upper_or = np.exp(ci_upper)
    odds_ratios = np.exp(coef)
    odds_ratios = np.round(odds_ratios, 3)

    # Create a DataFrame of features, coefficients, and odds ratios
    feature_importance_df = pd.DataFrame({
        'Feature': feature_importance.index,
        'Importance': feature_importance.values,
        'Coefficient': coef.flatten(),
        'Odds Ratio': odds_ratios.flatten(),
        'CI Lower': ci_lower_or,
        'CI Upper': ci_upper_or
    })

    ft_impt_file_path = os.path.join(target_folder, f'{model_name}_Feature_Importance_Table_{date_time}.csv')
    feature_importance_df.to_csv(ft_impt_file_path, index=False)
    
    print('\n', feature_importance_df)
    
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

    plt.show()
    plt.close()

    return coef, ci_lower_or, ci_upper_or, odds_ratios, feature_importance, feature_importance_df, or_ci_plot


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

    global metrics_ft, conf_matrix, correlation_matrix, auc, fpr, tpr, confusion_matrix_plot, correlation_matrix_plot, roc_curve_plot, model_name, date_time

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

    # Print metrics
    for metric, value in metrics_ft:
        print(f'{metric}: {value}')

    return conf_matrix, correlation_matrix, auc, metrics_ft 


def save_outputs():
    '''
    
    Function to save the output of models to a PDF file.
    
    1. Get data and time.
    2. Split the file path by the directory separator; split on folder & file name.
    3. Add text data (evaluations) to PDF in tabular format for selected features of model.  
    4. Add image data to PDF: confusion matrix, correlation matrix, ROC curve.  
    4. Save the PDF to a specific folder.

    '''
    global date_time, file_name, metrics_ft, coef, odds_ratios, ci_lower_or, ci_upper_or, feature_importance, selected_features, confusion_matrix_plot, correlation_matrix_plot, roc_curve_plot, or_ci_plot, img_txt_output_pdf, text_output_pdf, feature_importance_df

    # Text data
    data_text = {
        'Model': file_name,
        'Date': date_time,
        'Accuracy': metrics_ft[0][1],
        'Balanced Accuracy': metrics_ft[1][1],
        'Precision': metrics_ft[2][1],
        'Recall': metrics_ft[3][1],
        'F1 Score': metrics_ft[4][1],
        'Log Loss': metrics_ft[5][1],
        'ROC AUC': metrics_ft[6][1],

        # 'Coefficients': coef.tolist(),
        # 'Odds Ratios': odds_ratios.tolist(),
        # 'Confidence Intervals': (ci_lower_or.tolist(), ci_upper_or.tolist()),
        # 'Feature Importance': feature_importance.tolist(),
        # 'Selected Features': selected_features.tolist()
    }

    # Specify the file name for the new PDF
    text_output_pdf = os.path.join(target_folder, f'{model_name}_Text_Output_{date_time}.pdf')

    # Create a canvas for the PDF, specifying the file name and page size
    c = canvas.Canvas(text_output_pdf, pagesize=A4)
    width, height = A4 # Get the width and height of the page

    # Set 1-inch margins (72 points)
    margin = 72

    # Usable width and height after applying 1-inch margins
    usable_width = width - 2 * margin
    usable_height = height - 2 * margin

    # Put data_text into a table in the PDF
    # data = [[key, str(value)] for key, value in data_text.items()]
    data = [[feature, str(coef), str(odds_ratio), str(importance), str(ci_lower), str(ci_upper)]
    for feature, coef, odds_ratio, importance, ci_lower, ci_upper in zip(
        selected_features, coef, odds_ratios, feature_importance, ci_lower_or, ci_upper_or
    )]
    t = Table(data, colWidths=[100, 400])  # Adjust column widths as needed
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Courier-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    # Wrap the table within the usable area and draw it on the canvas
    t.wrapOn(c, usable_width, usable_height)
    t.drawOn(c, margin, height - margin - t._height)  # Positioning the table within the margins

    # Finalize the PDF file
    c.showPage()  # Finish the current page
    c.save()  # Save the PDF

    pdf_files = [
        text_output_pdf,
        confusion_matrix_plot,
        correlation_matrix_plot,
        roc_curve_plot,
        or_ci_plot
    ]

    # Create a PdfMerger object
    merger = PdfMerger()

    # Append each file to the merger if it exists
    for pdf_file in pdf_files:
        merger.append(f"{pdf_file}")
    

    # Write the merged PDF to a new file
    img_txt_output_pdf = os.path.join(target_folder, f'{model_name}_merged_output_{date_time}.pdf')
    with open(img_txt_output_pdf, "wb") as merged_pdf: 
        merger.write(merged_pdf)
    merger.close()

    print(f"Merged PDF created as '{img_txt_output_pdf}'.")



def Preprocessing(explore, rename, split, log_eval, save):

    """

    Function to preprocess the data and then evaluate the model on feature selection.

    - explore: bool - whether to explore the data (head, info, describe).
    - rename: bool - whether to rename the columns of the dataset.
    - split: bool - whether to split the data after categorical encoding.
    - log_eval: bool - whether to create, perform feature selection
        parameter tuning, feature importance calculations
        and evaluate the model.
    - save: bool - whether to save the outputs to a PDF file.
    
    """
    if explore == True:
        Data_Exploration(df)
    if rename == True:
        Rename_Col(df)
    if split == True:
        Cat_Encoder()
        Ttv_Split()
    if log_eval == True:
        Create_Model()
        Feature_Selection()
        Parameter_Tuning()
        Feature_Importance()
        Evaluations(y_val, y_pred_ft, y_proba_ft, X_train_ft)
    if save == True:
        save_outputs()

if __name__ == "__main__":
    Preprocessing(explore=True, rename=True, split=True, log_eval=True, save=True)


    
    # To add SMOTE if used, P-Value, Error, model parameters.


    # # Specify the file name for the new PDF
    # text_output_pdf = "text_output_pdf.pdf"

    # # Create a canvas for the PDF, specifying the file name and page size
    # c = canvas.Canvas(text_output_pdf, pagesize=A4)
    # width, height = A4 # Get the width and height of the page

    # # # Adding some text to the PDF for demonstration
    # # c.drawString(100, height - 100, "This is the Model Attributes PDF")



    # # Save the PDF
    # c.save()

    # print(f"New PDF created as '{text_output_pdf}'.")

    # List of PDF files to merge (ensure they have the .pdf extension)