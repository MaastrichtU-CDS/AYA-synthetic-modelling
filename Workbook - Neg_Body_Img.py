import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, log_loss, classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


"""
Global Variables:
    - X_train (pd.DataFrame): The training feature set.
    - y_train (pd.Series): The training target variable.
    - X_val (pd.DataFrame): The validation feature set.
    - y_val (pd.Series): The validation target variable.
    - y_pred_ft (np.ndarray): The predicted labels for the validation set.
    - log_reg_e (LogisticRegression): The initial logistic regression model.
    - X_train_ft (pd.DataFrame): The transformed training feature set with selected features.
    - X_val_ft (pd.DataFrame): The transformed validation feature set with selected features.
    - y_proba_ft (np.ndarray): The predicted probabilities for the validation set.
    - log_reg_e_ft (LogisticRegression): The refitted logistic regression model with the best hyperparameters.

"""

file_path = input("Please enter the file path: ")
df_og = pd.read_csv(file_path, delimiter=',', encoding='latin1')
df = df_og.copy()

# Data Exploration Function
def Data_Exploration(df):
    print("\n DF data here only pertains to original data \n")
    print(df.head())
    print(df.info())
    print(df.describe())

# Data_Exploration(df)

def Rename_Col(df, column_name, new_column_name): # can make into a dictionary so can do multiple columns at once
    df.rename(columns = {column_name:new_column_name}, inplace = True)
    return df

def Cat_Encoder():

    global refs, df_encoded

    refs = {
        "edu": 3.0,
        "tumsoort_hoofd_new2": 1.0,
        "stadium_opgeschoond": 1.0,
        "tijddiagngroep": 3.0,
        "BMIgroups2": 1.0
    }

    # Extract categorical columns and their reference values to drop
    cat_columns = list(refs.keys())
    drop_specs = [refs[col] for col in cat_columns]
    
    # Create a OneHotEncoder with drop specifications
    ohe = OneHotEncoder(drop=drop_specs, sparse_output=False, handle_unknown='ignore')
    
    # Apply the encoder to the specified categorical columns
    column_transformer = ColumnTransformer(
        transformers=[
            ('ohe', ohe, cat_columns)  # Apply the encoder to the categorical columns
        ], remainder='passthrough'  # Keep other columns unchanged
    )
    
    # Fit and transform the DataFrame
    df_transformed = column_transformer.fit_transform(df)
    
    # Get the transformed column names
    ohe_feature_names = column_transformer.named_transformers_['ohe'].get_feature_names_out(cat_columns)
    
    # Convert the result back to a DataFrame
    df_encoded = pd.DataFrame(df_transformed, columns=np.append(ohe_feature_names, df.drop(columns=cat_columns).columns))
    print(df_encoded.head())

    
    return df_encoded


Cat_Encoder()

target =  input('Target name: ') #'BIneg_new'
X = df_encoded.drop(columns=[target])
y = df_encoded[target]


# Train Test Val Split Function
def Ttv_Split():

    global X, y, X_train, X_test, X_val, y_val, y_train, y_test

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    print(f'\n X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, \n X_test shape: {X_test.shape}, y_test shape: {y_test.shape}, \n X_val shape: {X_val.shape}, y_val shape: {y_val.shape}')

    scaler = StandardScaler() 
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=df_encoded.drop(columns=[target]).columns)
    X_val = pd.DataFrame(X_val, columns=df_encoded.drop(columns=[target]).columns)
    X_test = pd.DataFrame(X_test, columns=df_encoded.drop(columns=[target]).columns)

    return  X_train, X_test, X_val, y_val, y_train, y_test

# Ttv_Split()


def Create_Model(): #grid search, elastic net

    """
        Perform logistic regression with elastic net regularization, feature selection, and hyperparameter tuning.
        This function performs the following steps:
        1. Fits a logistic regression model with elastic net regularization on the training data.
    
    """

    global X_train, y_train, log_reg_e

    log_reg_e = LogisticRegression(penalty='elasticnet', solver ='saga', l1_ratio=0.5, C = 0.1)
    # C controls the sparsity: the smaller C the fewer features selected. 
    # With Lasso, the higher the alpha parameter, the fewer features selected.

    log_reg_e.fit(X_train, y_train)

    return log_reg_e

# Create_Model()



def Feature_Selection():

    """

        2. Selects features based on the fitted model.
    
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
    3. Computes class weights to handle class imbalance.
    4. Performs a grid search to find the best hyperparameters for the logistic regression model.
    5. Refits the logistic regression model with the best hyperparameters on the selected features.
    6. Makes predictions on the validation data.
    7. Computes feature importance, coefficients, and odds ratios.
    """

    global  y_train, X_train_ft, X_val_ft, log_reg_e, log_reg_e_ft, y_proba_ft, y_pred_ft

    # # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    param_grid_elas = {
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],  # Range of l1_ratio values to search
        'C': [0.1, 1, 10, 100],  # Range of regularization strengths. Smaller values, stronger regularization. 0.01 resulted in 'no predicted samples' for precision and would have to initiate the zero_division parameter.
            }
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    grid_search = GridSearchCV(log_reg_e, param_grid=param_grid_elas, scoring=scoring_metrics, refit='accuracy', cv=5) 
    grid_search.fit(X_train_ft, y_train) 
    best_params = grid_search.best_params_
    print("Best parameters found: ", best_params)

    # Refit the logistic regression model on X_train_ft
    log_reg_e_ft = LogisticRegression(class_weight=class_weight_dict, **best_params).fit(X_train_ft, y_train)
    print('Class weight', class_weight_dict)

    #TODO: intentional resampling of pts w/ neg body img. SMOTE. To include in elastic net. aka internal trick to learn more from underrepped group.
    #Separate workbook/pipeline test. 

    # Predictions
    y_pred_ft = log_reg_e_ft.predict(X_val_ft)
    y_proba_ft = log_reg_e_ft.predict_proba(X_val_ft)

    return y_train, X_train_ft, X_val_ft, log_reg_e, log_reg_e_ft, y_proba_ft, y_pred_ft

# Parameter_Tuning()

def Feature_Importance():

    global log_reg_e_ft, selected_features, X_train_ft, coef, odds_ratios, ci_lower_or, ci_upper_or

    feature_importance = pd.Series(log_reg_e_ft.coef_.flatten(), index=selected_features)

    coef = log_reg_e_ft.coef_.flatten()
    coef = np.sort(coef, axis=None)[::-1] # to sort df by coef

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

    # Create a DataFrame of features, coefficients, and odds ratios
    feature_importance_df = pd.DataFrame({
        'Feature': feature_importance.index,
        'Importance': feature_importance.values,
        'Coefficient': coef.flatten(),
        'Odds Ratio': odds_ratios.flatten(),
        'CI Lower': ci_lower_or,
        'CI Upper': ci_upper_or
    })
    print('\n', feature_importance_df)
    
    # Plot OR & CI
    plt.figure(figsize=(12, 12))
    plt.errorbar(feature_importance_df['Odds Ratio'], feature_importance_df['Feature'],
                xerr=[feature_importance_df['Odds Ratio'] - feature_importance_df['CI Lower'],
                    feature_importance_df['CI Upper'] - feature_importance_df['Odds Ratio']],
                fmt='o', ecolor='r', capsize=5, capthick=2)
    plt.axvline(x=1, color='gray', linestyle='--')
    plt.yticks(rotation=0)
    plt.ylabel('Features')
    plt.xlabel('Odds Ratio')
    plt.title('Odds Ratios with 95% Confidence Intervals')
    plt.tight_layout()
    plt.show()

    return coef, ci_lower_or, ci_upper_or, odds_ratios

# Feature_Importance()

def Evaluations():

    global y_val, y_pred_ft, X_train_ft, y_proba_ft, metrics, conf_matrix, correlation_matrix, auc 

    #Confusion Matrix
    conf_matrix = confusion_matrix(y_val, y_pred_ft)
    plt.figure(figsize=(4, 4)) 
    sns.set(font_scale=0.7)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Correlation matrix
    correlation_matrix = X_train_ft.corr() # TODO: what's the number output in the matrix? eg. pearson's r
    plt.figure(figsize=(12, 12))
    sns.set(font_scale=1)
    sns.heatmap(correlation_matrix, annot=True, cmap="YlGn", linewidths=.5)
    plt.title("Correlation Matrix Heatmap of All Variables")
    plt.legend(loc='best')
    plt.show()

    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y_val, y_proba_ft[:, 1])
    auc = roc_auc_score(y_val, y_proba_ft[:, 1])
    plt.figure(figsize=(12, 12))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()

    metrics = {
            'Accuracy Score': accuracy_score(y_val, y_pred_ft),
            'Precision Score': precision_score(y_val, y_pred_ft),
            'Recall Score': recall_score(y_val, y_pred_ft),
            'F1 Score': f1_score(y_val, y_pred_ft),
            'Log Loss': log_loss(y_val, y_proba_ft)
        }

    # Print metrics one line after the other
    for metric, value in metrics.items():
        print(f'{metric}: {value}')

    return metrics, conf_matrix, correlation_matrix, auc

# Evaluations()

def Preprocessing(explore, rename, split, log_eval):
    if explore == True:
        Data_Exploration(df)
    if rename == True:
        column_name = input('Column Name to Change')
        new_column_name = input(f'Original Column Name {column_name}. New Column Name: ')
        Rename_Col(df, column_name, new_column_name)
    if split == True:
        Cat_Encoder()
        Ttv_Split()
    if log_eval == True:
        Create_Model()
        Feature_Selection()
        Parameter_Tuning()
        Feature_Importance()
        Evaluations()

if __name__ == "__main__":
    Preprocessing(False, False, True, True)


# # TODO: Function to save outputs
# def save_outputs():