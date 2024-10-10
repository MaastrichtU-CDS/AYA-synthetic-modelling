import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, log_loss, classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_class_weight


# Import File
file_path = input("Please enter the file path: ")
df_og = pd.read_csv(file_path, delimiter=',', encoding='latin1')
df = df_og.copy()

#Data Exploration Function
def data_exploration(df):
    print(df.head())
    print(df.info())
    print(df.describe())

# data_exploration(df)

def rename_col(df, column_name, new_column_name): # can make into a dictionary so can do multiple columns at once
    df.rename(columns = {column_name:new_column_name}, inplace = True)
    return df

global X, y, X_train, X_test, X_val, y_val, y_train, y_test 
target =  input('Target name: ') #'BIneg_new'
X = df.drop(columns=[target])
y = df[target]


# Train Test Val Split Function
def ttv_split():

    global X, y, X_train, X_test, X_val, y_val, y_train, y_test 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    print('\n X_train shape:', X_train.shape, 'y_train shape:', y_train.shape, '\n X_test shape:', X_test.shape, 'y_test shape:', y_test.shape, '\n X_val shape:', X_val.shape, 'y_val shape:', y_val.shape)

    scaler = StandardScaler() 
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=df.drop(columns=[target]).columns)
    X_val = pd.DataFrame(X_val, columns=df.drop(columns=[target]).columns)
    X_test = pd.DataFrame(X_test, columns=df.drop(columns=[target]).columns)

    return X, y, X_train, X_test, X_val, y_val, y_train, y_test

# ttv_split()


def log_reg(): #grid search, elastic net

    global X_train, y_train, X_val, y_val, y_pred_ft, log_reg_e, X_train_ft, X_val_ft, y_proba_ft, log_reg_e_ft

    log_reg_e = LogisticRegression(penalty='elasticnet', solver ='saga', l1_ratio=0.5, C = 0.1)
    # C controls the sparsity: the smaller C the fewer features selected. 
    # With Lasso, the higher the alpha parameter, the fewer features selected.

    log_reg_e.fit(X_train, y_train)
    
    # Feature Selection
    model_log_features = SelectFromModel(log_reg_e, prefit=True)
    selected_features_mask = model_log_features.get_support()
    selected_features = X_train.columns[selected_features_mask]

    X_train_ft = pd.DataFrame(model_log_features.transform(X_train), columns=selected_features)
    X_val_ft = pd.DataFrame(model_log_features.transform(X_val), columns=selected_features)

    # # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    param_grid_elas = {
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],  # Range of l1_ratio values to search
        'C': [0.01, 0.1, 1, 10, 100],  # Range of regularization strengths. Smaller values, stronger regularization.
            }
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    grid_search = GridSearchCV(log_reg_e, param_grid=param_grid_elas, scoring=scoring_metrics, refit='accuracy', cv=5)
    grid_search.fit(X_train_ft, y_train) 
    best_params = grid_search.best_params_
    print("Best parameters found: ", best_params)

    # Refit the logistic regression model on X_train_ft
    log_reg_e_ft = LogisticRegression(class_weight=class_weight_dict, **best_params).fit(X_train_ft, y_train)
    print('Class weight', class_weight_dict)

    # Predictions
    y_pred_ft = log_reg_e_ft.predict(X_val_ft)
    y_proba_ft = log_reg_e_ft.predict_proba(X_val_ft)  # pred probability

    feature_importance = pd.Series(log_reg_e_ft.coef_.flatten(), index=selected_features)
   
    # The coefficients
    coef = log_reg_e_ft.coef_

    # Convert coefficients to odds ratios
    odds_ratios = np.exp(coef)
   
    # Create a DataFrame for better visualization 
    feature_importance_df = pd.DataFrame({
        'Feature': feature_importance.index,
        'Importance': feature_importance.values,
        'Coefficient': coef.flatten(),
        'Odds Ratio': odds_ratios.flatten()
    })
    print('\n', feature_importance_df)

    return X_train, y_train, X_val, y_val, y_pred_ft, log_reg_e, X_train_ft, X_val_ft, y_proba_ft, log_reg_e_ft

# log_reg()

def evaluations(log_reg):

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
    correlation_matrix = X_train_ft.corr()
    plt.figure(figsize=(20, 20))
    sns.set(font_scale=1)
    sns.heatmap(correlation_matrix, annot=True, cmap="YlGn", linewidths=.5)
    plt.title("Correlation Matrix Heatmap of All Variables")
    plt.legend(loc='best')
    plt.show()

    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y_val, y_proba_ft[:, 1])
    auc = roc_auc_score(y_val, y_proba_ft[:, 1])
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()

# evaluations(log_reg)

def preprocessing(explore, rename, split, log_eval):
    if explore == True:
        data_exploration(df)
    if rename == True:
        column_name = input('Column Name to Change')
        new_column_name = input(f'Original Column Name {column_name}. New Column Name: ')
        rename_col(df, column_name, new_column_name)
    if split == True:
        ttv_split()
    if log_eval == True:
        log_reg()
        evaluations(log_reg)

preprocessing(True, False, True, True)

