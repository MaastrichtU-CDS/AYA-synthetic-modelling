import logging
import os 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# Create model
# Feature selection
# Parameter tuning
# Feature importance

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
    

    plt.show()
    plt.close()

    return coef, ci_lower_or, ci_upper_or, odds_ratios, feature_importance, feature_importance_df, or_ci_plot
