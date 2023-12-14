from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from lightgbm import early_stopping
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def feature_importance_stratified_lgbm(X_train, y_train, num_folds=5, random_state=42, verbose=False):
    important_columns = set()
    score = 1
    i = 1

    while score > 0.75:
        if verbose:
            print(f"Iteration {i}:")

        # removing the features which have been selected from the modelling data
        selection_data = X_train.drop(list(important_columns), axis=1)
        # defining the CV strategy
        fold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
        # reinitializing the score
        score = 0
        model_feature_importance = np.zeros_like(selection_data.columns)

        # doing K-Fold Cross validation
        for fold_num, (train_indices, val_indices) in enumerate(fold.split(selection_data, y_train), 1):
            if verbose:
                print(f"\t\tFitting fold {fold_num}")

            # defining the train and validation data
            x_train_fold = selection_data.iloc[train_indices]
            x_val_fold = selection_data.iloc[val_indices]
            y_train_fold = y_train.iloc[train_indices]
            y_val_fold = y_train.iloc[val_indices]

            # instantiating the LightGBM Classifier
            lg = LGBMClassifier(n_jobs=-1, random_state=random_state)
            lg.fit(x_train_fold, y_train_fold)

            # appending the feature importance of each feature averaged over different folds
            model_feature_importance += lg.feature_importances_ / num_folds
            # average k-fold ROC-AUC Score
            score += roc_auc_score(y_val_fold, lg.predict_proba(x_val_fold)[:, 1]) / num_folds

        # getting the non-zero feature importance columns
        imp_cols_indices = np.where(np.abs(model_feature_importance) > 0)
        # names of non-zero feature importance columns
        cols_imp = X_train.columns[imp_cols_indices]

        if score > 0.7:
            important_columns.update(cols_imp)
            if verbose:
                print(f"\tNo. of important columns kept = {len(important_columns)}")
        if verbose:
            print(f"\tCross Validation score = {score}")
        i += 1

    important_columns = list(important_columns)

    # Returning the important columns and their corresponding weights
    return important_columns

def feature_importance_lgbm(X_train, y_train):
    LIGHTGBM_PARAMS = {'boosting_type': 'goss', 'n_estimators': 10000, 'learning_rate': 0.005134,
        'num_leaves': 54, 'max_depth': 10, 'subsample_for_bin': 240000, 'reg_alpha': 0.436193,
        'reg_lambda': 0.479169, 'colsample_bytree': 0.508716, 'min_split_gain': 0.024766,
        'subsample': 1, 'is_unbalance': False,'silent':-1,'verbose':-1
    }
    feature_importances = np.zeros(X_train.shape[1])
    model = LGBMClassifier(**{**LIGHTGBM_PARAMS})
    for i in range(2):
        train_features, valid_features, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.25, random_state=i)
        model.fit(train_features, train_y, eval_set=[(valid_features, valid_y)], eval_metric='auc',
                  callbacks=[early_stopping(stopping_rounds=100)])

        feature_importances += model.feature_importances_

    feature_importances /= 2
    zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
    return zero_features


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def feat_importances_show(feature_names, model, num_features=20, figsize=(10, 15), save_path=None):
    '''
    Function to display the top most important features.

    Inputs:
        feature_names: numpy array
            Names of features of the training set
        model: sklearn model
            The trained model (e.g., LogisticRegression, RandomForestClassifier)
        num_features: int, default = 10
            Number of top features importances to display
        figsize: tuple, default = (10, 15)
            Size of the figure to be displayed
        save_path: str or None, default=None
            Path to save the figure. If None, the figure will not be saved.

    Returns:
        None
    '''

    # Getting the top features indices and their names
    feature_importance = np.abs(model.coef_[0])

    # Create a DataFrame to display feature names and their importance
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(num_features)

    # Plotting a horizontal bar plot of feature importances
    plt.figure(figsize=figsize)
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, orient='h')
    plt.title(f'Top {num_features} features as per classifier')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Names')
    plt.grid()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        print(f"Figure saved to: {save_path}")

    plt.show()
    print('=' * 100)