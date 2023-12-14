from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import re
from utils import * 
import numpy as np
import pandas as pd
import optuna

def fit_model_lgbm_gridsearchCV(): 
    with timer('Load data'):
        df = pd.read_csv('../processed_data/dataframe_merged.csv')
        df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], inplace=True)
        df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    with timer('Replacing inf with nan'):
        df = df.replace([np.inf, -np.inf], np.nan)
    with timer('Splitting into train and test'):
        train = df[df['TARGET'].notnull()]
        test = df[df['TARGET'].isnull()]
        X_train = train.drop(columns=['SK_ID_CURR', 'TARGET'])
        y_train = train['TARGET']
        X_test = test.drop(columns=['SK_ID_CURR', 'TARGET'])
        print(f'X_train shape: {X_train.shape}')
        print(f'X_test shape: {X_test.shape}')
    with timer('Feature selection'):
        drop_columns = feature_importance_lgbm(X_train, y_train)
        X_train.drop(columns=drop_columns, inplace=True)
        X_test.drop(columns=drop_columns, inplace=True)
        columns = X_train.columns
        print(f'X_train shape: {X_train.shape}')
        print(f'X_test shape: {X_test.shape}')
    with timer('Fill nan'):
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
    with timer('Scale by StandardScaler'):
        std = StandardScaler()
        X_train = std.fit_transform(X_train)
        X_test = std.transform(X_test)
    with timer('GridsearchCV for logistic regression: '):
        model = LogisticRegression()
        param_grid = { 'solver': ['liblinear'], 'penalty': ['l2'], 'C': [ 0.1, 1]}
        class_0 = np.linspace(0.3, 0.75, 3)
        class_1 = np.linspace(4.5, 6, 3)
        # Add class weights to the param_grid
        param_grid['class_weight'] = [{0: w0, 1: w1} for w0, w1 in zip(class_0, class_1)]

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid,verbose=10,scoring='roc_auc_ovr')
        grid_search.fit(X_train, y_train)
        print(f'Best param: {grid_search.best_params_}')
        y_prob_train = grid_search.predict_proba(X_train)[:,1]
        y_prob_test = grid_search.predict_proba(X_test)[:,1]
        print(f'AUC: {roc_auc_score(y_train, y_prob_train)}')
    with timer('Submit and save the result: '):
        submit = test[['SK_ID_CURR']]
        submit.loc[:, 'TARGET'] = y_prob_test
        if not os.path.exists("../results/"):
            os.makedirs("../results/")
        submit.to_csv('../results/lgbm_gridsearchCV.csv', index=False)
    with timer('Draw important features:'):
        feat_importances_show(feature_names=columns, model=final_model, save_path='../results/lgbm_gridsearchCV.png')

if __name__ == '__main__':
    fit_model_lgbm_gridsearchCV()