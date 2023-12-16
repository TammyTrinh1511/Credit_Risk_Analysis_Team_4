from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from joblib import Parallel, delayed  
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


def calculate_vif(i):
    return variance_inflation_factor(X_train.values, i)
            
def fit_model_VIF_gridsearchCV(file_directory=None):
    with timer('Load data'):
        df = pd.read_csv('/content/drive/MyDrive/input/dseb63_final_project_DP_dataset/dataframe_merged.csv')
        df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], inplace=True)
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
    with timer('Fill nan'):
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        X_train.columns = X_train.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)
    with timer('Feature selection by VIF with threshold = 10'):    
        # Calculate VIF in parallel with tqdm progress bar
        vif_factors = Parallel(n_jobs=2)(delayed(calculate_vif)(i) for i in tqdm(range(X_train.shape[1]), desc="Calculating VIF"))
        num_cores_used = joblib.num_cpus()
        print(f"Number of CPU cores used by joblib: {num_cores_used}")
        # Create a DataFrame with VIF factors and corresponding features
        vif = pd.DataFrame({'VIF Factor': vif_factors, 'Features': X_train.columns})
        if not os.path.exists("../results/"):
            os.makedirs("../results/")
        vif.to_csv('../results/VIF_data.csv')
        # Filter columns with VIF >= 10
        high_vif_columns = vif[vif['VIF Factor'] >= 10]['Features'].tolist()
        X_train = X_train.drop(columns=high_vif_columns)
        X_test = X_test.drop(columns=high_vif_columns)
        print(f'X_train shape: {X_train.shape}')
        print(f'X_test shape: {X_test.shape}')

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
        submit.to_csv('../results/VIF_gridsearchCV.csv', index=False)
    
if __name__ == '__main__':
    fit_model_VIF_gridsearchCV()