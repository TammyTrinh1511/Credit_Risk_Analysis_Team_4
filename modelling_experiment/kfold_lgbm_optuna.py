from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import optuna

def fit_model_kfold_lgbm_optuna(file_directory=None):
    with timer('Load data'):
        df = pd.read_csv('../processed_data/dseb63_final_project_DP_dataset/dataframe_merged.csv')
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
        important_columns = feature_importance_stratified_lgbm(X_train, y_train)
        X_train = X_train[important_columns]
        X_test = X_test[important_columns]
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
    with timer('Optuna optimization for logistic regression: '):
        def objective(trial):
            class_0 = np.linspace(0.3, 0.95, 200)
            class_1 = np.linspace(4, 7, 200)
            hyperparameters = {
                'tol': trial.suggest_uniform('tol', 1e-6, 1e-3),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'newton-cg', 'saga']),
                'max_iter': trial.suggest_int('max_iter', 100, 2500),
                'C': trial.suggest_loguniform('C', 0.001, 100),
                'penalty': trial.suggest_categorical('penalty', ['l2']),
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                'random_state': trial.suggest_categorical('random_state', [0, 42, 2021, 555]),
                'n_jobs': -1,
                'warm_start': True,
                'class_weight': {
                 0: trial.suggest_float('class_weight_0', class_0.min(), class_0.max()),
                 1: trial.suggest_float('class_weight_1', class_1.min(), class_1.max())
                },
            }

            model = LogisticRegression(**hyperparameters)
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_train)[:, 1]
            roc_auc = roc_auc_score(y_train, preds)

            return roc_auc
        study = optuna.create_study(direction='maximize')
      
        with tqdm(total=7) as pbar:
            def update_pbar(study, trial):
                pbar.update(1)

        study.optimize(objective, n_trials=7, callbacks=[update_pbar])

        # Get the best hyperparameters
        best_params = study.best_params
        print("Best parameters found: ", best_params)
        best_params = {
            'tol': best_params['tol'],
            'solver': best_params['solver'],
            'max_iter': best_params['max_iter'],
            'C': best_params['C'],
            'penalty': best_params['penalty'],
            'fit_intercept': best_params['fit_intercept'],
            'random_state': best_params['random_state'],
            'class_weight': {0: best_params['class_weight_0'], 1: best_params['class_weight_1']},
            'n_jobs' : -1,
            'warm_start': True,
        }
        final_model = LogisticRegression(**best_params)
        final_model.fit(X_train, y_train)
        y_prob_train = final_model.predict_proba(X_train)[:,1]
        y_prob_test = final_model.predict_proba(X_test)[:,1]
        print(f'AUC: {roc_auc_score(y_train, y_prob_train)}')
    with timer('Submit and save the result: '):
        submit = test[['SK_ID_CURR']]
        submit.loc[:, 'TARGET'] = y_prob_test
        if not os.path.exists("../results/"):
            os.makedirs("../results/")
        submit.to_csv('../results/kfold_lgbm_optuna.csv', index=False)
    with timer('Important features:'):
        feat_importances_show(feature_names=columns, model=final_model, save_path='../results/kfold_lgbm_optuna.png')

if __name__ == '__main__':
    fit_model_kfold_lgbm_optuna()
